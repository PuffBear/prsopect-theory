"""
Discrete(K)-per-agent action shell for value-based learners (QMIX / IQL).

The supply-chain substrate has a continuous Box action (order quantity), and the
PPO/MAPPO pipeline maps a normalized continuous action in [-1, 1] to a target
base-stock level S in [low_bound, high_bound] via ScaledBaseStockWrapper. Value-
based learners need a DISCRETE action space, so this wrapper bins S into K levels
and exposes Discrete(K) per agent.

PLACEMENT (plan section 3, supply-chain pipeline, OUTER -> INNER):
    GlobalStateWrapper                              (joint state for the mixer)
      -> DiscreteBaseStockWrapper (THIS)            index -> normalized [-1,1] action
        -> CPTRewardWrapper (blend w, then CPT)
          -> ScaledBaseStockWrapper                 [-1,1] -> S in [0,500] -> order qty
            -> NetworkManagement-v1

The K levels are S in {0, 500/(K-1), ..., 500} (K=11 -> {0,50,...,500}). Index k
is converted to the normalized action a = 2*(S/range) - 1 that ScaledBaseStock
already understands, so NOTHING downstream changes -- the discretization is the
ONLY new thing relative to the continuous runs. This keeps the reward pipeline
and collapse label byte-identical (the value-based cells test paradigm + action-
space jointly; the discretized-PPO control in the plan isolates the two).

Each agent's underlying Box action may be multi-dimensional (a node with several
suppliers); ScaledBaseStockWrapper only reads action[0] to set a single base-
stock S, so we emit a length-`shape` vector with the mapped value in slot 0 and
zeros elsewhere -- identical to how the continuous policy's action[0] is used.
"""
import numpy as np
import gymnasium as gym
from pettingzoo.utils import BaseParallelWrapper


class DiscreteBaseStockWrapper(BaseParallelWrapper):
    """Expose Discrete(K) per agent; map index -> normalized [-1,1] base-stock action."""

    def __init__(self, env, K=11):
        super().__init__(env)
        self.K = int(K)
        # K base-stock fractions evenly spaced in [0,1] -> normalized action in [-1,1]
        self._levels = np.linspace(0.0, 1.0, self.K, dtype=np.float64)  # fraction of range
        self._norm = 2.0 * self._levels - 1.0                          # [-1,1] for ScaledBaseStock
        self.action_spaces = {a: gym.spaces.Discrete(self.K)
                              for a in self.possible_agents}
        # remember the underlying continuous action shape per agent (>=1 dims)
        self._cont_shape = {a: int(np.prod(self.env.action_space(a).shape))
                            for a in self.possible_agents}

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def _to_continuous(self, agent, k):
        """Discrete index k -> length-`shape` vector, mapped value in slot 0."""
        k = int(np.asarray(k).item()) if np.ndim(k) else int(k)
        k = max(0, min(self.K - 1, k))
        vec = np.zeros(self._cont_shape[agent], dtype=np.float64)
        vec[0] = self._norm[k]            # ScaledBaseStockWrapper reads action[0]
        return vec

    def step(self, actions):
        cont = {a: self._to_continuous(a, k) for a, k in actions.items()}
        return self.env.step(cont)
