"""
Global-state observation augmentation for centralized-critic MAPPO (CTDE).

Each agent's observation is augmented to:

    own_part(i)      = [ own_obs(i) | onehot_id(i) ]
    augmented_obs(i) = [ own_part(i) | global_state ]
    global_state     = concat_j own_part(j)   for j in possible_agents (fixed order)

The one-hot agent ID is REQUIRED for parameter-shared MAPPO: a single shared
actor fed symmetric egocentric observations produces symmetric actions and
cannot specialize agents to distinct landmarks (Yu et al. 2022). Without it the
centralized critic still leaves coverage stuck (~0.24 occupancy on MPE).

The actor (decentralized execution) consumes only the first `local_dim` entries
(own_part = own_obs + id); the centralized critic consumes the remainder
(global_state). This is
the standard SB3-compatible way to get a centralized critic without leaving the
single-observation vector-env framework: a custom policy slices the vector.

This wrapper sits OUTERMOST in the parallel stack (after the reward wrappers) so
it only touches observations -- rewards and info (incl. true_reward / r_global /
occupancy used by evaluation) pass through unchanged.
"""
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper


class GlobalStateWrapper(BaseParallelWrapper):
    """Append the joint observation (global state) to every agent's obs."""

    def __init__(self, env):
        super().__init__(env)
        self._agents = list(self.possible_agents)
        self._n = len(self._agents)
        self._idx = {a: i for i, a in enumerate(self._agents)}
        # base obs dim from any agent's existing obs space (homogeneous agents)
        base_space = self.env.observation_space(self._agents[0])
        self._base_dim = int(np.prod(base_space.shape))
        # own_part = base obs + one-hot agent id  ->  this is the actor input
        self.local_dim = self._base_dim + self._n
        # augmented = own_part + global_state(= concat of all own_parts)
        aug_dim = self.local_dim * (1 + self._n)
        low = -np.inf * np.ones(aug_dim, dtype=np.float32)
        high = np.inf * np.ones(aug_dim, dtype=np.float32)
        self._aug_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # local_dim is read by the MAPPO policy to slice actor vs critic input
    def observation_space(self, agent):
        return self._aug_space

    def action_space(self, agent):
        return self.env.action_space(agent)

    def _own_part(self, agent, obs):
        """[ base_obs | one-hot agent id ]  (dim = local_dim)."""
        base = np.asarray(obs, dtype=np.float32).ravel()
        onehot = np.zeros(self._n, dtype=np.float32)
        onehot[self._idx[agent]] = 1.0
        return np.concatenate([base, onehot], axis=0)

    def _global_state(self, parts_by_agent):
        """Concat every agent's own_part in fixed order (zeros if absent)."""
        zeros = np.zeros(self.local_dim, dtype=np.float32)
        return np.concatenate(
            [parts_by_agent.get(a, zeros) for a in self._agents], axis=0)

    def _augment(self, obs_dict):
        parts = {a: self._own_part(a, o) for a, o in obs_dict.items()}
        g = self._global_state(parts)
        return {a: np.concatenate([parts[a], g], axis=0).astype(np.float32)
                for a in obs_dict}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        return self._augment(obs), rewards, term, trunc, info
