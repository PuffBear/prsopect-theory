"""
RWARE (Robot Warehouse) environment wrapper with incentive-misalignment w-dial.

Wraps the `rware` gymnasium environment into a PettingZoo ParallelEnv, then
applies the same w-blending + CPT stack used for the supply-chain experiments.

=== INCENTIVE MISALIGNMENT STRUCTURE ===

RWARE agents must (a) pick up a requested shelf and (b) deliver it to a goal
zone to earn a delivery reward.  The local incentive we impose is:

    r_local(i) = CARRY_BONUS  if  agent i is currently carrying any shelf
                 0.0           otherwise

Carrying is individually rational (you are "doing work") but never earns
delivery reward by itself.  At w=0 (pure local), agents learn to always carry a
shelf -- they pick one up and never put it down.  Shelves tied up in transit are
not available for re-requesting, and the delivery count collapses to zero.

This is exactly the supply-chain hoarding analogue:
  supply chain:  hoard stock at your node  ->  downstream nodes starve
  RWARE:         carry shelf indefinitely  ->  warehouse throughput collapses

At w=1 (pure global = delivery reward), agents learn the full pick-up-navigate-
deliver cycle.  A sharp phase transition in delivery rate appears at some w*.

=== WRAPPER STACK ===

    rware gymnasium env
      -> RWAREParallelWrapper     (gymnasium -> PettingZoo parallel API)
      -> RWARELocalRewardWrapper  (defines r_local, blends with w)
      -> CPTRewardWrapper         (applies alpha/beta/lambda)
      -> SuperSuit vectorisation

=== INSTALL ===

    pip install rware

=== COLLAPSE DEFINITION ===

    mean cumulative TRUE reward (deliveries) per episode  <  RWARE_COLLAPSE_THRESHOLD
    Default threshold: 1.0 (fewer than one delivery per episode = collapsed).
    Calibrate with rware_calibrate.py after pilot runs.
"""
import os
import json as _json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import BaseParallelWrapper

try:
    import rware as _rware_module  # noqa: F401 -- verifies install
except ImportError:
    raise ImportError(
        "RWARE not found.  Install with:  pip install rware\n"
        "Then verify:  python -c 'import rware'"
    )

# How much reward an agent earns PER STEP while carrying a shelf.
# Calibrated so its per-episode scale is comparable to the delivery reward:
# typical delivery rate ~0.05/step * 500 steps = ~25 deliveries per episode.
# Carrying reward of 0.1/step * 500 steps = ~50 (if always carrying).
# The ratio is ~2, which ensures the local signal is learnable but global can
# dominate at moderate w.  Adjust if needed post-calibration.
CARRY_BONUS = 0.1


# ---------------------------------------------------------------------------
# Step 1: PettingZoo parallel-env wrapping the gymnasium RWARE env
# ---------------------------------------------------------------------------

class RWAREParallelWrapper(ParallelEnv):
    """
    Thin PettingZoo ParallelEnv adapter for the `rware` gymnasium environment.

    - Converts tuple obs / list rewards / scalar done to the dict-based
      PettingZoo parallel API.
    - Exposes the underlying gymnasium env at self._gym_env so that wrappers
      above can access agent state (carrying_shelf, etc.).
    - Per-agent observation and action spaces match what RWARE returns.
    """

    metadata = {"render_modes": [], "name": "rware_parallel_v1"}
    render_mode = None   # required by SuperSuit's MarkovVectorEnv

    def __init__(self, env_id="rware-tiny-3ag-easy-v2", max_cycles=500):
        super().__init__()
        self._gym_env = gym.make(env_id)
        inner = self._gym_env.unwrapped
        self._n_agents = inner.n_agents
        self.max_cycles = max_cycles
        self._step_count = 0

        self.possible_agents = [f"agent_{i}" for i in range(self._n_agents)]
        self.agents = self.possible_agents[:]

        # Build observation and action spaces from the gymnasium tuple spaces
        gym_obs_space = self._gym_env.observation_space
        gym_act_space = self._gym_env.action_space
        self.observation_spaces = {
            a: gym_obs_space[i] for i, a in enumerate(self.possible_agents)
        }
        self.action_spaces = {
            a: gym_act_space[i] for i, a in enumerate(self.possible_agents)
        }

    # PettingZoo API ----------------------------------------------------------

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        obs_tuple, info = self._gym_env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self._step_count = 0
        obs_dict = {a: obs_tuple[i] for i, a in enumerate(self.possible_agents)}
        info_dict = {a: (info[i] if isinstance(info, (list, tuple)) else {})
                     for i, a in enumerate(self.possible_agents)}
        return obs_dict, info_dict

    def step(self, actions):
        action_list = [
            actions.get(a, 0) for a in self.possible_agents
        ]
        obs_tuple, rewards_list, term, trunc, info = self._gym_env.step(action_list)
        self._step_count += 1

        # RWARE returns a single bool for term/trunc (not per-agent).
        # Treat max_cycles as the episode-end condition.
        done = bool(term or trunc or self._step_count >= self.max_cycles)

        obs_dict = {a: obs_tuple[i] for i, a in enumerate(self.possible_agents)}
        rew_dict = {a: float(rewards_list[i]) for i, a in enumerate(self.possible_agents)}
        term_dict = {a: done for a in self.possible_agents}
        trunc_dict = {a: False for a in self.possible_agents}
        info_dict = {a: {} for a in self.possible_agents}

        if done:
            self.agents = []

        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

    def render(self):
        return self._gym_env.render()

    def close(self):
        self._gym_env.close()


# ---------------------------------------------------------------------------
# Step 2: Local reward wrapper (carrying incentive + w-blending)
# ---------------------------------------------------------------------------

class RWARELocalRewardWrapper(BaseParallelWrapper):
    """
    Implements the carrying-based local reward and the w-dial.

    r_local(i) = CARRY_BONUS  if  agent i is carrying a shelf this step
                 0.0          otherwise

    r_global   = mean delivery reward across all agents

    r_blend(i) = (1 - w) * r_local(i)  +  w * r_global

    THEN CPTRewardWrapper applies the PT value function to r_blend(i).

    The raw delivery reward (r_global computed from base env) is stored in
    info[agent]["true_reward"] so the evaluation metric is always the number
    of deliveries, not the blended reward.
    """

    def __init__(self, env, w):
        """
        Args:
            env: RWAREParallelWrapper (or further wrapped parallel env)
            w:   reward centralization weight in [0, 1]
                 0 = pure carrying reward (hoarding -> collapse)
                 1 = pure delivery reward (coordination -> good throughput)
        """
        super().__init__(env)
        self.w = float(w)

    def _get_inner(self):
        """Walk the wrapper chain to find the RWAREParallelWrapper."""
        inner = self.env
        while hasattr(inner, "env"):
            if isinstance(inner, RWAREParallelWrapper):
                return inner
            inner = inner.env
        if isinstance(inner, RWAREParallelWrapper):
            return inner
        return None

    def _carrying(self, agent_idx):
        """Return True if agent agent_idx is currently carrying a shelf."""
        base = self._get_inner()
        if base is None:
            return False
        try:
            gym_inner = base._gym_env.unwrapped
            return gym_inner.agents[agent_idx].carrying_shelf is not None
        except (IndexError, AttributeError):
            return False

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)

        # Global = mean delivery reward (number of deliveries this step, shared)
        r_global = sum(rewards.values()) / max(len(rewards), 1)

        new_rewards = {}
        for idx, agent in enumerate(self.possible_agents):
            if agent not in rewards:
                continue

            r_local = CARRY_BONUS if self._carrying(idx) else 0.0
            r_blend = (1.0 - self.w) * r_local + self.w * r_global

            new_rewards[agent] = r_blend

            if agent not in info:
                info[agent] = {}
            # Store raw delivery reward as true_reward for evaluation/collapse metric
            info[agent]["true_reward"] = rewards[agent]  # raw delivery reward
            info[agent]["r_local"] = r_local
            info[agent]["r_global"] = r_global
            info[agent]["r_blend"] = r_blend
            info[agent]["is_carrying"] = self._carrying(idx)

        return obs, new_rewards, term, trunc, info


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def make_rware_env(env_id="rware-tiny-3ag-easy-v2", max_cycles=500):
    """Create a raw RWARE PettingZoo parallel env (no w-dial)."""
    return RWAREParallelWrapper(env_id=env_id, max_cycles=max_cycles)


def make_rware_env_with_w(w, env_id="rware-tiny-3ag-easy-v2", max_cycles=500):
    """
    Convenience factory: RWAREParallelWrapper -> RWARELocalRewardWrapper(w).
    Pass the result to CPTRewardWrapper for the PT transform.
    """
    base = make_rware_env(env_id=env_id, max_cycles=max_cycles)
    return RWARELocalRewardWrapper(base, w=w)


# ---------------------------------------------------------------------------
# Collapse definition
# ---------------------------------------------------------------------------

_THRESHOLD_JSON = os.path.join(os.path.dirname(__file__), "..",
                               "docs", "exp_rware", "rware_collapse_threshold.json")
_DEFAULT_THRESHOLD = 1.0  # < 1 delivery per episode = collapsed


def load_rware_collapse_threshold(default=_DEFAULT_THRESHOLD):
    try:
        with open(_THRESHOLD_JSON) as f:
            return float(_json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return float(default)


RWARE_COLLAPSE_THRESHOLD = load_rware_collapse_threshold()


def is_collapsed_rware(mean_reward, threshold=None):
    """
    Collapse definition for RWARE.

    Args:
        mean_reward: mean cumulative TRUE reward (deliveries) per eval episode.
    Returns:
        True if fewer than `threshold` deliveries per episode on average.
    """
    theta = load_rware_collapse_threshold() if threshold is None else threshold
    return bool(mean_reward < theta)
