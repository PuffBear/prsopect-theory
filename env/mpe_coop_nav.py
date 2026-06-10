"""
MPE Cooperative Navigation environment wrapper for PettingZoo.

Wraps MPE simple_spread (Cooperative Navigation) into the same interface used
by the or-gym experiments:
  - PettingZoo ParallelEnv with continuous actions
  - Compatible with CPTRewardWrapper (for CPT transform AFTER w-blending)
  - Compatible with SuperSuit vectorisation for SB3

Handles the MPE package split:
  - PettingZoo <= 1.23:  from pettingzoo.mpe import simple_spread_v3
  - PettingZoo >= 1.24:  pip install mpe separately, then:
                          from mpe import simple_spread_v3

INSTALL (for PettingZoo 1.26.1):
  pip install mpe          # standalone MPE package
  # do NOT downgrade pettingzoo -- that breaks ScaledBaseStockWrapper

Collapse definition:
  mean cumulative true reward per episode < MPE_COLLAPSE_REWARD_THRESHOLD
  Calibrate threshold via pilot: run w=0.0 (fully local, worst) and w=1.0
  (fully global, best) across 5 seeds and pick midpoint.

=== REWARD STRUCTURE ===

simple_spread gives every agent the same global team reward:
    r_global = -sum_over_landmarks(min_dist(landmark, agents))

We need a per-agent LOCAL reward to make the w-dial meaningful:
    r_local(i) = -distance(agent_i, nearest_landmark)

The w-blending wrapper (MPELocalRewardWrapper) computes:
    r_blend(i) = (1-w) * r_local(i) + w * r_global

THEN CPTRewardWrapper applies the PT value function to r_blend(i).

Wrapper stack (innermost first):
    simple_spread_v3.parallel_env
    -> MPELocalRewardWrapper  (defines r_local, blends with w)
    -> CPTRewardWrapper       (applies alpha/beta/lambda)
    -> SuperSuit vectorisation
"""
import os
import numpy as np
from pettingzoo.utils import BaseParallelWrapper

# Handle the MPE package split across PettingZoo versions
try:
    from pettingzoo.mpe import simple_spread_v3        # pettingzoo <= 1.23 (bundled)
except (ImportError, ModuleNotFoundError):
    try:
        from mpe2 import simple_spread_v3              # Farama successor for pettingzoo >= 1.25
    except (ImportError, ModuleNotFoundError):
        try:
            from mpe import simple_spread_v3           # older standalone package
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "MPE environment not found.\n"
                "For PettingZoo 1.26.x: pip install mpe2\n"
                "For PettingZoo <= 1.23: pip install 'pettingzoo[mpe]'"
            )


def make_mpe_coop_nav(n_agents=3, max_cycles=25, continuous_actions=True):
    """Create a raw MPE Cooperative Navigation (simple_spread) parallel env."""
    env = simple_spread_v3.parallel_env(
        N=n_agents,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
        local_ratio=0.5,   # raw env default; we override reward below anyway
    )
    return env


class MPELocalRewardWrapper(BaseParallelWrapper):
    """
    Replaces the shared global reward of simple_spread with per-agent blended
    rewards that interpolate between a local signal and the global signal.

    r_local(i) = -distance(agent_i, nearest_landmark)   [agent-specific]
    r_global   = sum over landmarks of -min dist(landmark, agents)  [team score]

    r_blend(i) = (1 - w) * r_local(i) + w * r_global

    The blended reward is stored and returned; CPTRewardWrapper (applied on top)
    will apply the PT value function to r_blend(i).

    Requires access to the underlying MPE world object to read agent/landmark
    positions. Works with simple_spread_v3 from pettingzoo <= 1.23 and the
    standalone mpe package.
    """

    def __init__(self, env, w):
        """
        Args:
            env: raw parallel MPE env (simple_spread_v3.parallel_env)
            w:   reward centralization weight in [0,1]
                 0 = fully local (each agent only cares about its nearest landmark)
                 1 = fully global (all agents share the team score)
        """
        super().__init__(env)
        self.w = float(w)

    def _get_world(self):
        """Find the MPE world object (agents/landmarks with .state.p_pos)."""
        # Most robust: PettingZoo envs expose .unwrapped to the base env, which
        # for simple_spread carries the .world attribute (verified on mpe2 1.1.0).
        base = getattr(self.env, "unwrapped", self.env)
        if hasattr(base, "world"):
            return base.world
        # Fallback: walk the .env chain.
        inner = self.env
        while hasattr(inner, "env"):
            inner = inner.env
            if hasattr(inner, "world"):
                return inner.world
        return None

    def _local_reward(self, agent_idx, world):
        """
        r_local(i) = -distance(agent_i, its nearest landmark).
        Returns 0.0 if world state is unavailable (graceful fallback).
        """
        if world is None:
            return 0.0
        try:
            a_pos = world.agents[agent_idx].state.p_pos
            dists = [
                float(np.linalg.norm(a_pos - lm.state.p_pos))
                for lm in world.landmarks
            ]
            return -min(dists)
        except (IndexError, AttributeError):
            return 0.0

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)

        world = self._get_world()

        # Global team reward: sum of per-landmark min-distances (negated)
        # In simple_spread all agents receive the same value; take from any.
        r_global = sum(rewards.values()) / max(len(rewards), 1)

        new_rewards = {}
        for idx, agent in enumerate(self.possible_agents):
            if agent not in rewards:
                continue
            r_local = self._local_reward(idx, world)
            r_blend = (1.0 - self.w) * r_local + self.w * r_global

            new_rewards[agent] = r_blend

            # Populate info for diagnostics
            if agent not in info:
                info[agent] = {}
            info[agent]["r_local"] = r_local
            info[agent]["r_global"] = r_global
            info[agent]["r_blend"] = r_blend
            info[agent]["true_reward"] = rewards[agent]  # raw MPE reward

        return obs, new_rewards, term, trunc, info


def make_mpe_env_with_w(w, n_agents=3, max_cycles=25):
    """
    Convenience factory: raw MPE -> MPELocalRewardWrapper(w).
    Pass the result to CPTRewardWrapper for the PT transform.
    """
    raw = make_mpe_coop_nav(n_agents=n_agents, max_cycles=max_cycles,
                            continuous_actions=True)
    return MPELocalRewardWrapper(raw, w=w)


# --- Collapse definition for MPE -------------------------------------------
# theta is DATA-DRIVEN: calibrate with exp_mpe_calibrate.py (trained w=0 vs w=1),
# then analyze_mpe_calibrate.py writes docs/exp_mpe/mpe_collapse_threshold.json.
# is_collapsed_mpe loads that file when present; the default below is only a
# pre-calibration placeholder (do NOT trust collapse labels until the json exists).
import json as _json

_THRESHOLD_JSON = os.path.join(os.path.dirname(__file__), "..",
                               "docs", "exp_mpe", "mpe_collapse_threshold.json")
_DEFAULT_THRESHOLD = -30.0  # placeholder; overridden once calibration json exists


def load_collapse_threshold(default=_DEFAULT_THRESHOLD):
    """Return the calibrated theta from the json if present, else the default."""
    try:
        with open(_THRESHOLD_JSON) as f:
            return float(_json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return float(default)


# loaded at import for convenience; analysis code should prefer load_collapse_threshold()
MPE_COLLAPSE_REWARD_THRESHOLD = load_collapse_threshold()
THRESHOLD_IS_CALIBRATED = os.path.exists(_THRESHOLD_JSON)


def is_collapsed_mpe(mean_reward, threshold=None):
    """
    Collapse definition for MPE Cooperative Navigation.

    Args:
        mean_reward: mean cumulative TRUE reward (from info['true_reward']) over
                     eval episodes. Negative; lower is worse.
        threshold:   override theta; if None, uses the calibrated value (json) or
                     the placeholder default.

    Returns:
        True if the run is classified as collapsed.
    """
    theta = load_collapse_threshold() if threshold is None else threshold
    return bool(mean_reward <= theta)


# --- Threshold calibration helper ------------------------------------------
def calibrate_collapse_threshold(n_episodes=20, seed=42):
    """
    Run random-policy at w=0 and perfect-w at w=1 to bracket the threshold.
    Print suggested MPE_COLLAPSE_REWARD_THRESHOLD.

    Usage:
        python -c "from env.mpe_coop_nav import calibrate_collapse_threshold; calibrate_collapse_threshold()"
    """
    import sys
    results = {}
    for w_cal in [0.0, 1.0]:
        env = make_mpe_coop_nav()
        rewards_ep = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_r = 0.0
            while env.agents:
                actions = {a: env.action_space(a).sample() for a in env.agents}
                obs, rews, term, trunc, info = env.step(actions)
                ep_r += sum(
                    info[a].get("true_reward", rews.get(a, 0.0))
                    for a in rews
                )
            rewards_ep.append(ep_r)
        results[w_cal] = float(np.mean(rewards_ep))
        print(f"  w={w_cal}: mean_reward = {results[w_cal]:.2f}", file=sys.stderr)
    threshold = (results[0.0] + results[1.0]) / 2.0
    print(f"\nSuggested MPE_COLLAPSE_REWARD_THRESHOLD = {threshold:.1f}", file=sys.stderr)
    return threshold
