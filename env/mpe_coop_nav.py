"""
MPE Cooperative Navigation environment wrapper for PettingZoo.

Wraps MPE simple_spread (Cooperative Navigation) into the same interface used
by the or-gym experiments:
  - PettingZoo ParallelEnv with continuous actions
  - Compatible with CPTRewardWrapper (for CPT transform AFTER w-blending)
  - Compatible with SuperSuit vectorisation for SB3

Handles the MPE package split:
  - PettingZoo <= 1.23:  from pettingzoo.mpe import simple_spread_v3
  - PettingZoo >= 1.24:  pip install mpe2, then:
                          from mpe2 import simple_spread_v3

INSTALL (for PettingZoo 1.26.1):
  pip install mpe2         # Farama standalone MPE package
  # do NOT downgrade pettingzoo -- that breaks ScaledBaseStockWrapper

=== REWARD STRUCTURE ===

simple_spread gives every agent the same global team reward:
    r_global = -sum_over_landmarks(min_dist(landmark, agents))

The local reward that creates incentive misalignment:
    r_local(i) = -HERDING_SCALE * mean_distance(agent_i, other_agents)

Herding instinct: staying near teammates is individually rational
(you can locally coordinate) but collectively disastrous (all agents
cluster at the same location, leaving landmarks uncovered).  This is the
navigation analogue of supply-chain hoarding.

The w-blending wrapper (MPELocalRewardWrapper) computes:
    r_blend(i) = (1-w) * r_local(i) + w * r_global

THEN CPTRewardWrapper applies the PT value function to r_blend(i).

=== HERDING_SCALE CALIBRATION ===

Without scaling (HERDING_SCALE=1), the herding signal is O(1) while the
global team score is O(3-4).  The two equilibria are equal at w ≈ 0.34,
which places the phase transition at the very bottom of any reasonable
W_LIST and gives a near-step-function rather than a sigmoid.

HERDING_SCALE=4 is calibrated to put the theoretical indifference point
at w* ≈ 0.62, safely in the middle of W_LIST = [0.10..0.90].

Derivation (arena scale ≈ 1.2 units, 3 agents, 3 landmarks):
  clustered: r_herd ≈ -0.1 × scale,  r_global ≈ -3.5
  spread:    r_herd ≈ -1.2 × scale,  r_global ≈ -0.4
  indifference:
    (1-w) × scale × 1.1 = w × 3.1
    w* = scale × 1.1 / (scale × 1.1 + 3.1)
       ≈ 0.62 for scale=4

=== COLLAPSE DEFINITION (v2 — occupancy-based) ===

  PRIMARY metric: mean per-step landmark OCCUPANCY (fraction of landmarks
  with an agent within OCCUPANCY_RADIUS=0.3), averaged over eval episodes.
  collapsed := mean_occupancy < theta_occ (calibrated midpoint).

  WHY NOT episode r_global? Scripted-policy probe (2026-06-13, 20 seeds):
    spread (greedy-to-landmark):  r_global ≈ -21/ep
    still (no-op):                r_global ≈ -46/ep
    cluster (greedy-to-centroid): r_global ≈ -48/ep
    random:                       r_global ≈ -46/ep
  The collapse/coordination gap in r_global (-48 vs -21) is only ~1.5x the
  seed-level std of trained runs (±18), and a no-op policy sits at -46 —
  indistinguishable from full herding collapse. The earlier docstring ranges
  (-200..-250 vs -20..-35) were derived from a miscalibrated model of the
  arena scale and are WRONG. Occupancy separates the same probe policies as
  ~0.0 (cluster/still) vs ~0.8-1.0 (spread) — a wide, noise-robust gap.
  r_global is still logged as a secondary continuous metric.

  Theoretical w* from the measured scripted returns (cluster vs spread,
  herding_scale=4): (1-w)*42.3 = w*26.3  ->  w* ≈ 0.62. Design intact.

  Calibrate threshold: run exp_mpe_calibrate.py (w=0 vs w=1 endpoints)
  then analyze_mpe_calibrate.py writes
  docs/exp_mpe/mpe_collapse_threshold_v2.json. The main sweep MUST be gated
  on clean_separation=true in that file (this gate was violated in the v1
  run and burned ~1000 jobs).

=== WRAPPER STACK ===

    simple_spread_v3.parallel_env  (local_ratio=0.0 -- REQUIRED, see below)
      -> MPELocalRewardWrapper     (r_local=herding × scale, w-blend)
      -> CPTRewardWrapper          (alpha/beta/lambda)
      -> SuperSuit vectorisation

NOTE: local_ratio MUST be 0.0 in the raw env.  If it is 0.5 (the default),
r_global in the wrapper is contaminated with the nearest-landmark term and
the true_reward evaluation metric is wrong.  This wrapper enforces local_ratio=0
via make_mpe_coop_nav().
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
        local_ratio=0.0,   # pure global team score; our wrapper defines r_local separately
    )
    return env


HERDING_SCALE = 4.0   # see module docstring for calibration rationale


class MPELocalRewardWrapper(BaseParallelWrapper):
    """
    Replaces the shared global reward of simple_spread with per-agent blended
    rewards that interpolate between a local herding signal and the global
    team-coverage score.

    r_local(i) = -herding_scale * mean_dist(agent_i, other_agents)
    r_global   = -sum_lm(min_dist(landmark, agents))  [pure team score]

    r_blend(i) = (1 - w) * r_local(i) + w * r_global

    herding_scale (default 4.0) places the theoretical phase-transition at
    w* ≈ 0.62, centred in W_LIST = [0.10..0.90].  See module docstring.

    The raw simple_spread reward (true_reward) is stored in
    info[agent]["true_reward"] for evaluation / collapse labelling.
    CPTRewardWrapper preserves this key (does not overwrite it).
    """

    def __init__(self, env, w, herding_scale=HERDING_SCALE):
        """
        Args:
            env:           raw parallel MPE env (local_ratio MUST be 0.0)
            w:             centralization weight in [0,1]
                           0 = fully local herding  -> collapse
                           1 = fully global team score -> coordination
            herding_scale: multiplier on the herding distance signal (default 4.0)
        """
        super().__init__(env)
        self.w = float(w)
        self.herding_scale = float(herding_scale)

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
        r_local(i) = -herding_scale * mean_distance(agent_i, other_agents).

        Individually rational: staying close to teammates minimises this penalty.
        Collectively disastrous: all agents converge; no landmark gets covered.
        Exact navigation analogue of supply-chain hoarding.

        herding_scale=4.0 calibrates the signal magnitude so the theoretical
        phase-transition w* ≈ 0.62 (see module docstring).

        Returns 0.0 if world state is unavailable (graceful fallback).
        """
        if world is None:
            return 0.0
        try:
            a_pos = world.agents[agent_idx].state.p_pos
            others = [world.agents[j].state.p_pos
                      for j in range(len(world.agents)) if j != agent_idx]
            if not others:
                return 0.0
            raw = -float(np.mean([np.linalg.norm(a_pos - o) for o in others]))
            return self.herding_scale * raw
        except (IndexError, AttributeError):
            return 0.0

    OCCUPANCY_RADIUS = 0.3   # landmark counts as covered if any agent within this

    def _occupancy(self, world):
        """Fraction of landmarks covered (min agent dist < OCCUPANCY_RADIUS)."""
        if world is None:
            return float("nan")
        try:
            covered = 0
            for lm in world.landmarks:
                d = min(np.linalg.norm(lm.state.p_pos - ag.state.p_pos)
                        for ag in world.agents)
                if d < self.OCCUPANCY_RADIUS:
                    covered += 1
            return covered / max(len(world.landmarks), 1)
        except (IndexError, AttributeError):
            return float("nan")

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)

        world = self._get_world()
        occupancy = self._occupancy(world)

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
            info[agent]["occupancy"] = occupancy   # fraction of landmarks covered
            info[agent]["true_reward"] = rewards[agent]  # raw MPE reward

        return obs, new_rewards, term, trunc, info


def make_mpe_env_with_w(w, n_agents=3, max_cycles=25,
                        herding_scale=HERDING_SCALE):
    """
    Convenience factory: raw MPE -> MPELocalRewardWrapper(w, herding_scale).
    Pass the result to CPTRewardWrapper for the PT transform.
    """
    raw = make_mpe_coop_nav(n_agents=n_agents, max_cycles=max_cycles,
                            continuous_actions=True)
    return MPELocalRewardWrapper(raw, w=w, herding_scale=herding_scale)


# --- Collapse definition for MPE -------------------------------------------
# theta is DATA-DRIVEN: calibrate with exp_mpe_calibrate.py (trained w=0 vs w=1),
# then analyze_mpe_calibrate.py writes docs/exp_mpe/mpe_collapse_threshold.json.
# is_collapsed_mpe loads that file when present; the default below is only a
# pre-calibration placeholder (do NOT trust collapse labels until the json exists).
import json as _json

_THRESHOLD_JSON = os.path.join(os.path.dirname(__file__), "..",
                               "docs", "exp_mpe", "mpe_collapse_threshold.json")
_DEFAULT_THRESHOLD = -100.0  # placeholder (order-of-magnitude estimate post herding_scale=4 fix)
# Expected post-calibration: w=0 episode true_reward ≈ -200...-250,
#                            w=1 episode true_reward ≈ -20...-30, midpoint ≈ -110...-140.
# Run exp_mpe_calibrate.py then analyze_mpe_calibrate.py to get the exact value.


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


# --- v2 occupancy-based collapse (PRIMARY) ----------------------------------
_THRESHOLD_JSON_V2 = os.path.join(os.path.dirname(__file__), "..",
                                  "docs", "exp_mpe",
                                  "mpe_collapse_threshold_v2.json")
_DEFAULT_OCC_THRESHOLD = 0.40  # placeholder; calibrate before trusting labels


def load_occupancy_threshold(default=_DEFAULT_OCC_THRESHOLD):
    """Return calibrated occupancy theta from v2 json if present, else default."""
    try:
        with open(_THRESHOLD_JSON_V2) as f:
            return float(_json.load(f)["threshold_occupancy"])
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return float(default)


def is_collapsed_mpe_occupancy(mean_occupancy, threshold=None):
    """
    v2 collapse definition: mean per-step landmark occupancy below theta.
    mean_occupancy in [0,1]; lower is worse.
    """
    theta = load_occupancy_threshold() if threshold is None else threshold
    return bool(mean_occupancy < theta)


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
