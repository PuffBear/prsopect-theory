"""
MAgent2 `battle_v4` wrapped for the CPT centralized-critic MAPPO stack
(4th substrate). Validated viable by run_magent_smoke_local.py (CNN-PPO @ w=1
coordinates: red team reward -16.3 -> -2.0).

Stack (outermost last):
    battle_v4.parallel_env
      -> RedTeamWrapper        (control RED; BLUE = scripted random opponent;
                                obs transposed to channels-first (C,H,W))
      -> CPTRewardWrapper       (alpha/beta/lambda; w = global_reward_weight blends
                                own reward vs RED-TEAM-sum -> the coordination dial)
      -> GlobalStateImageWrapper (channel-stacked joint image for the CNN critic)

w=0 -> selfish (own kills), w=1 -> red-team reward (coordinated focus-fire). The
loss-aversion sweep raises lambda; the breadth claim is that w_crit rises with
lambda here as in supply chain / MPE / LBF.
"""
import numpy as np
from pettingzoo.utils import BaseParallelWrapper
from gymnasium import spaces

from magent2.environments import battle_v4
from agents.cpt_wrapper import CPTRewardWrapper
from env.global_state_image_wrapper import GlobalStateImageWrapper

MAP_SIZE = 16      # -> 6 red agents (3x the coordination of the size-12 gate);
                   # matches LBF/MPE team scale. Critic sees (6+1)*5 = 35 channels.
MAX_CYCLES = 200


class RedTeamWrapper(BaseParallelWrapper):
    """Expose ONLY the red team; step blue with random actions internally. Obs is
    transposed to channels-first (C,H,W) for SB3's CNN. With red-only agents,
    CPTRewardWrapper's global_reward = sum(red rewards) = the red TEAM reward."""

    def __init__(self, env):
        super().__init__(env)
        self._all = env.possible_agents
        self.possible_agents = [a for a in self._all if a.startswith("red")]
        self.agents = list(self.possible_agents)
        self._obs_spaces = {}
        for a in self.possible_agents:
            s = env.observation_space(a)
            h, w, c = s.shape
            self._obs_spaces[a] = spaces.Box(
                low=np.transpose(s.low, (2, 0, 1)),
                high=np.transpose(s.high, (2, 0, 1)),
                shape=(c, h, w), dtype=s.dtype)

    def observation_space(self, a): return self._obs_spaces[a]
    def action_space(self, a): return self.env.action_space(a)

    @staticmethod
    def _cf(o):
        return np.transpose(np.asarray(o), (2, 0, 1))

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = [a for a in self.env.agents if a.startswith("red")]
        return ({a: self._cf(obs[a]) for a in self.agents},
                {a: info.get(a, {}) for a in self.agents})

    def step(self, red_actions):
        acts = dict(red_actions)
        for a in self.env.agents:
            if a.startswith("blue"):
                acts[a] = self.env.action_space(a).sample()
        obs, rew, term, trunc, info = self.env.step(acts)
        self.agents = [a for a in self.env.agents if a.startswith("red")]
        keep = lambda d: {a: d[a] for a in self.agents if a in d}
        kept_obs = {a: self._cf(obs[a]) for a in self.agents if a in obs}
        return kept_obs, keep(rew), keep(term), keep(trunc), keep(info)


def make_magent_env(w, lam, alpha, beta, map_size=MAP_SIZE, max_cycles=MAX_CYCLES,
                    with_global_state=True):
    """Full CPT MAgent stack. with_global_state adds the channel-stacked joint
    image for the centralized critic (omit it for the PPO control)."""
    base = battle_v4.parallel_env(map_size=map_size, max_cycles=max_cycles,
                                  extra_features=False)
    red = RedTeamWrapper(base)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta}
              for a in red.possible_agents}
    cpt = CPTRewardWrapper(red, params, reward_scale=1.0, global_reward_weight=w)
    return GlobalStateImageWrapper(cpt) if with_global_state else cpt


# --- Collapse label ----------------------------------------------------------
import os, json as _json
_THRESHOLD_JSON = os.path.join(os.path.dirname(__file__), "..",
                               "docs", "exp_magent", "magent_collapse_threshold.json")
_DEFAULT_THRESHOLD = -10.0  # placeholder; calibrate from data (see exp_magent_calibrate)


def load_magent_collapse_threshold(default=_DEFAULT_THRESHOLD):
    try:
        with open(_THRESHOLD_JSON) as f:
            return float(_json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return float(default)


def is_collapsed_magent(team_reward, threshold=None):
    """collapsed := mean red-team TRUE reward per eval episode below theta
    (coordination failure -> the team gets wiped / fails to focus-fire)."""
    theta = load_magent_collapse_threshold() if threshold is None else threshold
    return bool(team_reward < theta)
