"""
Level-Based Foraging (LBF) environment wrapper with the incentive w-dial.

Replaces RWARE as the second transfer substrate (2026-06-13): vanilla PPO
never observes a delivery in RWARE at feasible horizons (local smoke test:
0.000 deliveries after 600k steps, value_loss ~ 3e-06), so no collapse
boundary can exist there. LBF is from the same research group (Christianos
et al. 2020 / Papoudakis et al. benchmark), is PPO-learnable at 1-2M steps,
and has a NATIVE local-vs-team incentive structure -- no artificial local
bonus is needed.

=== INCENTIVE MISALIGNMENT STRUCTURE (public-goods form) ===

In LBF, agents collect foods on a grid. A food of level L can only be
loaded if the summed level of simultaneously-loading adjacent agents is
>= L. The env emits PER-AGENT rewards (each loader's share of the food).

v1 of this wrapper used the native shares alone, but own-share and team
return are mostly ALIGNED in LBF (you only earn by participating), so the
w=0 end was not collectively destructive and no collapse axis existed
(local smoke test 2026-06-13: trained w=1 = 0.227 vs random 0.116).

v2 adds an EFFORT COST on the load action to the local channel, giving
the canonical public-goods structure (cooperation is costly to the
individual, valuable to the team):

    r_local(i)  = own_share(i) - EFFORT_COST * 1[action_i == LOAD]
    r_global    = sum_j own_share(j)            (team return, no costs)
    r_blend(i)  = (1-w) * r_local(i) + w * r_global

At w=0: loading is dominated by free-riding for low own-share payoffs;
agents avoid the load action; team return COLLAPSES.
At w=1: the team return dominates and the effort term vanishes from the
blend; agents learn coordinated loading. COORDINATION.

v2 also moves to a 2-player 5x5 grid so the w=1 (team-summed reward)
end is learnable by shared-policy PPO at 2M steps -- the 3p-8x8 variant
trains too slowly under diluted team credit.

=== COLLAPSE DEFINITION ===

Primary metric: mean TEAM RETURN per eval episode (sum over agents and
steps of the native reward; in LBF this is normalised so that collecting
all foods gives a fixed total). collapsed := team_return < theta, theta
calibrated from trained w=0 vs w=1 endpoints (see exp_lbf_calibrate.py),
gated on clean separation.

=== WRAPPER STACK ===

    gymnasium "Foraging-*-v3" (tuple obs / list rewards)
      -> LBFParallelWrapper      (gymnasium -> PettingZoo parallel)
      -> LBFLocalRewardWrapper   (w-blend of native local vs team reward)
      -> CPTRewardWrapper        (alpha/beta/lambda)
      -> SuperSuit vectorisation

=== INSTALL ===

    pip install lbforaging
"""
import os
import json as _json
import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
from pettingzoo.utils import BaseParallelWrapper

try:
    import lbforaging  # noqa: F401 -- registers the Foraging-* env ids
except ImportError:
    raise ImportError(
        "lbforaging not found. Install with:  pip install lbforaging\n"
        "Then verify:  python -c 'import lbforaging'"
    )

# 5x5 grid, 2 players, 2 foods: small enough that team-summed reward (w=1)
# is learnable by shared-policy PPO at 2M steps; food levels still include
# cooperation-requiring foods.
DEFAULT_ENV_ID = "Foraging-5x5-2p-2f-v3"
MAX_CYCLES = 50   # LBF default episode length

# Effort cost charged to the LOCAL channel when an agent uses the LOAD
# action (id 5). Native food shares are O(0.1-0.5); 0.02 per attempt makes
# loading individually unprofitable unless own-share returns are reliable,
# without bankrupting the coordinated regime at intermediate w.
# Calibrate via the smoke test / calibration gate before the sweep.
EFFORT_COST = 0.02
LOAD_ACTION = 5


class LBFParallelWrapper(ParallelEnv):
    """Thin PettingZoo ParallelEnv adapter for the gymnasium LBF env."""

    metadata = {"render_modes": [], "name": "lbf_parallel_v1"}
    render_mode = None   # required by SuperSuit's MarkovVectorEnv

    def __init__(self, env_id=DEFAULT_ENV_ID, max_cycles=MAX_CYCLES):
        super().__init__()
        self._gym_env = gym.make(env_id, max_episode_steps=max_cycles)
        inner = self._gym_env.unwrapped
        self._n_agents = len(inner.players)
        self.max_cycles = max_cycles
        self._step_count = 0

        self.possible_agents = [f"agent_{i}" for i in range(self._n_agents)]
        self.agents = self.possible_agents[:]

        gym_obs_space = self._gym_env.observation_space
        gym_act_space = self._gym_env.action_space
        self.observation_spaces = {
            a: gym_obs_space[i] for i, a in enumerate(self.possible_agents)
        }
        self.action_spaces = {
            a: gym_act_space[i] for i, a in enumerate(self.possible_agents)
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        obs_tuple, _info = self._gym_env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        self._step_count = 0
        obs_dict = {a: np.asarray(obs_tuple[i], dtype=np.float32)
                    for i, a in enumerate(self.possible_agents)}
        info_dict = {a: {} for a in self.possible_agents}
        return obs_dict, info_dict

    def step(self, actions):
        action_list = [int(actions.get(a, 0)) for a in self.possible_agents]
        obs_tuple, rewards_list, term, trunc, _info = self._gym_env.step(
            tuple(action_list))
        self._step_count += 1

        done = bool(
            (term if np.isscalar(term) else np.all(term))
            or (trunc if np.isscalar(trunc) else np.all(trunc))
            or self._step_count >= self.max_cycles
        )

        obs_dict = {a: np.asarray(obs_tuple[i], dtype=np.float32)
                    for i, a in enumerate(self.possible_agents)}
        rew_dict = {a: float(rewards_list[i])
                    for i, a in enumerate(self.possible_agents)}
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


class LBFLocalRewardWrapper(BaseParallelWrapper):
    """
    Public-goods w-dial over LBF rewards (v2).

        r_local(i) = own_share(i) - EFFORT_COST * 1[action_i == LOAD]
        r_global   = sum_j own_share(j)   (team return, no effort costs)
        r_blend(i) = (1-w) * r_local(i) + w * r_global

    info[agent]["true_reward"] = own_share(i) (native share, NO effort
    cost); the eval metric sums true_reward over agents and steps = team
    return per episode, so the collapse metric is environment-grounded
    and independent of EFFORT_COST, w, alpha, beta, lambda.
    """

    def __init__(self, env, w, effort_cost=EFFORT_COST):
        super().__init__(env)
        self.w = float(w)
        self.effort_cost = float(effort_cost)

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        r_global = float(sum(rewards.values()))

        new_rewards = {}
        for agent in self.possible_agents:
            if agent not in rewards:
                continue
            loaded = int(actions.get(agent, 0)) == LOAD_ACTION
            r_local = float(rewards[agent]) - (self.effort_cost if loaded else 0.0)
            r_blend = (1.0 - self.w) * r_local + self.w * r_global
            new_rewards[agent] = r_blend
            if agent not in info:
                info[agent] = {}
            info[agent]["true_reward"] = float(rewards[agent])
            info[agent]["r_local"] = r_local
            info[agent]["r_global"] = r_global
            info[agent]["r_blend"] = r_blend
        return obs, new_rewards, term, trunc, info


def make_lbf_env_with_w(w, env_id=DEFAULT_ENV_ID, max_cycles=MAX_CYCLES,
                        effort_cost=EFFORT_COST):
    """LBFParallelWrapper -> LBFLocalRewardWrapper(w, effort_cost)."""
    return LBFLocalRewardWrapper(
        LBFParallelWrapper(env_id=env_id, max_cycles=max_cycles),
        w=w, effort_cost=effort_cost)


# --- Collapse definition ----------------------------------------------------
_THRESHOLD_JSON = os.path.join(os.path.dirname(__file__), "..",
                               "docs", "exp_lbf", "lbf_collapse_threshold.json")
_DEFAULT_THRESHOLD = 0.5   # placeholder; calibrate before trusting labels


def load_lbf_collapse_threshold(default=_DEFAULT_THRESHOLD):
    try:
        with open(_THRESHOLD_JSON) as f:
            return float(_json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return float(default)


def is_collapsed_lbf(team_return, threshold=None):
    """collapsed := mean team return per eval episode below theta."""
    theta = load_lbf_collapse_threshold() if threshold is None else threshold
    return bool(team_return < theta)
