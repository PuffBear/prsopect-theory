"""
DeepMind Melting Pot `clean_up` substrate -> PettingZoo ParallelEnv adapter, wired
for the paper's w-dial + CPT reward machinery.

This is the *real* standard benchmark (google-deepmind/meltingpot), as opposed to the
lightweight in-repo `env/cleanup_env.py`. It exposes exactly the same reward-blend +
CPT contract as every other substrate in this pipeline, so the analysis and the frozen
collapse label carry over unchanged:

    raw env      : per-agent RGB obs + per-agent reward r_i^local (apples agent i eats)
    -> MeltingPotCleanupParallel : dm_env timestep -> PettingZoo parallel dicts
    -> CleanupMPLocalRewardWrapper : r_blend(i) = (1-w) r_i^local + w r^global
    -> CPTRewardWrapper           : r^PT = v(r_blend), v as in the paper (alpha,beta,lambda)

Reward definitions (mirrors env/cleanup_env.py so the two Cleanups are comparable):
    r_i^local = reward Melting Pot gives agent i this step (native per-agent apple reward).
    r^global  = MEAN per-agent reward this step = (sum_j r_j) / N  (team harvest rate).
                Mean (not sum) matches CleanupLocalRewardWrapper so w has the same
                meaning across both Cleanup substrates.
    "true reward" (for the collapse metric) = raw per-agent apple harvest, summed over
                the team over the episode = total apples collected. Cleaning is NOT
                rewarded -- that is the local-cost/global-benefit crux; w carries it.

=== TOPOLOGY / N ===
    Substrate `clean_up`. Default player count is the substrate's own (7 for clean_up),
    read from the Melting Pot config -- we do NOT hardcode it.

=== OBSERVATION ===
    Per-agent egocentric RGB (the substrate's `RGB` observation), transposed to
    channel-first uint8 (C,H,W) for an SB3 NatureCNN. No hand-built features.

=== INSTALL ===
    pip install dm-meltingpot   (pulls dmlab2d). GPU strongly preferred (pixels + CNN).
    Import is REQUIRED (no fallback); if meltingpot is missing this module raises at
    import time so a run can never silently proceed on the wrong environment.

References:
    Leibo et al. (2017) Multi-agent RL in Sequential Social Dilemmas. AAMAS.
    Agapiou et al. (2022) Melting Pot 2.0. arXiv:2211.13746.
    Yu et al. (2022) The Surprising Effectiveness of PPO in Cooperative MARL. NeurIPS.
"""
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import BaseParallelWrapper

# --- Hard dependency on the real Melting Pot suite (no fallback, by design) ---
from meltingpot import substrate as mp_substrate

DEFAULT_SUBSTRATE = "clean_up"
PLAYER_RGB_KEY = "RGB"          # per-player egocentric RGB observation key


def _build_mp_env(substrate_name=DEFAULT_SUBSTRATE, seed=None):
    """Build a raw Melting Pot substrate env and return (env, num_players).

    Uses the roles the substrate itself declares as its default population, so the
    player count is the benchmark's own (7 for clean_up), not a value we invented.
    """
    config = mp_substrate.get_config(substrate_name)
    roles = config.default_player_roles           # tuple, len == substrate's N
    env = mp_substrate.build_from_config(config, roles=roles)
    if seed is not None:
        # Melting Pot substrates seed via reset(); store for reset() to apply.
        pass
    return env, len(roles)


class MeltingPotCleanupParallel(ParallelEnv):
    """PettingZoo ParallelEnv adapter over the Melting Pot `clean_up` substrate.

    Returns the RAW per-agent reward (apple harvest) as the step reward. The w-blend
    is applied by CleanupMPLocalRewardWrapper on top; CPT on top of that -- identical
    layering to the other substrates.
    """

    metadata = {"render_modes": [], "name": "meltingpot_cleanup_v1"}

    def __init__(self, substrate_name=DEFAULT_SUBSTRATE, max_cycles=None):
        super().__init__()
        self._substrate_name = substrate_name
        self._env, self.n_agents = _build_mp_env(substrate_name)
        self.possible_agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agents = self.possible_agents[:]

        # max_cycles: default to the substrate's own episode length if not overridden.
        self.max_cycles = max_cycles
        self._step_count = 0

        # Derive obs/action spaces from the substrate's dm_env specs.
        obs_spec = self._env.observation_spec()      # list, one dict per player
        act_spec = self._env.action_spec()           # list, one per player

        # Per-agent RGB frame -> channel-first uint8 Box for NatureCNN.
        rgb_spec = obs_spec[0][PLAYER_RGB_KEY]
        h, w, c = rgb_spec.shape
        self._rgb_hwc = (h, w, c)
        rgb_box = spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

        # Discrete action count from the substrate's action spec.
        n_actions = int(act_spec[0].num_values)
        self._n_actions = n_actions

        self.observation_spaces = {a: rgb_box for a in self.possible_agents}
        self.action_spaces = {a: spaces.Discrete(n_actions) for a in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # -- helpers -------------------------------------------------------------
    def _obs_from_timestep(self, timestep):
        """dm_env timestep.observation (list of per-player dicts) -> agent->CHW uint8."""
        out = {}
        for i, agent in enumerate(self.possible_agents):
            rgb = np.asarray(timestep.observation[i][PLAYER_RGB_KEY], dtype=np.uint8)
            out[agent] = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
        return out

    def _rewards_from_timestep(self, timestep):
        """Per-player scalar reward -> agent->float (raw apple harvest)."""
        rew = timestep.reward
        return {agent: float(np.asarray(rew[i]).item())
                for i, agent in enumerate(self.possible_agents)}

    # -- PettingZoo API ------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self._step_count = 0
        timestep = self._env.reset()
        obs = self._obs_from_timestep(timestep)
        info = {a: {} for a in self.possible_agents}
        return obs, info

    def step(self, actions):
        self._step_count += 1
        # Melting Pot expects a list of int actions ordered by player index.
        action_list = [int(actions.get(a, 0)) for a in self.possible_agents]
        timestep = self._env.step(action_list)

        obs = self._obs_from_timestep(timestep)
        rewards = self._rewards_from_timestep(timestep)

        # Episode end: substrate LAST timestep, or our max_cycles override.
        substrate_done = bool(timestep.last())
        cap_done = (self.max_cycles is not None and self._step_count >= self.max_cycles)
        done = substrate_done or cap_done

        term = {a: done for a in self.possible_agents}
        trunc = {a: (cap_done and not substrate_done) for a in self.possible_agents}
        info = {a: {} for a in self.possible_agents}

        if done:
            self.agents = []
        return obs, rewards, term, trunc, info

    def render(self):
        return None

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# w-blend wrapper (identical contract to env.cleanup_env.CleanupLocalRewardWrapper)
# ---------------------------------------------------------------------------

class CleanupMPLocalRewardWrapper(BaseParallelWrapper):
    """Applies the w-dial to Melting Pot Cleanup.

        r_local(i) = apples collected by agent i this step (raw substrate reward)
        r_global   = mean apples collected per agent this step
        r_blend(i) = (1-w) * r_local(i) + w * r_global

    true_reward (collapse metric) = raw r_local (summed over team over episode by the
    evaluator) = total apples harvested. Set global_reward_weight=0.0 on the CPT wrapper
    above this so the blend is not double-applied.
    """

    def __init__(self, env, w):
        super().__init__(env)
        self.w = float(w)

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        n = max(len(rewards), 1)
        r_global = sum(rewards.values()) / n     # mean per-agent (team harvest rate)

        new_rewards = {}
        for agent in self.possible_agents:
            if agent not in rewards:
                continue
            r_local = rewards[agent]
            r_blend = (1.0 - self.w) * r_local + self.w * r_global
            new_rewards[agent] = r_blend
            if agent not in info:
                info[agent] = {}
            info[agent]["true_reward"] = r_local
            info[agent]["r_local"] = r_local
            info[agent]["r_global"] = r_global
            info[agent]["r_blend"] = r_blend
        return obs, new_rewards, term, trunc, info


def make_meltingpot_cleanup_with_w(w, substrate_name=DEFAULT_SUBSTRATE, max_cycles=None):
    """CleanupEnv (Melting Pot) -> CleanupMPLocalRewardWrapper(w).

    Pass the result to CPTRewardWrapper(..., global_reward_weight=0.0) for the PT
    transform, exactly like the other substrates.
    """
    base = MeltingPotCleanupParallel(substrate_name=substrate_name, max_cycles=max_cycles)
    return CleanupMPLocalRewardWrapper(base, w=w)
