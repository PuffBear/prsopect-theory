"""
Cleanup Sequential Social Dilemma (Leibo et al. 2017) as a PettingZoo ParallelEnv.

Based on the canonical Cleanup task from:
    Leibo, J.Z. et al. (2017). Multi-agent Reinforcement Learning in Sequential
    Social Dilemmas. AAMAS.  https://arxiv.org/abs/1702.03037

And studied under mixed incentives in:
    Hughes, E. et al. (2018). Inequity Aversion Improves Cooperation in
    Intertemporal Social Dilemmas. NeurIPS.

=== TASK DESCRIPTION ===

A shared 10x8 grid contains:
  - Waste region (cols 0-2):  waste tiles spawn stochastically each step.
  - Apple region (cols 7-9):  apples spawn at a rate proportional to cleanliness,
                              where cleanliness = 1 - (current_waste / max_waste).
  - Traversal space (cols 3-6): neutral tiles, agents move freely.

Agent actions: NOOP, UP, DOWN, LEFT, RIGHT, CLEAN.
  - CLEAN:  if agent is standing on a waste cell, removes the waste.
  - Moving onto an apple cell:  collects the apple automatically (+1 local reward).

Incentive structure:
  r_local(i) = apples collected by agent i this step
  r_global   = mean apples collected per agent this step (team harvest rate)
  r_blend(i) = (1-w) * r_local(i)  +  w * r_global

  At w=0 (pure local):  all agents harvest, none clean.  Waste accumulates,
     cleanliness → 0, apple spawn rate → 0.  COLLAPSE (tragedy of the commons).
  At w=1 (pure global): agents learn to specialise -- some clean, others harvest.
     Cleanliness maintained, apple supply sustained.  COORDINATION.

This is the same phenomenon as the supply-chain collapse:
  supply chain:  hoard stock  ->  downstream starves  ->  collapse
  Cleanup:       harvest only ->  waste grows         ->  apple supply collapses

=== OBSERVATION ===

Each agent observes a 5x5 local window centred on itself (3 channels):
    channel 0: waste tiles
    channel 1: apple tiles
    channel 2: agent tiles (number of agents on each cell)
Plus 2 scalars: (global_cleanliness, own_apple_count_normalised)
Total: 5*5*3 + 2 = 77-dimensional flat vector.

=== INSTALL ===

No additional packages needed; only NumPy + PettingZoo.
"""
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces

# --- Grid constants ----------------------------------------------------------
GRID_H = 8
GRID_W = 10

WASTE_COLS  = list(range(0, 3))       # columns 0-2 (waste region)
APPLE_COLS  = list(range(7, 10))      # columns 7-9 (apple region)
WASTE_CELLS = [(r, c) for r in range(GRID_H) for c in WASTE_COLS]
APPLE_CELLS = [(r, c) for r in range(GRID_H) for c in APPLE_COLS]
MAX_WASTE   = len(WASTE_CELLS)        # 24 cells

WASTE_SPAWN_PROB = 0.5    # per empty waste cell per step
APPLE_SPAWN_BASE = 0.3    # max apple spawn prob per empty apple cell per step

MAX_CYCLES_DEFAULT = 200  # episode length

# Action encoding
NOOP, UP, DOWN, LEFT, RIGHT, CLEAN = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6
DR = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), NOOP: (0, 0)}

# Observation window
OBS_HALF = 2   # 5x5 window
OBS_SIZE = (2*OBS_HALF+1) * (2*OBS_HALF+1) * 3 + 2  # 77


class CleanupEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv implementation of the Cleanup social dilemma.

    Compatible with SuperSuit vectorisation and the CPTRewardWrapper.
    The env returns r_local (per-agent apple harvest) as the step reward.
    The w-blending is done by CleanupLocalRewardWrapper (applied on top).
    """

    metadata = {"render_modes": [], "name": "cleanup_v1"}

    def __init__(self, n_agents=3, max_cycles=MAX_CYCLES_DEFAULT):
        super().__init__()
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(N_ACTIONS) for a in self.possible_agents
        }

        # Internal state
        self._waste = None    # (H, W) bool array
        self._apples = None   # (H, W) bool array
        self._pos = None      # {agent: (row, col)}
        self._step_count = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # -------------------------------------------------------------------------
    # PettingZoo API
    # -------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self._step_count = 0

        self._waste = np.zeros((GRID_H, GRID_W), dtype=bool)
        self._apples = np.zeros((GRID_H, GRID_W), dtype=bool)

        # Seed some initial waste (30% of waste cells)
        for (r, c) in WASTE_CELLS:
            if np.random.random() < 0.3:
                self._waste[r, c] = True

        # Seed some initial apples (20% of apple cells)
        for (r, c) in APPLE_CELLS:
            if np.random.random() < 0.2:
                self._apples[r, c] = True

        # Place agents randomly in traversal zone (cols 3-6)
        traverse_cells = [(r, c) for r in range(GRID_H) for c in range(3, 7)]
        np.random.shuffle(traverse_cells)
        self._pos = {}
        for i, a in enumerate(self.possible_agents):
            self._pos[a] = traverse_cells[i]

        obs = {a: self._observe(a) for a in self.possible_agents}
        info = {a: {} for a in self.possible_agents}
        return obs, info

    def step(self, actions):
        assert self._waste is not None, "Call reset() before step()."
        self._step_count += 1

        rewards = {a: 0.0 for a in self.possible_agents}

        # --- Process agent actions -------------------------------------------
        new_pos = {}
        for agent in self.possible_agents:
            act = int(actions.get(agent, NOOP))
            r, c = self._pos[agent]

            if act == CLEAN:
                if self._waste[r, c]:
                    self._waste[r, c] = False   # remove waste, no reward
                new_pos[agent] = (r, c)

            elif act in DR and act != NOOP:
                dr, dc = DR[act]
                nr, nc = r + dr, c + dc
                # Boundary check
                if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                    new_pos[agent] = (nr, nc)
                else:
                    new_pos[agent] = (r, c)
            else:
                new_pos[agent] = (r, c)

        self._pos = new_pos

        # --- Apple collection (agents on apple cells) ------------------------
        # If multiple agents on same apple cell, first one (by index) gets it.
        collected = set()
        for agent in self.possible_agents:
            r, c = self._pos[agent]
            if self._apples[r, c] and (r, c) not in collected:
                rewards[agent] += 1.0
                self._apples[r, c] = False
                collected.add((r, c))

        # --- Stochastic waste spawning ----------------------------------------
        cleanliness = 1.0 - (self._waste.sum() / MAX_WASTE)
        for (r, c) in WASTE_CELLS:
            if not self._waste[r, c] and np.random.random() < WASTE_SPAWN_PROB:
                self._waste[r, c] = True

        # --- Stochastic apple spawning (depends on cleanliness^2) -----------
        apple_prob = APPLE_SPAWN_BASE * (cleanliness ** 2)
        for (r, c) in APPLE_CELLS:
            if not self._apples[r, c] and np.random.random() < apple_prob:
                self._apples[r, c] = True

        # --- Build return dicts ----------------------------------------------
        done = self._step_count >= self.max_cycles
        obs  = {a: self._observe(a) for a in self.possible_agents}
        term = {a: done for a in self.possible_agents}
        trunc = {a: False for a in self.possible_agents}
        info = {a: {"cleanliness": cleanliness,
                    "waste_count": int(self._waste.sum()),
                    "apple_count": int(self._apples.sum())}
                for a in self.possible_agents}

        if done:
            self.agents = []

        return obs, rewards, term, trunc, info

    # -------------------------------------------------------------------------
    # Observation helper
    # -------------------------------------------------------------------------

    def _observe(self, agent):
        r, c = self._pos[agent]

        # 5x5 window (padded with zeros outside grid)
        window_waste  = np.zeros((2*OBS_HALF+1, 2*OBS_HALF+1), dtype=np.float32)
        window_apple  = np.zeros_like(window_waste)
        window_agents = np.zeros_like(window_waste)

        for di in range(-OBS_HALF, OBS_HALF+1):
            for dj in range(-OBS_HALF, OBS_HALF+1):
                nr, nc = r+di, c+dj
                wi, wj = di+OBS_HALF, dj+OBS_HALF
                if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                    window_waste[wi, wj]  = float(self._waste[nr, nc])
                    window_apple[wi, wj]  = float(self._apples[nr, nc])

        # Agent positions in window
        for other in self.possible_agents:
            or_, oc = self._pos[other]
            wi, wj = or_ - r + OBS_HALF, oc - c + OBS_HALF
            if 0 <= wi < 2*OBS_HALF+1 and 0 <= wj < 2*OBS_HALF+1:
                window_agents[wi, wj] += 1.0 / self.n_agents  # normalise

        flat = np.concatenate([
            window_waste.flatten(),
            window_apple.flatten(),
            window_agents.flatten(),
            np.array([
                1.0 - float(self._waste.sum()) / MAX_WASTE,  # global cleanliness
                float(self._step_count) / self.max_cycles,   # time fraction
            ], dtype=np.float32),
        ])
        return flat.astype(np.float32)


# ---------------------------------------------------------------------------
# Local reward wrapper (w-blending for Cleanup)
# ---------------------------------------------------------------------------

import os
import json as _json
from pettingzoo.utils import BaseParallelWrapper


class CleanupLocalRewardWrapper(BaseParallelWrapper):
    """
    Applies the w-dial to the Cleanup environment.

    r_local(i) = apples collected by agent i this step (from base env)
    r_global   = mean apples collected per agent this step
    r_blend(i) = (1-w) * r_local(i)  +  w * r_global

    True reward (for evaluation / collapse metric) = sum of r_local per episode
    = total apples collected by agent i across the episode.
    (We track TEAM total: sum over all agents, averaged.)
    """

    def __init__(self, env, w):
        super().__init__(env)
        self.w = float(w)

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)

        r_global = sum(rewards.values()) / max(len(rewards), 1)

        new_rewards = {}
        for agent in self.possible_agents:
            if agent not in rewards:
                continue
            r_local = rewards[agent]  # apples collected by this agent
            r_blend = (1.0 - self.w) * r_local + self.w * r_global
            new_rewards[agent] = r_blend

            if agent not in info:
                info[agent] = {}
            info[agent]["true_reward"] = r_local   # raw apple harvest
            info[agent]["r_local"]  = r_local
            info[agent]["r_global"] = r_global
            info[agent]["r_blend"]  = r_blend

        return obs, new_rewards, term, trunc, info


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def make_cleanup_env(n_agents=3, max_cycles=MAX_CYCLES_DEFAULT):
    """Create a raw Cleanup PettingZoo parallel env."""
    return CleanupEnv(n_agents=n_agents, max_cycles=max_cycles)


def make_cleanup_env_with_w(w, n_agents=3, max_cycles=MAX_CYCLES_DEFAULT):
    """
    Convenience factory: CleanupEnv -> CleanupLocalRewardWrapper(w).
    Pass the result to CPTRewardWrapper for the PT transform.
    """
    base = make_cleanup_env(n_agents=n_agents, max_cycles=max_cycles)
    return CleanupLocalRewardWrapper(base, w=w)


# ---------------------------------------------------------------------------
# Collapse definition
# ---------------------------------------------------------------------------

_THRESHOLD_JSON = os.path.join(os.path.dirname(__file__), "..",
                               "docs", "exp_cleanup", "cleanup_collapse_threshold.json")
_DEFAULT_THRESHOLD = 5.0   # < 5 total apples per episode = collapsed


def load_cleanup_collapse_threshold(default=_DEFAULT_THRESHOLD):
    try:
        with open(_THRESHOLD_JSON) as f:
            return float(_json.load(f)["threshold"])
    except (FileNotFoundError, KeyError, ValueError, OSError):
        return float(default)


CLEANUP_COLLAPSE_THRESHOLD = load_cleanup_collapse_threshold()


def is_collapsed_cleanup(mean_reward, threshold=None):
    """
    Collapse definition for Cleanup.

    Args:
        mean_reward: mean cumulative TRUE reward (apples harvested) per eval episode.
    Returns:
        True if the team harvested fewer than threshold apples per episode.
    """
    theta = load_cleanup_collapse_threshold() if threshold is None else threshold
    return bool(mean_reward < theta)
