"""
gym-anm (ANM6Easy) wrapped as a multi-agent PettingZoo ParallelEnv for the CPT
w-dial cascade test --- the SECOND directed-flow substrate (after the supply
chain) for the minority/downstream-dominance cascade.

WHY THIS ENV: the cascade is intrinsically about a directed resource-flow
topology with a well-defined upstream/downstream ordering. ANM6Easy is a radial
power feeder: a slack generator at bus 0 supplies, via branches, the
distributed-energy-resource (DER) agents at load buses 3, 4, 5. Power flows
0 -> {3,4,5}, and within the radial feeder bus 3 is electrically UPSTREAM of
buses 4 and 5. This is a genuinely different topology from the serial supply
chain (generation->distributed-load tree vs linear chain), so a replication of
the downstream-dominance asymmetry here turns the cascade from N=1 into a law.

AGENTS: the 3 controllable DERs --- RenewableGen@bus3, RenewableGen@bus4,
StorageUnit@bus5 (the slack gen @bus0 is not agent-controlled; it absorbs
imbalance). Each agent acts on its own device's control dims.

PER-NODE REWARD (principled decomposition of gym-anm's OWN published cost
r = -(energy_loss + lamb * constraint_penalty), Simulator._compute_reward):
  local_reward(i) = -( e_loss_i + lamb * penalty_i ) * delta_t
    e_loss_i  = agent i's device energy-loss / curtailment contribution
    penalty_i = voltage violation at agent i's bus
  global_reward = the full system reward gym-anm returns (sum over all nodes).
The CPT w-dial then blends: r_blend(i) = (1-w) local_i + w global. This is the
identical local-vs-global structure used on the supply chain; w=0 is selfish
(each DER minimizes only its own losses/violations), w=1 is system-optimal.

Pair with agents.cpt_wrapper.CPTRewardWrapper (set per-agent lambda to place a
loss-averse node upstream vs downstream).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

import gym_anm  # noqa: F401  (registers the gym_anm envs)

# Controllable DER device_id -> bus, in UPSTREAM->DOWNSTREAM feeder order.
# (device ids from sim.devices; slack gen @bus0 excluded -- not agent-controlled.)
AGENT_DEVICES = [("der_bus3", 2, 3), ("der_bus4", 4, 4), ("stor_bus5", 6, 5)]
UPSTREAM_AGENT = "der_bus3"      # electrically closest to the slack source
DOWNSTREAM_AGENT = "stor_bus5"   # feeder tail


class ANMParallelWrapper(ParallelEnv):
    """ANM6Easy as a PettingZoo parallel env with per-node local rewards and the
    full action vector split across the 3 controllable DER agents."""

    metadata = {"render_modes": [], "name": "anm6_cascade_v1"}
    render_mode = None

    def __init__(self, max_cycles=200):
        super().__init__()
        self._env = gym.make("gym_anm:ANM6Easy-v0")
        self._u = self._env.unwrapped
        self._sim = self._u.simulator
        self.max_cycles = max_cycles
        self._t = 0

        self.possible_agents = [a for a, _, _ in AGENT_DEVICES]
        self.agents = list(self.possible_agents)
        self._dev = {a: d for a, d, _ in AGENT_DEVICES}
        self._bus = {a: b for a, _, b in AGENT_DEVICES}

        # Split the global Box(6) action into per-agent slices. ANM action layout
        # is [P_curt(gen_rer x2), P_stor, Q_stor, ...]; we partition by index so
        # each agent owns a contiguous, fixed slice (the actor only sees its own).
        full = self._u.action_space
        self._adim = full.shape[0]
        # 3 agents over 6 dims -> 2 dims each (gen agents: [P_curt, pad]; storage: [P,Q])
        self._aslice = {"der_bus3": (0, 1), "der_bus4": (1, 2), "stor_bus5": (3, 5)}
        self._full_low, self._full_high = full.low, full.high

        obs_dim = int(np.asarray(self._u.observation_space.shape).prod())
        self._obs_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.observation_spaces = {a: self._obs_space for a in self.possible_agents}
        self.action_spaces = {
            a: spaces.Box(self._full_low[s0:s1], self._full_high[s0:s1],
                          (s1 - s0,), np.float64)
            for a, (s0, s1) in self._aslice.items()
        }

    def observation_space(self, a): return self.observation_spaces[a]
    def action_space(self, a): return self.action_spaces[a]

    def _obs_all(self, obs):
        o = np.asarray(obs, dtype=np.float32).ravel()
        return {a: o for a in self.agents}   # shared global obs (CTDE-friendly)

    def _local_rewards(self):
        """Principled per-node decomposition of gym-anm's own cost terms."""
        sim = self._sim
        lamb, dt = sim.lamb, sim.delta_t
        loc = {}
        for a in self.possible_agents:
            dev = sim.devices[self._dev[a]]
            # device energy-loss / curtailment contribution
            e_i = abs(float(getattr(dev, "p", 0.0)))
            if type(dev).__name__ == "RenewableGen":
                e_i = max(0.0, float(getattr(dev, "p_pot", 0.0) - getattr(dev, "p", 0.0)))
            # voltage-violation penalty at this agent's bus
            bus = sim.buses[self._bus[a]]
            v = abs(bus.v)
            pen_i = max(0.0, v - bus.v_max) + max(0.0, bus.v_min - v)
            loc[a] = -(e_i + lamb * pen_i) * dt
        return loc

    def reset(self, seed=None, options=None):
        obs, _info = self._env.reset(seed=seed, options=options)
        self._sim = self._u.simulator
        self._t = 0
        self.agents = list(self.possible_agents)
        return self._obs_all(obs), {a: {} for a in self.agents}

    def step(self, actions):
        # reassemble the global action from per-agent slices, then CLIP to the
        # env's action bounds. PPO (and supersuit action padding) can emit values
        # outside gym-anm's per-dim Box, which the simulator rejects; clipping is
        # the standard fix and is behavior-preserving (the env itself saturates).
        full = np.zeros(self._adim, dtype=np.float64)
        for a, (s0, s1) in self._aslice.items():
            full[s0:s1] = np.asarray(actions[a], dtype=np.float64).ravel()[: s1 - s0]
        full = np.clip(full, self._full_low, self._full_high)
        obs, g_reward, term, trunc, info = self._env.step(full)
        self._t += 1
        local = self._local_rewards()
        done = bool(term or trunc or self._t >= self.max_cycles)
        # info carries both pieces so the CPT wrapper's global=sum matches the
        # true system reward, and evaluation can read the env's true reward.
        infos = {a: {"true_reward": float(g_reward),
                     "local_reward": local[a],
                     "global_reward": float(g_reward)} for a in self.agents}
        rewards = local                       # CPTRewardWrapper blends with global=sum
        terms = {a: done for a in self.agents}
        truncs = {a: done for a in self.agents}
        if done:
            self.agents = []
        return self._obs_all(obs), rewards, terms, truncs, infos


def make_anm_env(w, lam_map=None, alpha=1.0, beta=1.0, max_cycles=200):
    """Full CPT ANM stack. lam_map: optional {agent: lambda} to place loss
    aversion at a specific node (upstream vs downstream) for the cascade test;
    scalar `lam` applied to all if a float is passed."""
    from agents.cpt_wrapper import CPTRewardWrapper
    base = ANMParallelWrapper(max_cycles=max_cycles)
    if lam_map is None:
        lam_map = {a: 1.0 for a in base.possible_agents}
    elif isinstance(lam_map, (int, float)):
        lam_map = {a: float(lam_map) for a in base.possible_agents}
    params = {a: {"lambda": lam_map[a], "alpha": alpha, "beta": beta}
              for a in base.possible_agents}
    return CPTRewardWrapper(base, params, reward_scale=1.0, global_reward_weight=w)
