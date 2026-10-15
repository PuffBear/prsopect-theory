"""
Shared rollout / training loop for the QMIX-IQL value-based cells.

One code path for BOTH environments (supply chain, MPE) and BOTH learners
(QMIX, IQL): the driver supplies an env factory that returns a PettingZoo
parallel env whose OUTERMOST wrapper is GlobalStateWrapper (so every agent obs is
[ own_part(i) | global_state ]). This module never imports an env -- it only
knows the GlobalStateWrapper contract -- so supply and MPE share it exactly,
the way exp_ppo_mpe / exp_mappo_mpe share their machinery.

ISO-SAMPLE BUDGET: training runs until `total_env_steps` ENVIRONMENT STEPS have
elapsed (counted as env.step() calls; one call advances all agents one tick),
matching how PPO/MAPPO total_timesteps is counted. Gradient steps are incidental.

TEAM REWARD for the joint TD target: the per-agent CPT rewards are summed into a
single team reward. The mixer (QMIX) / sum (IQL) regresses Q_tot onto this team
return. The collapse label is an env-outcome metric computed at eval and is
independent of this choice (plan section 1).
"""
import numpy as np
import torch

from agents.qmix_learner import QLearner, set_seed


def _split(aug_obs, local_dim):
    """GlobalStateWrapper obs = [ own_part(local_dim) | global_state ]."""
    a = np.asarray(aug_obs, dtype=np.float32).ravel()
    return a[:local_dim], a[local_dim:]


def make_learner(env, use_mixer, seed, device="cpu", **kw):
    """Build a QLearner sized to a GlobalStateWrapper-wrapped env."""
    agents = list(env.possible_agents)
    local_dim = env.local_dim
    aug_dim = int(np.prod(env.observation_space(agents[0]).shape))
    state_dim = aug_dim - local_dim
    n_actions = env.action_space(agents[0]).n  # Discrete
    return QLearner(n_agents=len(agents), local_dim=local_dim, state_dim=state_dim,
                    n_actions=n_actions, use_mixer=use_mixer, device=device,
                    seed=seed, **kw)


def train(env, learner, total_env_steps, seed, verbose=False):
    """Train `learner` on `env` for `total_env_steps` env steps. Returns learner."""
    set_seed(seed)
    agents = list(env.possible_agents)
    local_dim = learner.local_dim
    ep_count = 0

    while learner.env_steps < total_env_steps:
        obs, _ = env.reset(seed=seed + ep_count)
        # per-episode transition stores (stacked over agents in fixed order)
        ep = {k: [] for k in ("own_parts", "actions", "reward",
                              "state", "next_own_parts", "next_state", "done")}
        while env.agents:
            parts = {}
            states = {}
            for a in env.agents:
                op, st = _split(obs[a], local_dim)
                parts[a] = op
                states[a] = st
            act = learner.select_actions(parts, explore=True)
            next_obs, rewards, term, trunc, info = env.step(act)
            learner.env_steps += 1

            # stack in fixed agent order; absent agents -> zeros (episodes are
            # synchronous here, but keep it robust)
            order = agents
            own_parts = np.stack([parts.get(a, np.zeros(local_dim, np.float32))
                                  for a in order])
            actions = np.array([act.get(a, 0) for a in order], dtype=np.int64)
            team_reward = float(sum(rewards.values()))
            # global state is identical across agents (concat of all own_parts)
            state = states[order[0]] if order[0] in states else \
                np.concatenate([parts.get(a, np.zeros(local_dim, np.float32)) for a in order])
            next_parts = {}
            for a in order:
                if a in next_obs:
                    nop, _ = _split(next_obs[a], local_dim)
                    next_parts[a] = nop
            next_own_parts = np.stack([next_parts.get(a, np.zeros(local_dim, np.float32))
                                       for a in order])
            done = float(any(term.values()) or any(trunc.values()))
            # next global state from next own parts
            next_state = np.concatenate(
                [next_parts.get(a, np.zeros(local_dim, np.float32)) for a in order])

            ep["own_parts"].append(own_parts)
            ep["actions"].append(actions)
            ep["reward"].append(team_reward)
            ep["state"].append(state)
            ep["next_own_parts"].append(next_own_parts)
            ep["next_state"].append(next_state)
            ep["done"].append(done)

            obs = next_obs

        episode = {k: np.array(v) for k, v in ep.items()}
        learner.buffer.add(episode)
        ep_count += 1
        if ep_count % learner.cfg["train_every"] == 0:
            learner.train()
        if verbose and ep_count % 50 == 0:
            print(f"    ep={ep_count} env_steps={learner.env_steps} "
                  f"eps={learner.epsilon():.3f} value_loss={learner.last_value_loss:.4f}")
    return learner


def greedy_action_fn(learner, local_dim):
    """Return a function (agent, aug_obs) -> int greedy action for evaluation."""
    def _f(aug_obs):
        op, _ = _split(aug_obs, local_dim)
        with torch.no_grad():
            t = torch.as_tensor(op, dtype=torch.float32,
                                device=learner.device).unsqueeze(0)
            return int(learner.q(t).argmax(dim=1).item())
    return _f
