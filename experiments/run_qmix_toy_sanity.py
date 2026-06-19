"""
Known-good toy sanity check for the self-contained QMIX / IQL learner.

WHY: a hand-rolled QMIX is the one place a silent bug can hide (a mis-wired
mixer, a target that doesn't bootstrap, an argmax over the wrong axis). The env
wiring check proves the SHAPES are right; this proves the LEARNING is right on a
problem with a known optimum, BEFORE we spend cluster time.

THE TOY: a one-step, 2-agent cooperative matrix game. Each agent picks an action
in {0, 1}; the team reward is a function of the joint action with a UNIQUE
optimum. We use a coordination payoff that monotonic mixing can represent
(diagonal-dominant -- higher individual contributions never hurt the team), so a
correct QMIX MUST converge to argmax. IQL must too (no inter-agent dependence in
the optimal-action structure here). A learner that can't solve this is broken.

    Payoff matrix R[a0, a1]   (the team reward; both agents share it)
        a1=0   a1=1
   a0=0   0.0    1.0
   a0=1   1.0    8.0          <- unique optimum at (1,1)

This is monotone (increasing either agent's action weakly increases the best
attainable team reward), so QMIX's abs()-weighted mixer can represent the
ordering. We do NOT use the classic non-monotone matrix (where QMIX provably
fails) -- that would test the algorithm's known limitation, not our wiring.

PASS: after training, greedy joint action == (1,1) and greedy Q_tot picks the
true argmax for both QMIX and IQL. Runs in a few seconds on CPU.
"""
import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.qmix_learner import QLearner, set_seed

PAYOFF = np.array([[0.0, 1.0],
                   [1.0, 8.0]])  # R[a0, a1]; unique optimum (1,1)
N_AGENTS = 2
N_ACTIONS = 2
LOCAL_DIM = N_AGENTS          # own_part(i) = one-hot agent id (no other obs)
STATE_DIM = N_AGENTS * N_AGENTS  # global state = concat of own_parts


def _own_parts():
    """Stateless toy: own_part(i) = one-hot agent id; state = concat of them."""
    parts = np.eye(N_AGENTS, dtype=np.float32)            # [N, LOCAL_DIM]
    state = parts.reshape(-1)                             # [STATE_DIM]
    return parts, state


def make_episode(actions):
    """One-step episode for the given joint action -> buffer episode dict."""
    parts, state = _own_parts()
    reward = float(PAYOFF[actions[0], actions[1]])
    return {
        "own_parts": parts[None, :, :],                   # [T=1, N, LOCAL_DIM]
        "actions": np.array(actions, dtype=np.int64)[None, :],   # [1, N]
        "reward": np.array([reward], dtype=np.float32),          # [1]
        "state": state[None, :],                                 # [1, STATE_DIM]
        "next_own_parts": parts[None, :, :],
        "next_state": state[None, :],
        "done": np.array([1.0], dtype=np.float32),               # terminal one-step
    }


def train_toy(use_mixer, seed=0, episodes=4000):
    set_seed(seed)
    learner = QLearner(n_agents=N_AGENTS, local_dim=LOCAL_DIM,
                       state_dim=STATE_DIM, n_actions=N_ACTIONS,
                       use_mixer=use_mixer, seed=seed,
                       eps_anneal_steps=2000, eps_finish=0.05,
                       learning_starts=16, batch_size=32,
                       target_update_interval=50, buffer_size=2000)
    rng = np.random.RandomState(seed)
    for _ in range(episodes):
        # behaviour policy: epsilon-greedy via the learner so the buffer covers
        # all four joint actions (the matrix is tiny; uniform exploration is fine)
        parts, _ = _own_parts()
        own = {f"a{i}": parts[i] for i in range(N_AGENTS)}
        act = learner.select_actions(own, explore=True)
        a = [act[f"a{i}"] for i in range(N_AGENTS)]
        learner.env_steps += 1
        learner.buffer.add(make_episode(a))
        learner.train()
    return learner


def greedy_joint(learner):
    parts, _ = _own_parts()
    with torch.no_grad():
        acts = []
        for i in range(N_AGENTS):
            t = torch.as_tensor(parts[i], dtype=torch.float32).unsqueeze(0)
            acts.append(int(learner.q(t).argmax(dim=1).item()))
    return tuple(acts)


def check(use_mixer, seed=0):
    name = "QMIX" if use_mixer else "IQL"
    learner = train_toy(use_mixer, seed=seed)
    joint = greedy_joint(learner)
    ok = (joint == (1, 1))
    print(f"  [{name}] greedy joint action = {joint}  "
          f"(optimum (1,1))  value_loss={learner.last_value_loss:.4f}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=== QMIX/IQL toy sanity check (2-agent coordination matrix) ===")
    results = []
    for use_mixer in (True, False):
        # a couple of seeds; require the majority to land on the optimum
        seeds_ok = [check(use_mixer, seed=s) for s in (0, 1, 2)]
        results.append(sum(seeds_ok) >= 2)
    overall = all(results)
    print(f"\nTOY SANITY: {'PASS' if overall else 'FAIL'} "
          f"(QMIX={'ok' if results[0] else 'BAD'}, IQL={'ok' if results[1] else 'BAD'})")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
