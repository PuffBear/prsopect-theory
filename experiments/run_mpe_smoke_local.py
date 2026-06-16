"""
Local smoke test for the v2 MPE design. Run on your laptop (conda/venv with
sb3 + supersuit + mpe2 + pettingzoo):

    python experiments/run_mpe_smoke_local.py              # probe only (~30s)
    python experiments/run_mpe_smoke_local.py --train      # + 1 training run (~20-40 min)
    python experiments/run_mpe_smoke_local.py --train --timesteps 150000   # faster, weaker signal

PART 1 (probe, no training): scripted policies through the wrapper stack.
PASS criteria:
  - spread occupancy > 0.6 ;  cluster/still occupancy < 0.2
  - spread r_global/ep > -30 ; cluster r_global/ep < -40

PART 2 (--train): PPO at w=1.0, null condition, v2 training config.
PASS criterion (the one that killed v1):
  - trained occupancy must beat the no-op baseline occupancy by > 0.15
    (v1 trained policies scored BELOW no-op). At 300k this should be clearly
    visible if the 2M HPC runs are going to work; if it fails at 300k, do not
    submit anything -- report back instead.
"""
import sys, os, argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env.mpe_coop_nav import make_mpe_coop_nav, MPELocalRewardWrapper


def act_disc(v, vel):
    # DISCRETE actions (v3): 0=noop, 1=-x, 2=+x, 3=-y, 4=+y.
    # Move along the largest-error axis; brake by no-op when fast.
    u = v - 0.5 * vel
    if np.linalg.norm(v) < 0.1:
        return 0
    if abs(u[0]) >= abs(u[1]):
        return 2 if u[0] > 0 else 1
    return 4 if u[1] > 0 else 3


def run_policy(policy, n_ep=20):
    env = MPELocalRewardWrapper(make_mpe_coop_nav(continuous_actions=False), w=1.0)
    G, O = [], []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=100 + ep)
        g, o, n = 0.0, 0.0, 0
        while env.agents:
            world = env._get_world()
            obs, r, t, tr, info = env.step(policy(env, world))
            a0 = next(iter(info))
            g += info[a0]["r_global"]; o += info[a0]["occupancy"]; n += 1
        G.append(g); O.append(o / n)
    return float(np.mean(G)), float(np.mean(O))


def still(env, world):
    return {a: 0 for a in env.agents}

def cluster(env, world):
    cen = np.mean([ag.state.p_pos for ag in world.agents], axis=0)
    return {a: act_disc(cen - world.agents[i].state.p_pos, world.agents[i].state.p_vel)
            for i, a in enumerate(env.agents)}

def spread(env, world):
    return {a: act_disc(world.landmarks[i].state.p_pos - world.agents[i].state.p_pos,
                        world.agents[i].state.p_vel)
            for i, a in enumerate(env.agents)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    # v4: default to a real horizon. 300k was far too short to test the
    # coordination end (and too short for any LR schedule); the sweep runs 4M.
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    args = ap.parse_args()

    print("=== PART 1: scripted-policy probe ===")
    results = {}
    for name, pol in [("still", still), ("cluster", cluster), ("spread", spread)]:
        g, o = run_policy(pol)
        results[name] = (g, o)
        print(f"  {name:8s} r_global/ep={g:7.1f}   occupancy={o:.3f}")
    ok1 = (results["spread"][1] > 0.6 and results["cluster"][1] < 0.2
           and results["still"][1] < 0.2
           and results["spread"][0] > -30 and results["cluster"][0] < -40)
    print(f"  PART 1: {'PASS' if ok1 else 'FAIL'}")

    if not args.train:
        print("\n(probe only; rerun with --train for the PPO learning check)")
        return

    print(f"\n=== PART 2: PPO w=1.0 null condition, {args.timesteps} steps ===")
    # Shared training path (v4): identical config to sweep + calibration.
    from exp_ppo_mpe import _build_env, evaluate_mpe, train_mpe_model

    model, _ = train_mpe_model(1.0, 1.0, 1.0, 1.0, seed=42,
                               timesteps=args.timesteps, verbose=1)

    eval_env = _build_env(1.0, 1.0, 1.0, 1.0)
    r, occ = evaluate_mpe(model, eval_env, num_episodes=20, base_seed=420)
    noop_occ = results["still"][1]
    print(f"\n  trained: r_global/ep={r:.1f}  occupancy={occ:.3f}")
    print(f"  no-op baseline occupancy={noop_occ:.3f}")
    ok2 = occ > noop_occ + 0.15
    print(f"  PART 2: {'PASS' if ok2 else 'FAIL'} "
          f"(need trained occ > no-op + 0.15)")
    if not ok2:
        print("  -> Do NOT submit the HPC sweeps. Training still failing at w=1.")


if __name__ == "__main__":
    main()
