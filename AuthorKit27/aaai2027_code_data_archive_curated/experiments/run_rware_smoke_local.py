"""
Local smoke test for the v2 RWARE design. Run on your laptop:

    python experiments/run_rware_smoke_local.py            # env probe (~1 min)
    python experiments/run_rware_smoke_local.py --train    # + PPO at w=1 (~10-30 min)
    python experiments/run_rware_smoke_local.py --train --timesteps 2000000

PART 1: env probe -- random policy through the full wrapper stack; verifies
  reset/step/info plumbing and reports baseline deliveries (expected ~0).
PART 2 (--train): PPO at w=1.0 (pure delivery), null condition, v2 config.
  PASS: mean deliveries/episode >= 2.0 (clearly above the random ~0 baseline).
  This is the sparse-reward dead-end check -- if PPO can't find deliveries at
  this horizon locally, the HPC sweep cannot work either.
"""
import sys, os, argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env.rware_wrapper import make_rware_env_with_w


def random_baseline(n_ep=5):
    env = make_rware_env_with_w(1.0)
    D = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=100 + ep)
        d = 0.0
        while env.agents:
            obs, r, t, tr, info = env.step(
                {a: env.action_space(a).sample() for a in env.agents})
            d += sum(info[a]["true_reward"] for a in info)
        D.append(d)
    return float(np.mean(D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--timesteps", type=int, default=600_000)
    args = ap.parse_args()

    print("=== PART 1: env probe (random policy, 5 episodes) ===")
    base = random_baseline()
    print(f"  random-policy deliveries/episode = {base:.3f}  (expected ~0)")
    print("  PART 1: PASS (stack runs end-to-end)")

    if not args.train:
        print("\n(probe only; rerun with --train for the sparse-reward check)")
        return

    print(f"\n=== PART 2: PPO w=1.0 null condition, {args.timesteps} steps ===")
    import supersuit as ss
    from stable_baselines3 import PPO
    from exp_ppo_rware import _build_env, evaluate_rware, N_ENV_COPIES, PPO_KWARGS

    env = _build_env(1.0, 1.0, 1.0, 1.0)
    vec = ss.pad_action_space_v0(env)
    vec = ss.pad_observations_v0(vec)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                base_class='stable_baselines3')
    vec.seed = lambda s: None
    model = PPO("MlpPolicy", vec, verbose=1, seed=42, **PPO_KWARGS)
    model.learn(total_timesteps=args.timesteps)

    eval_env = _build_env(1.0, 1.0, 1.0, 1.0)
    d = evaluate_rware(model, eval_env, num_episodes=10, base_seed=420)
    print(f"\n  trained deliveries/episode = {d:.3f}  (random baseline {base:.3f})")
    ok = d >= 2.0
    print(f"  PART 2: {'PASS' if ok else 'FAIL'} (need >= 2.0 deliveries/episode)")
    if not ok:
        print("  -> Sparse-reward dead-end at this horizon. Do NOT submit the sweep;")
        print("     report back (options: longer horizon, reward shaping for pickup,")
        print("     or an easier RWARE variant).")


if __name__ == "__main__":
    main()
