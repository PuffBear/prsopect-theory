"""
Local smoke test for the LBF v2 design.

    python experiments/run_lbf_smoke_local.py            # probe (~30s)
    python experiments/run_lbf_smoke_local.py --train    # + PPO w=1 (~5-15 min)

PART 1: random-policy probe through the full stack (expect team return
  ~0.1-0.3/episode; verifies plumbing and a nonzero learning signal).
PART 2 (--train): trains BOTH endpoints (w=0 and w=1), null condition.
  PASS: w=1 team return >= 0.7 absolute (random ~0.46; max ~1.0)
  AND >= w=0 team return + 0.15.
  (w=0 should free-ride under the effort cost; w=1 should coordinate.
   This is a miniature of the calibration gate.)
"""
import sys, os, argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env.lbf_wrapper import make_lbf_env_with_w


def random_baseline(n_ep=20):
    env = make_lbf_env_with_w(1.0)
    T = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=100 + ep)
        R = 0.0
        while env.agents:
            obs, r, t, tr, info = env.step(
                {a: env.action_space(a).sample() for a in env.agents})
            R += sum(info[a]["true_reward"] for a in info)
        T.append(R)
    return float(np.mean(T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    args = ap.parse_args()

    print("=== PART 1: random-policy probe ===")
    base = random_baseline()
    print(f"  random team return/episode = {base:.3f}")
    ok1 = base > 0.0
    print(f"  PART 1: {'PASS' if ok1 else 'FAIL'} (need nonzero signal)")

    if not args.train:
        print("\n(probe only; rerun with --train for the learning check)")
        return

    import supersuit as ss
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from exp_ppo_lbf import _build_env, evaluate_lbf, N_ENV_COPIES, PPO_KWARGS

    class Heartbeat(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 200_000 < 48:   # 48 = 8 copies x 2 agents x 3? safe window
                print(f"    ... {self.num_timesteps:,} steps", flush=True)
            return True

    results = {}
    for w_end in (0.0, 1.0):
        print(f"\n=== PART 2: PPO w={w_end} null condition, {args.timesteps} steps ===")
        env = _build_env(w_end, 1.0, 1.0, 1.0)
        vec = ss.pad_action_space_v0(env)
        vec = ss.pad_observations_v0(vec)
        vec = ss.pettingzoo_env_to_vec_env_v1(vec)
        vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                    base_class='stable_baselines3')
        # reproducible env-spawn seeding (v3 fix): see exp_ppo_lbf.py
        _inner_vec = getattr(vec, "venv", vec)
        vec.seed = lambda s, _v=_inner_vec: _v.reset(seed=int(s))
        model = PPO("MlpPolicy", vec, verbose=0, seed=42, **PPO_KWARGS)
        _inner_vec.reset(seed=42)
        model.learn(total_timesteps=args.timesteps, callback=Heartbeat())
        eval_env = _build_env(w_end, 1.0, 1.0, 1.0)
        results[w_end] = evaluate_lbf(model, eval_env, num_episodes=20,
                                      base_seed=420)
        print(f"  trained w={w_end}: team return/episode = {results[w_end]:.3f}")

    r0, r1 = results[0.0], results[1.0]
    print(f"\n  w=0 -> {r0:.3f}   w=1 -> {r1:.3f}   (random baseline {base:.3f})")
    ok2 = (r1 >= 0.7) and (r1 >= r0 + 0.15)
    print(f"  PART 2: {'PASS' if ok2 else 'FAIL'} "
          f"(need w=1 >= 0.7 and w=1 >= w=0 + 0.15)")
    if not ok2:
        print("  -> Do NOT submit the HPC sweeps; report back with these numbers.")


if __name__ == "__main__":
    main()
