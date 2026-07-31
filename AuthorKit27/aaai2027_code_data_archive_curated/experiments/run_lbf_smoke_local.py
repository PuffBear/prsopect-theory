"""
Local smoke test for the LBF v2 design -- MAPPO (centralized critic) GATE.

    python experiments/run_lbf_smoke_local.py            # probe (~30s)
    python experiments/run_lbf_smoke_local.py --train    # + MAPPO w=0/w=1 (~5-15 min)

WHY MAPPO HERE: PPO/IPPO on LBF failed to coordinate. Before committing a night
to the full lambda x w sweep we need to know whether that failure was task
sparsity (LBF is dead, pick another 3rd domain) or just the missing centralized
critic (MAPPO fixes it, sweep is worth running). This is the GATE, not the result
-- "MAPPO coordinates on LBF" only licenses the sweep; the breadth result is the
sweep showing w_crit rises with lambda.

PART 0 (with --train): MAPPO wiring check -- centralized critic built, trainable,
  reads the joint-state slice (guards against silently falling back to IPPO).
PART 1: random-policy probe through the full stack (nonzero team return = plumbing
  + learning signal present).
PART 2 (--train): trains BOTH endpoints (w=0 and w=1), null condition, with MAPPO.
  PASS: w=1 team return >= 0.7 absolute (random ~0.46; max ~1.0)
  AND >= w=0 team return + 0.15.
  (w=0 should free-ride under the effort cost; w=1 should coordinate. Miniature
   of the calibration gate -- if w=1 flatlines like RWARE, LBF is dead.)
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
    from exp_ppo_lbf import _build_env as _build_ppo_env, evaluate_lbf, N_ENV_COPIES, PPO_KWARGS
    from env.global_state_wrapper import GlobalStateWrapper
    from agents.mappo_policy import CentralizedCriticPolicy, _CentralCriticExtractor

    def _build_mappo_env(w, lam, alpha, beta):
        # MAPPO = the SAME LBF stack with GlobalStateWrapper OUTERMOST so the
        # centralized critic can read the joint state; rewards/info pass through,
        # so evaluate_lbf (uses info['true_reward']) is unchanged.
        return GlobalStateWrapper(_build_ppo_env(w, lam, alpha, beta))

    def _build_mappo_vec(w, lam, alpha, beta, seed):
        env = _build_mappo_env(w, lam, alpha, beta)
        local_dim = env.local_dim
        vec = ss.pad_action_space_v0(env)
        vec = ss.pad_observations_v0(vec)
        vec = ss.pettingzoo_env_to_vec_env_v1(vec)
        vec = ss.concat_vec_envs_v1(vec, N_ENV_COPIES, num_cpus=0,
                                    base_class='stable_baselines3')
        _inner = getattr(vec, "venv", vec)
        vec.seed = lambda s, _v=_inner: _v.reset(seed=int(s))
        return vec, _inner, local_dim

    def _build_mappo_model(w, lam, alpha, beta, seed):
        vec, _inner, local_dim = _build_mappo_vec(w, lam, alpha, beta, seed)
        policy_kwargs = dict(local_dim=local_dim,
                             net_arch=dict(pi=[64, 64], vf=[128, 128]))
        model = PPO(CentralizedCriticPolicy, vec, verbose=0, seed=seed,
                    policy_kwargs=policy_kwargs, **PPO_KWARGS)
        return model, _inner

    class Heartbeat(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 200_000 < 48:
                print(f"    ... {self.num_timesteps:,} steps", flush=True)
            return True

    # --- PART 0: MAPPO wiring check (centralized critic is real) -----------
    print("\n=== PART 0: MAPPO wiring check (centralized critic reads joint state) ===")
    m0, _ = _build_mappo_model(1.0, 1.0, 1.0, 1.0, seed=42)
    pol = m0.policy
    obs_dim = m0.observation_space.shape[0]
    ldim = pol._local_dim
    assert isinstance(pol.vf_features_extractor, _CentralCriticExtractor), \
        "vf extractor is NOT the centralized critic (silently IPPO!)"
    opt_ids = {id(p) for g in pol.optimizer.param_groups for p in g["params"]}
    crit = list(pol.vf_features_extractor.parameters())
    assert crit and all(id(p) in opt_ids for p in crit), \
        "centralized critic params are NOT in the optimizer!"
    assert pol.vf_features_extractor.net[0].in_features == obs_dim - ldim, \
        "critic input != global-state slice"
    print(f"  local_dim={ldim} augmented_obs={obs_dim} global_slice={obs_dim - ldim}")
    print("  PART 0: PASS (centralized critic wired, trainable, reads joint state)")

    results = {}
    for w_end in (0.0, 1.0):
        print(f"\n=== PART 2: MAPPO w={w_end} null condition, {args.timesteps} steps ===")
        model, _inner_vec = _build_mappo_model(w_end, 1.0, 1.0, 1.0, seed=42)
        _inner_vec.reset(seed=42)
        model.learn(total_timesteps=args.timesteps, callback=Heartbeat())
        eval_env = _build_mappo_env(w_end, 1.0, 1.0, 1.0)
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
