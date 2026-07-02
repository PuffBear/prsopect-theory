"""
SNR mechanism, take 3 -- the NON-SURVIVOR-BIASED probe.

Context. The first SNR sweep (exp_snr_validation.py) reproduced the boundary
(collapse-fraction crossing at w=0.822, 150k) but the restoring-drift observable
did not localize w_crit. exp_snr_probe.py removed the state-distribution confound
with fixed IP-percentile probe states, and got closer, but a residual SURVIVOR
BIAS remained: the PPO Gaussian policy initializes at action mean ~0, which under
the base-stock map S = 250*(a+1) (LOW=0, HIGH=500) starts the policy at S~250 --
ABOVE the cooperative optimum (S~150). Collapsing runs therefore plunge DOWN
through the contested band (S~150-200) too fast to sample, so the early-window
drift is dominated by survivors that linger high. The conditioned drift never
sees the contested band at rest.

FIX (this driver, single variable changed): re-initialize the policy NEAR the
cooperative optimum by biasing the actor's output mean to a_0 = -0.4
(=> S = 250*(1-0.4) = 150). Training then begins INSIDE the contested band, so
both survivors and collapsers are sampled there during the slow early phase and
the restoring drift can localize w_crit. Everything else -- substrate, grid,
horizon, outcome label, IP-percentile probe observable, Langevin estimator,
idempotent CSV -- is identical to exp_snr_probe.py, so this is a clean
single-variable test of the survivor-bias hypothesis.

Pre-registered decision rule (fix BEFORE running): the mechanism is corroborated
at the MAGNITUDE level iff the restoring-drift zero-crossing's bootstrap 95% CI
(over seeds, on the best probe coordinate) contains the matched-horizon boundary
w_crit = 0.822. A null result (CI excludes 0.822, or no clean crossing) leaves
the mechanism at the sign-and-form level, as currently reported. Either outcome
is reported in full.

  --init_S sets the target initial base-stock (default 150). --init_action
  overrides the bias directly. 9 w x 20 seeds = 180 runs (PBS array 0..179).
"""
import os, sys, argparse, warnings
import numpy as np, pandas as pd
from filelock import FileLock

import supersuit as ss
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Reuse the EXACT probe substrate, grid, outcome metric, Langevin estimator,
# IP-probe callback, and idempotent append from exp_snr_probe.py -- only the
# policy initialization changes here.
from exp_snr_validation import (
    _build_env, map_run_idx, estimate_langevin,
    N_RUNS, S_FLOOR, PROFIT_CEIL,
)
from exp_snr_probe import (
    ProbeTrajectoryCallback, PROBE_PCTS, PROBE_LABELS, COLUMNS, _append_row,
)
from expA_interior_w import evaluate_and_extract_metrics, LOW_BOUND, HIGH_BOUND


def action_for_S(target_S):
    """Invert the ScaledBaseStockWrapper map S = LOW + 0.5*(a+1)*(HIGH-LOW)."""
    a = 2.0 * (target_S - LOW_BOUND) / (HIGH_BOUND - LOW_BOUND) - 1.0
    return float(np.clip(a, -1.0, 1.0))


def bias_policy_to_action(model, a0):
    """Shift the Gaussian actor's initial mean to a0 by setting the policy net's
    final-layer bias, so the deterministic action at the untrained policy is ~a0
    (=> initial base-stock S = LOW + 0.5*(a0+1)*(HIGH-LOW)). Leaves all weights
    and the value net untouched; this only moves the starting point of the
    Langevin coordinate, exactly the single variable under test."""
    import torch
    with torch.no_grad():
        # SB3 ActorCriticPolicy: action mean = action_net(latent_pi). Setting the
        # action_net bias to a0 and not touching its weights makes the mean output
        # a0 at initialization (latent features are ~0-mean at init, so the bias
        # dominates the untrained mean).
        net = model.policy.action_net
        if hasattr(net, "bias") and net.bias is not None:
            net.bias.fill_(float(a0))
        else:  # fallback: last Linear in a Sequential
            for layer in reversed(list(net.modules())):
                if isinstance(layer, torch.nn.Linear):
                    layer.bias.fill_(float(a0)); break


def run_one(run_idx, timesteps, window_steps, init_action, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}]")
    w, seed = map_run_idx(run_idx)
    np.random.seed(seed); warnings.filterwarnings("ignore")

    train_env = _build_env(w, seed)
    n_agents = len(train_env.possible_agents)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    # --- THE SINGLE CHANGE vs exp_snr_probe.py: start near the cooperative optimum
    bias_policy_to_action(model, init_action)

    cb = ProbeTrajectoryCallback(n_agents)
    model.learn(total_timesteps=timesteps, callback=cb)

    t = np.asarray(cb.t)
    window = max(int(np.sum(t <= window_steps)) if t.size else 0, 3)

    probe = np.asarray(cb.mu_probe)                  # [n_upd, n_probes, n_agents]
    med_series = np.nanmean(probe[:, 1, :], axis=1) if probe.size else np.array([])
    D_med, _, snr_med = estimate_langevin(med_series, window)
    mu_med0 = float(med_series[0]) if med_series.size else float("nan")
    mu_med_end = (float(np.nanmean(med_series[window - 3:window]))
                  if med_series.size >= window else float("nan"))

    eval_env = _build_env(w, seed)
    mean_S, _mo, _mi, _ls, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))

    traj_dir = os.path.join(os.path.dirname(out_path), "reprobe_trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    np.savez(os.path.join(traj_dir, f"run_{run_idx:04d}.npz"),
             w=w, seed=seed, t=t, mu_probe=probe, init_action=init_action,
             mu_grand=np.asarray(cb.mu_grand), sigma=np.asarray(cb.sigma),
             probe_pcts=np.asarray(PROBE_PCTS), n_agents=n_agents)

    row = {"run_idx": run_idx, "w": w, "seed": seed, "collapsed": collapsed,
           "profit": profit, "mean_S": mean_S, "window_updates": window,
           "drift_med": D_med, "snr_med": snr_med, "mu_med0": mu_med0,
           "mu_med_end": mu_med_end, "net_disp_med": mu_med_end - mu_med0}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} w={w} seed={seed} init_a={init_action:+.2f} -> "
          f"drift_med={D_med:+.2e} snr_med={snr_med:+.3f} "
          f"mu0={mu_med0:+.3f} collapsed={collapsed}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help=f"0..{N_RUNS-1}")
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--window_steps", type=int, default=60_000)
    ap.add_argument("--init_S", type=float, default=150.0,
                    help="target initial base-stock (cooperative optimum ~150)")
    ap.add_argument("--init_action", type=float, default=None,
                    help="override: set the initial actor mean directly in [-1,1]")
    ap.add_argument("--out", default="docs/expSNR_reprobe_figures/expSNR_reprobe_raw_data.csv")
    args = ap.parse_args()
    a0 = args.init_action if args.init_action is not None else action_for_S(args.init_S)
    run_one(args.run_idx, args.timesteps, args.window_steps, a0, args.out)


if __name__ == "__main__":
    main()
