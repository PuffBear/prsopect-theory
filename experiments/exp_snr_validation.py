"""
Mean-field SNR validation -- magnitude test of the phase-boundary theory.

The mean-field model (Sec. Mean-Field Analysis) claims the base-stock policy
mean mu obeys a Langevin equation  d mu = D(w) dt + sqrt(2 Theta(w)) dW, and that
w_crit is the signal-to-noise crossing  SNR(w) = D(w)/sqrt(Theta(w)) ~ kappa_c.

This harness MEASURES D(w) and Theta(w) directly from real PPO training (no
static-env approximation): it logs the policy's mean action every update and
estimates, over an early-training window starting from the neutral init,
    drift     D(w)     = mean increment of mu per update
    diffusion Theta(w) = Var(increment) / 2
    SNR(w)             = D(w) / sqrt(Var(increment))
Theory prediction (falsifiable): SNR(w) -- and in particular the drift sign --
crosses zero / its critical value at the empirical w_crit ~ 0.666.

Structural condition only (alpha=beta=1, lambda=1) so the magnitude is compared
to the structural boundary 0.666.  Mirrors exp_b_reduced.py (same substrate,
PPO config, filelock CSV, run_idx array job).

Outputs one summary row per run AND the full mu/sigma trajectory (npz) so the
drift/diffusion window can be re-chosen at analysis time without re-training.
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd
from filelock import FileLock

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from expA_interior_w import (evaluate_and_extract_metrics, is_collapsed,
                             LOW_BOUND, HIGH_BOUND)
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from env.diagnostic_wrapper import DiagnosticWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from env.scaled_base_stock_wrapper import ScaledBaseStockWrapper

# Fine w-grid around the structural boundary so the crossing is well resolved.
W_LIST = [0.50, 0.55, 0.60, 0.65, 0.666, 0.70, 0.75, 0.80, 0.90]
SEEDS = list(range(42, 62))          # 20
ALPHA, BETA, LAM = 1.0, 1.0, 1.0     # structural condition
N_W, N_SEED = len(W_LIST), len(SEEDS)
N_RUNS = N_W * N_SEED                 # 180
S_FLOOR, PROFIT_CEIL = 10.0, -127.1

COLUMNS = ["run_idx", "w", "seed", "drift_D", "diffusion_Theta", "snr",
           "mu0", "mu_end", "net_displacement", "collapsed", "profit", "mean_S",
           "window_updates"]


class MuTrajectoryCallback(BaseCallback):
    """Logs, per PPO update (rollout), the policy's mean deterministic action mu
    (averaged over the rollout-buffer observations and action dims) and the
    exploration scale sigma = exp(log_std).  These are the Langevin observables."""
    def __init__(self):
        super().__init__()
        self.mu, self.sigma, self.t = [], [], []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        obs = buf.observations  # [n_steps, n_envs, obs_dim]
        flat = obs.reshape(-1, obs.shape[-1])
        # deterministic (mean) action under current policy
        act, _ = self.model.predict(flat, deterministic=True)
        self.mu.append(float(np.mean(act)))                 # mean normalized action in [-1,1]
        try:
            log_std = self.model.policy.log_std.detach().cpu().numpy()
            self.sigma.append(float(np.mean(np.exp(log_std))))
        except Exception:
            self.sigma.append(float("nan"))
        self.t.append(int(self.num_timesteps))


def _build_env(w, seed):
    raw = MultiAgentNetInvMgmt()
    interv = DiagnosticWrapper(raw, scripted_nodes=["node_1"])
    params = {a: {"lambda": LAM, "alpha": ALPHA, "beta": BETA} for a in raw.agents}
    cpt = CPTRewardWrapper(interv, params, reward_scale=1.0, global_reward_weight=w)
    return ScaledBaseStockWrapper(cpt, low_bound=LOW_BOUND, high_bound=HIGH_BOUND)


def map_run_idx(run_idx):
    return W_LIST[run_idx // N_SEED], SEEDS[run_idx % N_SEED]


def estimate_langevin(mu, window):
    """Drift / diffusion / SNR from the first `window` updates of the mu series."""
    mu = np.asarray(mu[:window], dtype=float)
    if mu.size < 3:
        return float("nan"), float("nan"), float("nan")
    dmu = np.diff(mu)
    D = float(np.mean(dmu))                 # drift per update
    var = float(np.var(dmu, ddof=1))        # increment variance
    Theta = var / 2.0
    snr = D / np.sqrt(var) if var > 0 else float("nan")
    return D, Theta, snr


def run_one(run_idx, timesteps, window_steps, out_path):
    if not (0 <= run_idx < N_RUNS):
        raise ValueError(f"run_idx must be in [0,{N_RUNS-1}]")
    w, seed = map_run_idx(run_idx)
    np.random.seed(seed); warnings.filterwarnings("ignore")

    train_env = _build_env(w, seed)
    vec = ss.pad_action_space_v0(train_env)
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class='stable_baselines3')
    vec.seed = lambda s: None

    model = PPO("MlpPolicy", vec, verbose=0, n_steps=128, batch_size=256, seed=seed)
    cb = MuTrajectoryCallback()
    model.learn(total_timesteps=timesteps, callback=cb)

    # measurement window: updates whose cumulative timestep < window_steps
    t = np.asarray(cb.t)
    window = int(np.sum(t <= window_steps)) if t.size else 0
    window = max(window, 3)
    D, Theta, snr = estimate_langevin(cb.mu, window)

    # outcome (full-horizon) for cross-reference with the empirical boundary
    eval_env = _build_env(w, seed)
    mean_S, _mo, _mi, lost_sales, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))

    mu0 = float(cb.mu[0]) if cb.mu else float("nan")
    mu_end = float(np.mean(cb.mu[window-3:window])) if len(cb.mu) >= window else float("nan")

    # persist full trajectory for re-analysis
    traj_dir = os.path.join(os.path.dirname(out_path), "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    np.savez(os.path.join(traj_dir, f"run_{run_idx:04d}.npz"),
             w=w, seed=seed, t=np.asarray(cb.t), mu=np.asarray(cb.mu),
             sigma=np.asarray(cb.sigma))

    row = {"run_idx": run_idx, "w": w, "seed": seed, "drift_D": D,
           "diffusion_Theta": Theta, "snr": snr, "mu0": mu0, "mu_end": mu_end,
           "net_displacement": mu_end - mu0, "collapsed": collapsed,
           "profit": profit, "mean_S": mean_S, "window_updates": window}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} w={w} seed={seed} -> D={D:+.2e} Theta={Theta:.2e} "
          f"SNR={snr:+.3f} net_disp={mu_end-mu0:+.3f} collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    """Idempotent per run_idx: a row replaces any existing row with the same
    run_idx rather than blind-appending. This keeps array-job resubmits (and the
    leftover smoke-pilot rows) from leaving duplicate run_idx rows, which would
    skew analyze_snr.py's per-w means (it does not dedup on load)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with FileLock(out_path + ".lock"):
        if os.path.exists(out_path):
            df = pd.read_csv(out_path)
            df = df[df["run_idx"] != row["run_idx"]]
            df = pd.concat([df, pd.DataFrame([row], columns=COLUMNS)],
                           ignore_index=True)
        else:
            df = pd.DataFrame([row], columns=COLUMNS)
        df.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_idx", type=int, required=True, help=f"0..{N_RUNS-1}")
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--window_steps", type=int, default=60_000,
                    help="early-training window over which drift/diffusion are measured")
    ap.add_argument("--out", default="docs/expSNR_figures/expSNR_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.window_steps, args.out)


if __name__ == "__main__":
    main()
