"""
SNR mechanism, take 2 -- the PROBE-STATE observable (predictive-coordinate hunt).

The first SNR sweep reproduced the boundary (collapse-fraction crossing = 0.822)
but the Langevin observable -- mean action mu averaged over the whole rollout
buffer AND all agents -- did NOT separate survivors from collapsers, so the
mechanism stayed sign-and-form (see docs/expSNR_figures/snr_mechanism_diagnosis.md).

DIAGNOSIS: the grand mean averages mu over the states the policy VISITS, and
collapsing vs healthy policies visit DIFFERENT states -- so the average mixes
"where the policy IS" with "where it goes" (a state-distribution confound). The
single most informative axis the env exposes is inventory position (IP): the
same base-stock policy behaves very differently at low vs high IP (Q=max(0,S-IP)).

FIX (this driver): log the deterministic policy's action at a FIXED set of probe
states chosen by IP percentile. Each PPO update we harvest the rollout buffer,
pick the low / median / high IP observations (real, on-manifold states), freeze
them, and evaluate mu(s*) at each. Holding the probe states fixed removes the
confound; the IP percentile resolves the policy's action AS A FUNCTION of state,
which is what the Langevin coordinate should be.

NOTE (verified empirically before the sweep): the substrate is a SYMMETRIC
multi-echelon net under a SHARED policy, so per-agent mu(s*) at a fixed state is
identical across the non-scripted nodes (agents that take equal actions reach
equal IP; std across agents = 0). The per-agent axis is therefore degenerate and
is NOT a separate coordinate -- we average mu over agents at each probe level
purely to denoise. The live coordinate is the IP-probe level (lo/med/hi).

Same grid / horizon / outcome label / idempotent CSV as exp_snr_validation.py so
the boundary reference stays 0.822 (150k) and the two sweeps are directly
comparable. analyze_snr_probe.py tests the restoring-drift crossing on each probe
coordinate and reports which (if any) lands at w_crit.

  9 w x 20 seeds = 180 runs   (PBS array 0..179)
"""
import os, sys, argparse, warnings
import numpy as np, pandas as pd
from filelock import FileLock

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Reuse the EXACT validated SNR substrate, grid, outcome metric, idempotent append
from exp_snr_validation import (
    _build_env, map_run_idx, estimate_langevin,
    N_RUNS, S_FLOOR, PROFIT_CEIL,
)
from expA_interior_w import evaluate_and_extract_metrics

# Inventory-position proxy: dim 2 carries the on-hand/IP-like quantity that spans
# the widest real range (see the obs-probe inspection); we rank probe states by it.
IP_DIM = 2
PROBE_PCTS = (10, 50, 90)               # low / median / high inventory position
PROBE_LABELS = ("lo", "med", "hi")

COLUMNS = ["run_idx", "w", "seed", "collapsed", "profit", "mean_S",
           "window_updates",
           # restoring-drift summary on the BEST single scalar (median probe,
           # agent-averaged) for a quick at-a-glance row; the full per-probe,
           # per-agent trajectory lives in the npz for analyze_snr_probe.py.
           "drift_med", "snr_med", "mu_med0", "mu_med_end", "net_disp_med"]


class ProbeTrajectoryCallback(BaseCallback):
    """Per PPO update, evaluate the deterministic policy at fixed inventory-
    position percentile probe states, PER AGENT. Logs an array of shape
    [n_updates, n_probes, n_agents] for mu(s*), plus the legacy grand-mean mu and
    sigma for backward comparison with the first sweep."""
    def __init__(self, n_agents):
        super().__init__()
        self.n_agents = n_agents
        self.mu_probe = []      # each entry: [n_probes, n_agents]
        self.mu_grand = []      # legacy scalar
        self.sigma = []
        self.t = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        obs = buf.observations                      # [n_steps, n_envs, obs_dim]
        flat = obs.reshape(-1, obs.shape[-1])       # rows are interleaved agents
        # legacy grand mean (identical definition to exp_snr_validation.py)
        act_all, _ = self.model.predict(flat, deterministic=True)
        self.mu_grand.append(float(np.mean(act_all)))
        try:
            log_std = self.model.policy.log_std.detach().cpu().numpy()
            self.sigma.append(float(np.mean(np.exp(log_std))))
        except Exception:
            self.sigma.append(float("nan"))

        # PROBE: pick fixed-percentile states by the IP proxy, evaluate per agent.
        # supersuit interleaves agents row-wise (row i -> agent i % n_agents), so
        # we slice each agent's rows, rank them by IP_DIM, take the percentile
        # observations, and read the deterministic action there.
        na = self.n_agents
        probe_mu = np.full((len(PROBE_PCTS), na), np.nan)
        for a in range(na):
            rows = flat[a::na]
            if rows.shape[0] < len(PROBE_PCTS):
                continue
            ipv = rows[:, IP_DIM]
            order = np.argsort(ipv)
            for j, pct in enumerate(PROBE_PCTS):
                k = order[min(int(pct / 100.0 * (len(order) - 1)), len(order) - 1)]
                s_star = rows[k:k + 1]
                act, _ = self.model.predict(s_star, deterministic=True)
                probe_mu[j, a] = float(np.mean(act))
        self.mu_probe.append(probe_mu)
        self.t.append(int(self.num_timesteps))


def run_one(run_idx, timesteps, window_steps, out_path):
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
    cb = ProbeTrajectoryCallback(n_agents)
    model.learn(total_timesteps=timesteps, callback=cb)

    t = np.asarray(cb.t)
    window = max(int(np.sum(t <= window_steps)) if t.size else 0, 3)

    # at-a-glance scalar: median probe, agent-averaged restoring drift
    probe = np.asarray(cb.mu_probe)                 # [n_upd, n_probes, n_agents]
    med_series = np.nanmean(probe[:, 1, :], axis=1) if probe.size else np.array([])
    D_med, _, snr_med = estimate_langevin(med_series, window)
    mu_med0 = float(med_series[0]) if med_series.size else float("nan")
    mu_med_end = (float(np.nanmean(med_series[window - 3:window]))
                  if med_series.size >= window else float("nan"))

    eval_env = _build_env(w, seed)
    mean_S, _mo, _mi, _ls, profit, _bw = evaluate_and_extract_metrics(
        model, eval_env, num_episodes=10, base_seed=seed * 10)
    collapsed = bool((mean_S < S_FLOOR) and (profit <= PROFIT_CEIL))

    traj_dir = os.path.join(os.path.dirname(out_path), "probe_trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    np.savez(os.path.join(traj_dir, f"run_{run_idx:04d}.npz"),
             w=w, seed=seed, t=t, mu_probe=probe,
             mu_grand=np.asarray(cb.mu_grand), sigma=np.asarray(cb.sigma),
             probe_pcts=np.asarray(PROBE_PCTS), n_agents=n_agents)

    row = {"run_idx": run_idx, "w": w, "seed": seed, "collapsed": collapsed,
           "profit": profit, "mean_S": mean_S, "window_updates": window,
           "drift_med": D_med, "snr_med": snr_med, "mu_med0": mu_med0,
           "mu_med_end": mu_med_end, "net_disp_med": mu_med_end - mu_med0}
    _append_row(row, out_path)
    print(f"run_idx={run_idx} w={w} seed={seed} -> drift_med={D_med:+.2e} "
          f"snr_med={snr_med:+.3f} net_disp_med={mu_med_end - mu_med0:+.3f} "
          f"collapsed={collapsed}")
    return row


def _append_row(row, out_path):
    """Idempotent per run_idx (same contract as exp_snr_validation._append_row):
    a row replaces any existing row with the same run_idx, so array resubmits do
    not leave duplicates that would skew the analyzer's per-w means."""
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
    ap.add_argument("--out", default="docs/expSNR_figures/expSNR_probe_raw_data.csv")
    args = ap.parse_args()
    run_one(args.run_idx, args.timesteps, args.window_steps, args.out)


if __name__ == "__main__":
    main()
