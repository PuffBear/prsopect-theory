# Reviewer-response experiments (Points 2–4)

These three experiment drivers extend the paper to address reviewer points 2–4.
They mirror `exp_b_reduced.py` exactly: same OR-Gym substrate, same
`[0,500]` base-stock action, same eval (`evaluate_and_extract_metrics`,
10 episodes), and the same pre-registered collapse label
(`mean_S < 10` AND `profit <= -127.1`). Each is a SLURM array job; one task
per `run_idx`, appending one CSV row under a filelock.

| File | Reviewer point | Grid | Runs |
|------|----------------|------|------|
| `exp_shapley.py` | 2 — structural credit assignment | {difference, shapley} × λ∈{1,7} × 20 seeds | 80 |
| `exp_heterogeneous.py` | 3 — heterogeneous cognitive profiles | 4 configs × 5 w × 20 seeds | 400 |
| `exp_entropy_ablation.py` | 4 — exploration noise | 4 ent_coef × 5 w × 20 seeds | 400 |
| `exp_snr_validation.py` | mean-field magnitude test | 9 w × 20 seeds | 180 |

## Run

Single run (smoke):
```bash
python experiments/exp_shapley.py --run_idx 0 --timesteps 20000
```

Full sweep (SLURM array), e.g. Shapley:
```bash
sbatch --array=0-79 run_array.sh experiments/exp_shapley.py 300000
# heterogeneous: --array=0-399 ... exp_heterogeneous.py
# entropy:       --array=0-399 ... exp_entropy_ablation.py
```
(Use the same array wrapper you used for `exp_b_reduced.py`; each task calls
`--run_idx $SLURM_ARRAY_TASK_ID`.)

## What each tests

**Point 2 — `exp_shapley.py`.** Replaces the scalar-`w` blend with
`env/shapley_reward_wrapper.py`, which credits each agent with its marginal
contribution to system profit. Counterfactual baseline = "agent orders zero",
the same lazy reference point as the collapse attractor. Two modes:
`difference` (leave-one-out / Wonderful-Life / COMA-style, 1 counterfactual per
agent) and `shapley` (Monte-Carlo over coalitions, `n_perm=8`; exact when
`n_perm ≥ N!`). RQ: does structural credit assignment make the system robust to
loss aversion — i.e. does the collapse boundary vanish even at λ=7? Compare the
collapse fraction against the scalar-`w` λ=7 boundary (w_crit=0.851).

**Point 3 — `exp_heterogeneous.py`.** Mixed populations via the per-agent
params already supported by `CPTRewardWrapper`. Configs isolate the *number* and
*position* of loss-averse (λ=7) nodes against a λ=1 background:
`single_down` (one downstream node), `single_up` (one upstream node),
`half_down`, and `homog_LA` (reference). RQ: does a single loss-averse node
cascade upstream and shift w_crit for the whole system? Estimate w_crit per
config with the same logistic estimator and compare to the homogeneous
boundaries (λ=1→0.586, λ=7→0.851).

**Point 4 — `exp_entropy_ablation.py`.** Sweeps PPO `ent_coef` at fixed λ=1.
RQ: is w_crit driven by behavioral bias or by exploration noise overwhelming the
diluted signal at low `w`? This directly tests the mean-field prediction
(Sec. *Mean-Field Analysis*) that the boundary tracks the drift-to-noise ratio,
so w_crit should rise with `ent_coef`.

## Mean-field magnitude test — `exp_snr_validation.py`

This is the experiment that upgrades the theory from "predicts the *shape* of the
boundary" to "predicts its *magnitude*." The mean-field model says the policy
mean μ obeys a Langevin equation `dμ = D(w)dt + √(2Θ)dW` and that w_crit is the
signal-to-noise crossing `SNR(w) = D(w)/√Θ(w) ~ κ_c`. The harness **measures
D(w) and Θ(w) directly from real PPO training** (no static-env proxy): a callback
logs the policy's mean action every update; over an early-training window we
estimate drift = mean increment, diffusion = increment-variance/2, SNR =
drift/√variance. Structural condition only (α=β=1, λ=1) so it compares to the
structural boundary 0.666.

Run:
```bash
# smoke (single point, short)
python experiments/exp_snr_validation.py --run_idx 80 --timesteps 30000 --window_steps 15000
# full sweep
sbatch --array=0-179 run_array.sh exp_snr_validation.py    # 150k steps, 60k window (defaults)
# then:
python experiments/analyze_snr.py --csv docs/expSNR_figures/expSNR_raw_data.csv --kappa_c 0.0
```

**Falsifiable prediction:** the measured drift sign-change / SNR crossing lands at
w ≈ 0.666 (the empirical structural boundary). `analyze_snr.py` reports the
crossing with a run-level bootstrap 95% CI and states whether 0.666 falls inside
it. Full μ/σ trajectories are saved per run (`trajectories/run_XXXX.npz`) so the
drift/diffusion window can be re-chosen at analysis time without re-training.

Notes for the run:
- Start with the default `window_steps=60000`. If the drift is dominated by
  transient init, sweep the window (e.g. 30k/60k/90k) using the saved
  trajectories — `estimate_langevin` is re-runnable offline.
- `κ_c` defaults to 0, which tests the drift **sign-change** boundary (the
  cleanest claim). If the SNR curve is offset, set `--kappa_c` to the measured
  O(1) value; report both.
- This validates the *structural* magnitude. The λ-axis is already validated
  analytically (RMSE 0.008); to also validate λ-magnitudes, rerun with the Exp-B
  CPT params at λ∈{3,7} and compare crossings to 0.774 / 0.851.

## Run 1 outcomes (300k, 20 seeds)

- **Entropy ablation — confirmed.** w_crit rises with the entropy coefficient
  (0.685 / 0.690 / 0.715 / >1 for ent=0/0.001/0.01/0.05); ent=0 reproduces the
  null. Promoted to the paper (App. "Robustness", Fig. entropy_ablation).
- **Heterogeneous — confirmed and novel.** One downstream loss-averse node
  reaches the fully-loss-averse boundary (0.853 vs 0.851); one upstream node is
  far milder (0.726). Loss aversion cascades upstream, not downstream. Promoted
  to the paper (App. "Robustness", Fig. heterogeneous).
- **Shapley — first run was degenerate (FIXED, re-run needed).** All 80 runs
  collapsed to the identical floor (profit -128.1, mean_S 0) with critic
  explained_variance ~ -40: the unnormalised, high-variance counterfactual
  reward blew up the PPO value head. Fix applied in `exp_shapley.py`:
  `VecNormalize(norm_reward=True, clip_reward=10)` on the training reward (eval
  still uses raw profit, so the collapse label is unaffected; CPT/lambda is
  applied before normalisation, so the behavioural signal is preserved).
  **Re-run the 80-job array after syncing the updated script.**

## Validation status

- **Shapley/difference combinatorics**: unit-tested against a known 3-player
  cooperative game — exact Shapley, efficiency (Σφ = grand-coalition value), and
  leave-one-out difference rewards all verified (see `/tmp/test_shapley.py`
  logic; reproduce with any synthetic characteristic function).
- **Run-index maps**: verified bijective onto the full grid for all three
  drivers (80 / 400 / 400).
- **Syntax**: all modules `py_compile`-clean.
- **NOT yet validated on the live env** (the dev sandbox lacks the RL stack):
  the `ShapleyRewardWrapper` counterfactual path deep-copies the base OR-Gym env
  and pins the global NumPy RNG state so the demand realization is held fixed
  across coalitions. **Before launching the full Shapley sweep, do one smoke run
  and confirm** (a) `info[agent]["shapley_credit"]` is finite and varies across
  agents, and (b) Σ credit ≈ realized global profit in `difference`/exact modes.
  If demand is sampled through a generator other than the global
  `np.random` state, pin that generator instead in `_coalition_global_profit`.
