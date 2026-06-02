# Handoff v1 — Prospect-Theory MARL Supply Chain

*Date: 2026-06-03 · Branch: `main` · Maintainer: A. Yadav*

This document is a complete, factual inventory of the project: its goal, the
methods and code, every dataset and figure produced, the numbers in those
datasets, the analyses run on them, and the open threads. It records what is in
the repository as of this date. It does not contain recommendations or quality
judgments.

---

## Goal: what we are trying to accomplish here

Stated in [docs/design.md](../design.md):

> Simulate **Behavioral Phase Transitions** in supply chains using Multi-Agent
> Reinforcement Learning (MARL) where agents exhibit heterogeneous
> Prospect-Theoretic biases.

Core hypothesis (design.md): a minority of highly loss-averse agents can trigger
macroscopic system collapse (severe bullwhip, inventory starvation, cascading
stockouts) in an otherwise stable supply chain, and there exists a critical
loss-aversion threshold $\lambda_{crit}$ that initiates these cascades.

The Prospect-Theory (PT) value function applied to each agent's reward:

$$v(x) = \begin{cases} x^\alpha & x \ge 0 \\ -\lambda(-x)^\beta & x < 0 \end{cases}$$

design.md also specified Independent Q-Learning / DQN as the algorithm and listed
three metrics: bullwhip effect (variance of orders vs. variance of demand),
systemic starvation (lost-sales frequency/magnitude), and a phase-transition plot
of stability against the distribution of $\lambda$.

A second document, [docs/defense.md](../defense.md), frames a distinct claim that
the executed work also addresses: that **continuous PPO** fails structurally on
raw continuous order quantities in this environment (Gaussian policy vs.
zero-inflated optimal base-stock policy), and that this is a mathematical
limitation rather than a tuning failure. defense.md contains pre-written reviewer
responses on three points: (1) "did you tune it poorly," (2) contradicting SOTA
that reports MARL inventory success, (3) lack of a proposed cure.

[docs/litreview.md](../litreview.md) is a literature review (2015–2026) covering
the Gaussian clipping pathology, zero-inflation, hurdle/two-stage models, MARL
relative overgeneralization and lazy-agent pathologies, Deep Controlled Learning,
GNN/feasibility-enforcement-layer architectures, and structured base-stock
parameterizations, with extraction tables for individual papers (Temizöz et al.
2025, Gijsbrechts et al. 2022, Alvo et al. 2023, and others).

---

## Standing: what exists in the project right now

The work has been executed entirely with **continuous PPO (Stable-Baselines3)**,
not the IQL/DQN named in design.md. It proceeded through a numbered phase
sequence (Phase 0 through Phase 5). The following are present and committed/saved:

**Code**
- 1 base environment ([or_gym_network.py](../../or_gym_network.py), a vendored
  `NetworkManagement-v1`) plus the `or-gym` directory.
- 6 environment/agent wrappers ([env/](../../env/), [agents/](../../agents/)).
- 27 experiment scripts in [experiments/](../../experiments/).
- 12 utility modules in [utils/](../../utils/) (metrics + plotting).
- 6 test scripts at repo root.

**Datasets saved to disk**
- Phase 0: [phase0_data.csv](../results/phase0_data.csv) (122 rows).
- Phase 1 family: [phase1_results_pilot.csv](../results/phase1_results_pilot.csv),
  [learning_diagnostics.csv](../results/learning_diagnostics.csv),
  [phase1_5_diagnosis.csv](../results/phase1_5_diagnosis.csv),
  [phase1_5_5_cure.csv](../results/phase1_5_5_cure.csv),
  [phase1_double_sweep.csv](../results/phase1_double_sweep.csv),
  [sweep_data.csv](../results/sweep_data.csv),
  [intervention_results.csv](../results/intervention_results.csv),
  [ablation_scaling.csv](../results/ablation_scaling.csv),
  6 metadata JSONs in [docs/results/metadata/](../results/metadata/),
  [temp_freeze_model.zip](../results/temp_freeze_model.zip),
  6 agent trace CSVs in [logs/agent_traces/](../../logs/agent_traces/).
- Phase 2: [phase2_divergence.csv](../results/phase2_divergence.csv),
  [phase2a_replication.csv](../results/phase2a_replication.csv).
- Phase 3.9: [leadtime_results.csv](../phase3_9_figures/leadtime_results.csv).
- Phase 4: [phase4_raw_data.csv](../phase4_figures/phase4_raw_data.csv) (70 rows),
  [phase4_5_interaction_data.csv](../phase4_figures/phase4_5_interaction_data.csv)
  (25 rows),
  [regression_results.txt](../phase4_figures/regression_results.txt),
  [extended_regression_results.txt](../phase4_figures/extended_regression_results.txt).
- Phase 5: [master_seed_dataset.csv](../final_archive/master_seed_dataset.csv)
  (95 rows), [final_statistics.txt](../final_archive/final_statistics.txt).

**Figures** — 35 PNGs across [docs/assets/](../assets/) (5),
`docs/phase1_figures/` (12), `docs/phase2_figures/` (4),
`docs/phase2a_figures/` (4), `docs/phase3_5_figures/` (1),
`docs/phase3_6_figures/` (2), `docs/phase3_7_figures/` (1),
`docs/phase3_8_figures/` (1), `docs/phase3_95_figures/` (1),
`docs/phase3_96_figures/` (1), `docs/phase3_9_figures/` (1),
`docs/phase4_figures/` (2).

**Reference material** — 6 PDFs in [docs/papers/](../papers/) (Lin et al. Focal
Loss 2017; Loomes 2003; Novemsky & Kahneman 2005; "Preventing a loss of
accuracy"; Dabney et al. 2018 distributional RL; "RL with Prospect Theory").

**Statuses recorded by phase:**
- Phase 0–2a: completed, datasets saved.
- Phase 3–3.96: completed; some phases print summaries to stdout without saving a
  CSV (3, 3.5, 3.6, 3.6-replication, 3.7, 3.8, 3.95, 3.96 — only figures and/or
  console output persist).
- Phase 4, 4.5: completed, datasets + figures saved.
- Phase 4 regression: run; `regression_results.txt` holds the saved OLS summary;
  `extended_regression_results.txt` holds only the placeholder string
  `"Extended Stats Computed."`.
- Phase 5: run; master dataset + full factorial statistics saved.

**Uncommitted at session start** (git status): `.DS_Store`, the `or-gym`
directory, `commit_all.sh`, `docs/final_archive/`, `docs/handoff/`,
`docs/phase4_figures/`, `experiments/phase4_5_interaction.py`,
`experiments/phase4_regression_analysis.py`,
`experiments/phase5_final_archiving.py`.

The original design.md deliverable — an explicit graded $\lambda_{crit}$ phase
transition in macroscopic stability — does not appear as a saved result. The
final saved statistical artifact is the Phase 5 factorial regression of `Mean S`,
`Profit`, and `Lost Sales` on `log(λ)`, `Alpha` (reward centralization), and their
interaction.

---

## Methods: environment, wrappers, algorithm, and the experiment scripts

### Base environment — [or_gym_network.py](../../or_gym_network.py)
`NetworkManagement-v1`, a multi-echelon inventory network.
- `num_periods = 30`.
- Nodes: node 0 is the market (sink); nodes 1–6 are the "main nodes" (hold
  inventory, place orders); nodes 7, 8 are raw-material sources.
- Initial inventory `I0`: node 1 = 100, node 2 = 110, node 3 = 80, node 4 = 400,
  node 5 = 350, node 6 = 380.
- Demand: Poisson with `mu = 20` on the retail edge (1, 0).
- `uniform_lead_time` kwarg overrides every edge lead time `L` (used in Phase 3.9).
- Per-period node profit is read from `base_env.P`; inventory from `X`; pipeline
  from `Y`; unfulfilled/lost from `U`; demand from `D`; realized orders from `R`.

### Multi-agent wrapper — [env/marl_or_gym_wrapper.py](../../env/marl_or_gym_wrapper.py)
`MultiAgentNetInvMgmt(ParallelEnv)`. Agents = `node_1`..`node_6`. Each agent
receives the **full global observation**. Each agent's action is the order
quantities on its own reorder links (`Box(low=0, high=capacity)`).

### Reward wrapper — [agents/cpt_wrapper.py](../../agents/cpt_wrapper.py)
`CPTRewardWrapper(env, agent_params, reward_scale=1.0, global_reward_weight=0.0)`.
For each agent:
`blended = (1 - w)·local + w·global`, then `scaled = blended / reward_scale`, then
PT: `scaled^α` if `scaled ≥ 0` else `-λ·(-scaled)^β`. `info` carries
`raw_local_reward`, `raw_global_reward`, `blended_reward`, `scaled_reward`,
`cpt_reward`, and `true_reward` (= raw local reward). `w` is
`global_reward_weight`; in several scripts the symbol "Alpha" denotes this `w`
(reward centralization), distinct from the PT curvature `alpha`.

### Action-reparameterization wrappers
- [env/base_stock_wrapper.py](../../env/base_stock_wrapper.py) — action space
  `Box[0, 500]` interpreted as target base-stock `S`; order `Q = max(0, S − IP)`
  where inventory position `IP = on_hand + pipeline − backlog`.
- [env/scaled_base_stock_wrapper.py](../../env/scaled_base_stock_wrapper.py) —
  action `[-1, 1]` mapped to `[low_bound, high_bound]` (default `[200, 700]`);
  optional CSV logging of `step, agent, S, IP, Q`.
- [env/offset_base_stock_wrapper.py](../../env/offset_base_stock_wrapper.py) —
  `S = clip(S_init + action·action_scale, 0, 500)`; centers the action mapping on
  `S_init`; logs the same columns.

### Scripting / intervention wrappers
- [env/diagnostic_wrapper.py](../../env/diagnostic_wrapper.py) — hides
  `scripted_nodes` from PPO and injects a fixed pull policy
  (`target_qty`: node_1 = 20, all others = 10, split across reorder links).
- [env/intervention_wrapper.py](../../env/intervention_wrapper.py) — hides
  `node_1` and injects the constant action `[10, 10]`.

### Algorithm
PPO `MlpPolicy` throughout, `n_steps=128`, `batch_size=256`. SuperSuit pipeline:
`pad_action_space_v0` → `pettingzoo_env_to_vec_env_v1` → `concat_vec_envs_v1`.
Training horizon varies by script (60k / 100k / 150k / 250k timesteps).
Evaluation is deterministic (`model.predict(deterministic=True)`), 3 or 5
episodes; later phases vary the eval seed per episode.

### Utilities — [utils/](../../utils/)
[metrics.py](../../utils/metrics.py) (bullwhip, systemic lost sales),
[phase1_metrics.py](../../utils/phase1_metrics.py),
[learning_diagnostics.py](../../utils/learning_diagnostics.py), and plotters
`plot_ablation`, `plot_cure`, `plot_diagnosis`, `plot_intervention`,
`plot_phase0`, `plot_phase1`, `plot_phase2`, `plot_phase2a`,
`plot_phase3_5_success`, `plot_tipping_point`.

### Experiment scripts (chronological)
| Script | Configuration |
|---|---|
| [phase0_validation.py](../../experiments/phase0_validation.py) | Single-agent `InvManagement-v1`; PT shaping with `reward/100`, loss-only `λ`; λ ∈ {1,2,5,10}; 30×5000 = 150k steps. |
| [parameter_sweep.py](../../experiments/parameter_sweep.py) | MARL; num_averse 0–6 (λ=5 for averse), seeds {42,1337,2026}, 60k steps. |
| [phase1_runner.py](../../experiments/phase1_runner.py) | MARL φ ∈ {0,0.5,1.0} (averse λ=5), seeds {0,1}, 250k; logs traces + metadata. |
| [phase1_intervention.py](../../experiments/phase1_intervention.py) | `InterventionWrapper` (node_1 scripted), nodes 2–6 PPO, λ=1, 150k. |
| [phase1_5_diagnosis.py](../../experiments/phase1_5_diagnosis.py) | Scripted-node sweep A0/A1/A2/B2/B3 and Freeze→Release (C). |
| [phase1_5_5_cure.py](../../experiments/phase1_5_5_cure.py) | No scripted nodes; `global_reward_weight` ∈ {0,0.1,0.25,0.5,1.0}; 100k. |
| [phase1_double_sweep.py](../../experiments/phase1_double_sweep.py) | {scripted, team_reward} × φ ∈ {0,0.25,0.5,0.75,1.0}; 250k, seed 42. |
| [ablation_scaling.py](../../experiments/ablation_scaling.py) | φ ∈ {0,1} × reward_scale ∈ {1,10,100}; logs raw/scaled/cpt reward stats. |
| [phase2_divergence.py](../../experiments/phase2_divergence.py) | Scripted node_1, decentralized; averse = {node_2,4,6}; λ ∈ {1,2,3,4,5,7,10}; 150k, seed 42. |
| [phase2a_replication.py](../../experiments/phase2a_replication.py) | Same, λ ∈ {1,3,5,10} × seeds {0–4}. |
| [phase3_validation.py](../../experiments/phase3_validation.py) | Introduces `BaseStockWrapper`; φ=0, scripted node_1, decentralized; 5 seeds, 150k. |
| [phase3_5_challenges.py](../../experiments/phase3_5_challenges.py) | Base-stock + env mods: empty warehouse (I0=0), short episode (100 periods), shifted range [200,700]. |
| [phase3_6_audit.py](../../experiments/phase3_6_audit.py) | `LoggingBaseStockWrapper`; bounds [0,500] vs [200,700]; logs S/IP/Q; plots histograms + P(Q>0). |
| [phase3_6_replication.py](../../experiments/phase3_6_replication.py) | Shifted base-stock [200,700], 5 seeds. |
| [phase3_7_verification.py](../../experiments/phase3_7_verification.py) | Scaled base-stock; ranges [0,500],[100,600],[200,700],[300,800] (S trajectories) + 20-seed replication on [200,700]. |
| [phase3_8_incentives.py](../../experiments/phase3_8_incentives.py) | Factorial bounds {[0,500],[200,700]} × `global_reward_weight` {0,0.25,0.5,1.0} × 5 seeds. |
| [phase3_9_leadtime.py](../../experiments/phase3_9_leadtime.py) | Lead time {0,1,2,4,8} × α {0,1} × 5 seeds; records value-loss + explained-variance via callback. |
| [phase3_95_reward_landscape.py](../../experiments/phase3_95_reward_landscape.py) | Static evaluation of fixed S = 0..300 (step 10) × α {0,1}, 50 episodes; PPO objective vs global profit. |
| [phase3_96_basin_attraction.py](../../experiments/phase3_96_basin_attraction.py) | Offset base-stock, α=0; S_init {0,25,50,80,150,250} × seeds {42,43,44}; converged S = last-10k mean. |
| [phase4_pt_replication.py](../../experiments/phase4_pt_replication.py) | Scaled base-stock [0,500], **centralized** (PT α=1, w=1); λ ∈ {1,2,3,4,5,7,10} × seeds 42–51 (70 runs); 150k. |
| [phase4_5_interaction.py](../../experiments/phase4_5_interaction.py) | Same, **decentralized** (PT α=0, w=0); λ ∈ {1,2,3,5,10} × seeds 42–46 (25 runs). |
| [phase4_regression_analysis.py](../../experiments/phase4_regression_analysis.py) | OLS of Mean S / Lost Sales / Profit on log(λ); ANOVA η²; writes placeholder text file. |
| [phase5_final_archiving.py](../../experiments/phase5_final_archiving.py) | Merges centralized + decentralized into master dataset; factorial OLS with Cohen's f² and partial η². |
| [phase_transition.py](../../experiments/phase_transition.py), [sb3_phase_transition.py](../../experiments/sb3_phase_transition.py), [visualize_policy.py](../../experiments/visualize_policy.py), [visualize_supply_chain.py](../../experiments/visualize_supply_chain.py) | Phase-transition analysis and visualization scripts. |

---

## Results: the numbers in each dataset

### Phase 0 — [phase0_data.csv](../results/phase0_data.csv)
Single-agent PT shaping, `true_reward` (un-shaped) by training step:
- λ=1.0: rises from −275 (5k) to +60…+112 (≥95k); order qty ~1–2.5; inventory ~42–57.
- λ=2.0: rises from −293 to ~+40…+69 by 150k.
- λ=5.0: remains negative throughout (−279 at 5k; oscillates −150…−286; −262 at 150k).
- λ=10.0: rises from −254 to ~+20…+54 by 150k.

### Phase 1 pilot — [phase1_results_pilot.csv](../results/phase1_results_pilot.csv)
All six rows (φ ∈ {0,0.5,1.0} × seeds {0,1}) are **identical**:
`bullwhip = 0.0`, `lost_sales = 0.8195`, `inventory_variance = 25219.15`,
`economic_reward = −11.99`. Metadata JSONs record the averse node sets
(e.g. φ=0.5 → {node_6, node_3, node_2}).

### Phase 1 learning diagnostics — [learning_diagnostics.csv](../results/learning_diagnostics.csv)
`order_qty ≈ 0` across training for all configs. `true_reward` floor depends on φ:
φ=0 ≈ −363; φ=0.5 ≈ −1273; φ=1.0 ≈ −2589 (CPT-shaped reward becomes more negative
as more agents are loss-averse). Inventory ~221 throughout; lost_sales recorded 0.

### Phase 1 intervention — [intervention_results.csv](../results/intervention_results.csv)
Node_1 scripted, nodes 2–6 PPO (λ=1): order qty ~1.84→1.98; inventory 197→193;
lost_sales 0.51→0.46; `true_reward` improves −128.1 → −71.07 over 150k.

### Phase 1.5 diagnosis — [phase1_5_diagnosis.csv](../results/phase1_5_diagnosis.csv)
Final-block values:
- A0 (0 scripted): order ~0, inv ~221, LS ~0.83, reward ~−363.
- A1 (node_1 scripted): order ~1.83, inv ~196, LS ~0.51, reward −128.1.
- A2 (node_1,2,3 scripted): order ~3.64, inv ~151, LS ~0.05–0.07, reward +151.66.
- B2 (node_2,3 scripted): order ~1.82, inv ~195, LS ~0.83, reward ~+295.
- B3 (node_4,5,6 scripted): order ~2.73, inv ~294, LS ~0.83, reward ~+41.
- C_Freeze (node_1 scripted): reward −128.1 → −98.65.
- C_Release (release to 0 scripted, resumed): order collapses to ~0.03–0.07,
  inv ~220, LS ~0.83, reward ~−363.

### Phase 1.5.5 cure — [phase1_5_5_cure.csv](../results/phase1_5_5_cure.csv)
No scripted nodes, `global_reward_weight` sweep. `true_reward` stays negative for
all weights; highest observed is w=1.0 reaching −246.75 (order qty 1.19) at 100k;
w=0 stays ~−363. None reach positive profit.

### Phase 1 double sweep — [phase1_double_sweep.csv](../results/phase1_double_sweep.csv)
- `scripted`: bullwhip ≈ 0 (1e-8…7e-4); profit −128.1 → −79.5 across φ.
- `team_reward`: bullwhip ≈ 0; profit −279 → −365 across φ.

### Parameter sweep — [sweep_data.csv](../results/sweep_data.csv)
num_averse 0–6: bullwhip ≈ 0 for all but one cell (0.23 at num_averse 0/seed
2026); lost_sales ≈ 0.834; inventory_variance ≈ 103 — flat across φ.

### Ablation scaling — [ablation_scaling.csv](../results/ablation_scaling.csv)
φ ∈ {0,1} × scale ∈ {1,10,100}: order qty ≈ 0; inventory ~221; lost_sales 0;
true_reward ~−363. `scaled_reward = raw/scale`; with φ=1 the `cpt_reward` is ~7×
more negative than raw (λ=5 applied to negative rewards).

### Phase 2 divergence — [phase2_divergence.csv](../results/phase2_divergence.csv)
λ ∈ {1,…,10}, averse={node_2,4,6}, decentralized, single seed: order qty
0.002–0.26; inventory 192–197; lost_sales 0.45–0.50; profit −128.1 → −70.76. No
monotonic trend in λ.

### Phase 2a replication — [phase2a_replication.csv](../results/phase2a_replication.csv)
λ ∈ {1,3,5,10} × seeds 0–4: profit ranges −128.1 to −41.65; high seed-to-seed
variance; many cells sit at the −128.1 floor.

### Phase 3.9 lead time — [leadtime_results.csv](../phase3_9_figures/leadtime_results.csv)
| Lead | α | Mean S | Mean Profit | Mean VLoss | Mean ExpVar |
|---|---|---|---|---|---|
| 0 | 0 | 0.18 | −128.1 | 2279 | 0.019 |
| 0 | 1 | 19.62 | 326.5 | 11707 | 0.031 |
| 1 | 0 | 0.07 | −128.1 | 2231 | 0.017 |
| 1 | 1 | 19.45 | 93.3 | 10766 | 0.020 |
| 2 | 0 | 0.19 | −128.1 | 2235 | 0.020 |
| 2 | 1 | 20.95 | 283.8 | 10699 | 0.017 |
| 4 | 0 | 0.12 | −128.1 | 2195 | 0.034 |
| 4 | 1 | 21.45 | 389.4 | 9415 | 0.012 |
| 8 | 0 | 0.02 | −128.1 | 2164 | 0.054 |
| 8 | 1 | 25.88 | 309.1 | 6823 | 0.077 |

At α=0 (decentralized): learned S ≈ 0 and profit is pinned at −128.1 for every
lead time. At α=1 (centralized): S ≈ 19–26 and profit positive (93–389). Value
loss is several-fold higher at α=1 than α=0.

### Phase 4 centralized — [phase4_raw_data.csv](../phase4_figures/phase4_raw_data.csv)
70 runs (λ ∈ {1,2,3,4,5,7,10} × seeds 42–51). High within-λ variance: many seeds
collapse to `Mean S ≈ 0`, `Profit = −128.1`, `Lost Sales = 0.5148`, `Bullwhip = 0`;
other seeds reach `Mean S ≈ 130–186`, `Profit ≈ +200…+469`, `Bullwhip ≈ 1–5`.
(Examples: λ=1/seed42 → S 183.8, profit 466.4; λ=5/seed49 → S 175.9, profit 468.8;
λ=10/seed45,47,48 → S 0, profit −128.1.)

### Phase 4 OLS — [regression_results.txt](../phase4_figures/regression_results.txt)
`Mean S ~ log(λ)` over 70 obs: intercept 95.96 (SE 12.44); `log(λ)` coefficient
**−24.99** (SE 8.42), 95% CI [−41.78, −8.19], **p = 0.0041**, **R² = 0.115**,
adj-R² = 0.102, F = 8.82. Durbin-Watson 2.43.
[extended_regression_results.txt](../phase4_figures/extended_regression_results.txt)
contains only the string `"Extended Stats Computed."` (the ANOVA η², Lost-Sales,
and Profit regressions computed by the script were printed to stdout, not saved).

### Phase 4.5 decentralized — [phase4_5_interaction_data.csv](../phase4_figures/phase4_5_interaction_data.csv)
All 25 runs (λ ∈ {1,2,3,5,10} × seeds 42–46) are identical:
`Mean S = 0`, `Profit = −128.1`, `Lost Sales = 0.5148`.

### Phase 5 factorial — [final_statistics.txt](../final_archive/final_statistics.txt)
Master dataset [master_seed_dataset.csv](../final_archive/master_seed_dataset.csv)
= 70 centralized (Alpha=1) + 25 decentralized (Alpha=0) = 95 rows (decentralized
rows have NaN for Mean Order / Inventory / Bullwhip). Model:
`Y = b0 + b1·log(λ) + b2·Alpha + b3·(log(λ)·Alpha)`.

| Metric | R² | b1 log(λ) | b2 Alpha | b3 Interaction | Cohen f² | partial η² (interaction) |
|---|---|---|---|---|---|---|
| Mean S | 0.358 | ≈0 (p=1.00) | +95.96 (p<0.001) | −24.99 (p=0.064) | 0.558 | 0.037 |
| Profit | 0.344 | ≈0 (p=1.00) | +269.27 (p<0.001) | −66.13 (p=0.093) | 0.524 | 0.031 |
| Lost Sales | 0.320 | ≈0 (p=1.00) | −0.235 (p<0.001) | +0.065 (p=0.069) | 0.470 | 0.036 |

Constants: Mean S ≈ 0; Profit −128.1 (p=0.006); Lost Sales 0.5148 (p<0.001).

### Figures (saved)
`docs/assets/`: baseline_rational, cascade_averse, phase0_validation, pt_curve,
tipping_point. `phase1_figures/`: bullwhip, order_qty, profit; cure (lost_sales,
order_qty, true_reward); interventions exp_A/B/C; ablation_order_qty;
intervention_order_qty; learning_diagnostics. `phase2_figures/` &
`phase2a_figures/`: inventory, lost_sales, order_qty, profit each.
`phase3_5_figures/`: success_dynamics. `phase3_6_figures/`: action_histograms,
prob_order_over_time. `phase3_7_figures/`: ablation_S_trajectories.
`phase3_8_figures/`: backlog_sharing_trajectories. `phase3_95_figures/`:
reward_landscape. `phase3_96_figures/`: basin_attraction. `phase3_9_figures/`:
leadtime_causality. `phase4_figures/`: pt_replication_ci, pt_interaction.

---

## Analysis: what each analysis examined, and why it was run

This section records the analytical purpose behind each method and what the
corresponding data showed.

**Phase 0 — validate PT shaping on a single agent before MARL.** Purpose: confirm
PPO can learn on the un-distorted task and observe how loss aversion alone changes
learned behavior, on a single-agent environment, before adding multi-agent
coordination. The data show recovery to positive reward for λ ∈ {1,2,10} and a
sustained-negative trajectory for λ=5 within the 150k budget.

**Phase 1 pilot + diagnostics — test the design.md hypothesis directly.** Purpose:
sweep the fraction φ of loss-averse agents (averse λ=5) and measure bullwhip /
lost sales / profit. The saved pilot rows are identical across φ and seed, and the
diagnostics show order quantity ≈ 0 with profit floors that scale with φ through
the CPT shaping. This is why subsequent phases shifted from φ-sweeps to isolating
the cause of the zero-order behavior.

**Phase 1 intervention & Phase 1.5 diagnosis — locate the collapse.** Purpose:
script one or more nodes with a fixed pull policy to remove their learning burden,
and vary *which* nodes are scripted, to see whether collapse is a property of
specific echelons or of the multi-agent learning itself (the "relative
overgeneralization" hypothesis named in `intervention_wrapper.py`). Findings:
scripting more nodes (A2: node_1,2,3) raises profit to +151 and cuts lost sales to
~5%; scripting different node groups (B2 distributors vs B3 factories) changes
profit and inventory levels; releasing a frozen scripted node back to PPO control
(C_Release) returns the system to the −363 floor. The location and freeze/release
contrasts were run to test path dependence and echelon sensitivity of the collapse.

**Phase 1.5.5 cure & double sweep — test reward sharing and a 2-D map.** Purpose:
test whether blending in a global team reward (`global_reward_weight`) prevents
collapse without scripting, and map protocol × φ jointly. The cure sweep shows
profit remains negative for all weights at 100k; the double sweep shows both
`scripted` and `team_reward` protocols collapse (bullwhip ≈ 0) across φ. These
motivated reparameterizing the *action* rather than only the reward.

**Ablation scaling — rule out reward magnitude as the cause.** Purpose: vary
`reward_scale` ∈ {1,10,100} to check whether reward magnitude drives the zero-order
behavior. Order qty stays ≈ 0 across scales; the script also logs raw vs scaled vs
cpt reward to document the λ=5 amplification of negative rewards.

**Phase 2 / 2a — λ-sweep with a fixed averse fraction, plus replication.**
Purpose: with node_1 scripted and 60% of nodes averse, sweep λ and replicate
across seeds to test for a monotonic λ effect on macroscopic metrics under the raw
continuous action. The single-seed sweep shows no monotonic trend; the replication
shows high seed variance with many cells at the −128.1 floor.

**Phase 3 validation — reparameterize the action as base-stock.** Purpose: address
the continuous-action issue described in defense.md/litreview.md (Gaussian policy
vs. zero-inflated optimum) by having the agent output a target base-stock `S`
(`Q = max(0, S − IP)`) instead of a raw order quantity. Run with all-rational
agents (φ=0) across 5 seeds to establish whether the substrate is learnable; the
pass/fail criterion in the script is bullwhip > 0.01.

**Phase 3.5 challenges — stress the base-stock substrate.** Purpose: test the
base-stock parameterization under an empty warehouse (I0=0), a shorter episode
(100 periods), and a shifted action range [200,700], each checked against the
bullwhip > 0.01 criterion, to see which conditions produce non-degenerate ordering.

**Phase 3.6 audit & replication — inspect the action mechanics.** Purpose: log the
internal `S`, inventory position `IP`, and resulting order `Q` per step, and
compare action bounds [0,500] vs [200,700], to characterize when `Q` stays at zero
(`P(Q>0)` over training — labeled the "gradient dead zone" in the script) and
whether the shifted range reproduces higher profit across 5 seeds.

**Phase 3.7 verification — action-range ablation + 20-seed replication.** Purpose:
sweep the base-stock range over [0,500],[100,600],[200,700],[300,800], plot the S
trajectory per range, and run a 20-seed replication on [200,700] to quantify the
mean/SD of profit, order qty, and bullwhip at that range.

**Phase 3.8 incentives — factorial of action bounds × reward sharing.** Purpose:
cross the action bounds {[0,500],[200,700]} with `global_reward_weight`
{0,0.25,0.5,1.0} across 5 seeds, plotting learned-S trajectories per cell, to
separate the contribution of action parameterization from the contribution of
reward centralization.

**Phase 3.9 lead time — causality and credit-assignment metrics.** Purpose: vary
lead time {0,1,2,4,8} crossed with α {0,1}, while recording PPO value loss and
explained variance via a callback, to test whether temporal delay drives collapse.
The saved table shows profit pinned at −128.1 for all lead times at α=0 and
positive at α=1, with value loss several-fold higher at α=1 — i.e., in this dataset
the centralization factor α separates the outcomes, across all lead times.

**Phase 3.95 reward landscape — map the static objective.** Purpose: hold S fixed
at each value 0..300 and evaluate the PPO objective vs. true global profit (50
episodes) for α {0,1}, locating the optimum S, to characterize the shape of the
landscape the agent is optimizing under decentralized vs. centralized reward.

**Phase 3.96 basin of attraction — test initialization dependence.** Purpose: use
the offset wrapper to start S at {0,25,50,80,150,250} (α=0) across seeds and record
the converged S (last-10k mean), to test whether the decentralized policy is drawn
to a particular S regardless of where it starts (the figure marks an S=0 basin and
an S=80 peak as reference lines).

**Phase 4 / 4.5 — PT λ-sweep on the base-stock substrate, centralized vs
decentralized.** Purpose: with the learnable base-stock substrate in place,
re-introduce loss aversion and sweep λ to measure its effect on learned S, profit,
lost sales, and bullwhip — under centralized reward (Phase 4, w=1) and under
decentralized reward (Phase 4.5, w=0). Centralized produces a spread of outcomes
across seeds (collapse to S≈0 and recovery to S≈180 both occur at multiple λ);
decentralized produces an identical S=0 / profit −128.1 outcome for every λ and
seed.

**Phase 4 regression — quantify the centralized λ effect.** Purpose: regress
`Mean S` on `log(λ)` over the 70 centralized runs, with ANOVA η² to compare
λ-variance against seed-variance, and parallel regressions for Lost Sales and
Profit. The saved OLS gives a negative `log(λ)` slope (−24.99, p=0.004) with
R² = 0.115. The η² / Lost-Sales / Profit results were computed but not written to
disk (the output file holds only a placeholder).

**Phase 5 factorial — combine both arms and estimate the interaction.** Purpose:
merge the centralized and decentralized runs and fit
`Y = b0 + b1·log(λ) + b2·Alpha + b3·(log(λ)·Alpha)` for Mean S, Profit, and Lost
Sales, with Cohen's f² and partial η² for the interaction. In all three models the
`log(λ)` main effect is ≈ 0 (p=1.00) — because at Alpha=0 the dependent variables
are constant — the Alpha main effect is large and significant (p<0.001), and the
λ×Alpha interaction term carries the entire λ effect (Mean S interaction = −24.99,
matching the centralized-only slope; p=0.064). R² is 0.32–0.36 across the three
metrics; partial η² for the interaction is 0.031–0.037.

---

## Next: open threads grounded in the current repository state

These are continuations that follow directly from artifacts and gaps already
present; they are listed without ranking.

1. **Unsaved statistics.**
   [phase4_regression_analysis.py](../../experiments/phase4_regression_analysis.py)
   computes the ANOVA η² and the Lost-Sales and Profit regressions but writes only
   `"Extended Stats Computed."` to
   [extended_regression_results.txt](../phase4_figures/extended_regression_results.txt).
   Re-running with the numeric outputs written to disk would persist them.

2. **Reward-centralization resolution.** The Phase 4/4.5/5 arms use
   `global_reward_weight` at the endpoints {0, 1} only.
   [phase3_8_incentives.py](../../experiments/phase3_8_incentives.py) already sweeps
   intermediate weights {0, 0.25, 0.5, 1.0} but does not save its per-cell results
   to CSV (console + figure only). Running the λ-sweep across intermediate weights,
   or saving the Phase 3.8 cells, would populate the interior of the Alpha axis the
   Phase 5 model currently fits with only two levels.

3. **Heterogeneous λ.** Phase 4/4.5/5 use homogeneous λ across all nodes. The
   φ-fraction design from design.md (and the averse-subset logic in
   [phase2_divergence.py](../../experiments/phase2_divergence.py) and
   [phase1_runner.py](../../experiments/phase1_runner.py)) has not been re-run on
   the base-stock substrate.

4. **Phases without saved tables.** Phase 3, 3.5, 3.6, 3.6-replication, 3.7, 3.8,
   3.95, and 3.96 print summaries to stdout and/or save figures but do not write
   result CSVs. Their numeric outputs are not currently recoverable from disk
   without re-running.

5. **Seed count.** The centralized arm uses 10 seeds (42–51); the decentralized
   arm uses 5 (42–46); Phase 3.7 demonstrates a 20-seed protocol that has not been
   applied to the Phase 4 configurations.

6. **design.md deliverable.** A graded $\lambda_{crit}$ phase-transition plot in
   macroscopic stability — the original target metric — has not been produced as a
   saved artifact.

---

### Reproduction notes
- Run all scripts from the repository root; paths are relative to CWD.
- Multiprocessing scripts set `spawn` start method in `__main__`.
- PPO settings are uniform: `MlpPolicy`, `n_steps=128`, `batch_size=256`.
- The PT `alpha`/`beta` curvature parameters are held at 1.0 in every experiment;
  only `lambda` (loss aversion) and `global_reward_weight` (centralization) are
  varied. The token "Alpha" in Phase 3.8–5 refers to `global_reward_weight`.
- `venv/` is checked into the repo (SB3, SuperSuit, PettingZoo, gymnasium,
  statsmodels, or_gym).
