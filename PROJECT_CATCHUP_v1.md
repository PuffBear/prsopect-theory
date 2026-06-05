# Project Catch-up v1 — Prospect-Theory MARL Supply Chain

*Factual record of methods implemented, experiments run, and results obtained.
No interpretation. Date: 2026-06-03.*

Branch with the latest work: `experiments/w-lambda-collapse-surface`.

---

## 1. Goal and framing (as stated in project docs)

- `docs/design.md`: simulate behavioral phase transitions in a multi-echelon supply
  chain with MARL where agents carry heterogeneous Prospect-Theory (PT) biases;
  hypothesis that a minority of loss-averse agents triggers macroscopic collapse;
  find a critical loss-aversion threshold λ_crit.
- `docs/defense.md`: a separate claim that continuous PPO fails structurally on raw
  continuous order quantities (Gaussian policy vs. zero-inflated optimal base-stock).
- `docs/litreview.md`: literature review (2015–2026) on continuous-control inventory
  pathologies, MARL collapse, and structured base-stock parameterizations.

The executed work uses **continuous PPO (Stable-Baselines3)** throughout (not the
IQL/DQN named in design.md).

---

## 2. Environment and methods implemented

### Base environment — `or_gym_network.py` (`NetworkManagement-v1`)
- Multi-echelon inventory network. `num_periods = 30`.
- Node 0 = market sink; nodes 1–6 = main nodes (hold inventory, place orders);
  nodes 7,8 = raw-material sources.
- Initial inventory I0: node1=100, node2=110, node3=80, node4=400, node5=350,
  node6=380.
- Demand: Poisson μ=20 on retail edge (1,0).
- `uniform_lead_time` kwarg overrides all edge lead times.
- State matrices: P (profit), X (inventory), Y (pipeline), U (unfulfilled), D
  (demand), R (realized orders).

### Multi-agent wrapper — `env/marl_or_gym_wrapper.py`
`MultiAgentNetInvMgmt(ParallelEnv)`. Agents node_1..node_6, each given the full
global observation; per-agent action = order quantities on its reorder links.

### Reward wrapper — `agents/cpt_wrapper.py`
`CPTRewardWrapper(env, agent_params, reward_scale, global_reward_weight)`:
`blended = (1-w)·local + w·global`, then PT value `scaled^α` if ≥0 else
`-λ·(-scaled)^β`. `w = global_reward_weight` (reward centralization). info carries
raw_local, raw_global, blended, scaled, cpt, and true_reward (= raw local).

### Action-reparameterization wrappers
- `env/base_stock_wrapper.py`: action Box[0,500] = target base-stock S;
  order Q = max(0, S − IP), IP = on_hand + pipeline − backlog.
- `env/scaled_base_stock_wrapper.py`: action [-1,1] mapped to [low,high]
  (default [200,700]); optional S/IP/Q logging.
- `env/offset_base_stock_wrapper.py`: S = clip(S_init + action·action_scale, 0, 500).

### Scripting wrappers
- `env/diagnostic_wrapper.py`: hides `scripted_nodes`, injects fixed pull policy
  (node_1=20, others=10).
- `env/intervention_wrapper.py`: hides node_1, injects [10,10].

### Algorithm config
PPO `MlpPolicy`, `n_steps=128`, `batch_size=256`, SuperSuit vectorization
(pad_action_space_v0 → pettingzoo_env_to_vec_env_v1 → concat_vec_envs_v1).
Evaluation deterministic; later experiments use 10 episodes with per-episode seed.

### Frozen collapse definition (`docs/collapse_definition.md`)
`collapsed(run) := (mean_S < 10) AND (profit ≤ −128.1 + 1.0)`.
For the scaled substrate mean_S is the single-reset action mapped via 0.5·(a+1)·500;
for the offset substrate, collapse is labeled from converged_S (last-10k logged S).

---

## 3. Earlier phase experiments (Phases 0–5) and results

### Phase 0 — single-agent PT shaping (`experiments/phase0_validation.py`)
InvManagement-v1, reward/100 with loss-only λ, λ∈{1,2,5,10}, 150k steps.
Result (`docs/results/phase0_data.csv`): λ=1 true reward −275→~+100; λ=2 →~+60;
λ=5 stays negative (~−260 at 150k); λ=10 →~+24..54.

### Phase 1 family (MARL, raw continuous action)
- Pilot (`phase1_runner.py`): φ∈{0,0.5,1.0} (averse λ=5), seeds {0,1}, 250k. All six
  rows identical: bullwhip 0, lost_sales 0.8195, inv_var 25219, economic_reward −11.99.
- Learning diagnostics: order_qty ≈ 0; true_reward floor scales with φ (φ=0 ≈ −363;
  0.5 ≈ −1273; 1.0 ≈ −2589).
- Intervention (`phase1_intervention.py`, node_1 scripted): order ~1.84→1.98, lost
  sales 0.51→0.46, true_reward −128.1→−71.07 over 150k.
- 1.5 diagnosis (`phase1_5_diagnosis.py`): A0 (0 scripted) reward ~−363; A1 (node_1)
  −128.1; A2 (node_1,2,3) +151.66, lost sales ~0.05; B2 (node_2,3) ~+295; B3
  (node_4,5,6) ~+41; C freeze−128.1→−98.65, release back to ~−363.
- 1.5.5 cure (`phase1_5_5_cure.py`): global_reward_weight∈{0,0.1,0.25,0.5,1.0}; all
  stay negative (best w=1.0 → −246.75 at 100k).
- Double sweep (`phase1_double_sweep.py`): {scripted, team_reward}×φ; bullwhip ≈ 0;
  scripted profit −128→−79; team_reward −279→−365.
- Parameter sweep (`parameter_sweep.py`): num_averse 0–6; bullwhip ≈ 0; lost sales
  ~0.834; inv_var ~103 (flat across φ).
- Ablation scaling (`ablation_scaling.py`): reward_scale∈{1,10,100}; order ≈ 0;
  reward ~−363 regardless of scale.

### Phase 2 (raw action, λ sweep, 60% averse)
- Divergence (`phase2_divergence.py`): λ∈{1..10}, single seed; order 0.002–0.26;
  profit −128→−70; no monotonic trend.
- 2a replication (`phase2a_replication.py`): λ∈{1,3,5,10}×seeds 0–4; profit −128 to
  −41; high seed variance.

### Phase 3 family (base-stock reparameterization introduced)
- 3 validation (`phase3_validation.py`): BaseStockWrapper, φ=0, 5 seeds.
- 3.5 challenges (`phase3_5_challenges.py`): empty warehouse, short episode, shifted
  range [200,700].
- 3.6 audit (`phase3_6_audit.py`): logs S/IP/Q; bounds [0,500] vs [200,700];
  P(Q>0) over training.
- 3.6 replication, 3.7 verification (`phase3_7_verification.py`): range ablation
  [0,500]..[300,800] + 20-seed replication on [200,700].
- 3.8 incentives (`phase3_8_incentives.py`): bounds×global_reward_weight {0,.25,.5,1}
  (per-run CSV saving added this session).
- 3.9 lead time (`phase3_9_leadtime.py`): lead∈{0,1,2,4,8}×α{0,1}×5 seeds; records
  value_loss + explained_variance. Result (`leadtime_results.csv`): α=0 profit pinned
  at −128.1 for all lead times, learned S ≈ 0; α=1 profit +93..+389, S ≈ 19–26.
- 3.95 reward landscape (`phase3_95_reward_landscape.py`): static S grid 0..300,
  α{0,1}, 50 episodes.
- 3.96 basin (`phase3_96_basin_attraction.py`): offset wrapper, S_init∈{0..250}.

### Phase 4 (PT λ sweep on base-stock substrate)
- Centralized (`phase4_pt_replication.py`, w=1): λ∈{1,2,3,4,5,7,10}×seeds 42–51 = 70
  runs (`phase4_raw_data.csv`). High variance; some collapse to S≈0/profit −128.1,
  some reach S≈130–186/profit +200..+469.
- Decentralized (`phase4_5_interaction.py`, w=0): λ∈{1,2,3,5,10}×seeds 42–46 = 25
  runs; all identical S=0, profit −128.1, lost sales 0.5148.
- Regression (`phase4_regression_analysis.py`): OLS Mean S ~ log λ: coef −24.99,
  p=0.0041, R²=0.115. Extended stats re-saved this session: η²=0.132; Lost Sales
  coef +0.065 (p=0.005); Profit coef −66.1 (p=0.009).

### Phase 5 factorial (`phase5_final_archiving.py`)
Merged 95 rows (70 centralized + 25 decentralized). Model Y = b0 + b1·logλ + b2·w +
b3·(logλ·w). Mean S: R²=0.358, w coef +95.96 (p<0.001), interaction −24.99 (p=0.064),
logλ main ≈0 (p=1.00). Profit: R²=0.344, w +269.3 (p<0.001), interaction −66.1
(p=0.093). Lost Sales: R²=0.320, w −0.235 (p<0.001), interaction +0.065 (p=0.069).

### Phase 6 (pre-existing, untracked file `experiments/phase6_behavioral_claim.py`)
Heterogeneous-λ minority sweep at w=1.0: num_averse 0–6, 10 seeds, base-stock
substrate. `docs/phase6_figures/phase6_raw_data.csv` present. Not authored in this
session.

---

## 4. This session — the w×λ collapse study

### Day 0 diagnostic (`docs/expA_figures/mean_S_bimodality.png`)
Pooled 95 existing runs (phase4 w=1 + phase4.5 w=0). Mean_S bimodal: 40/95 in [0,10);
collapse count robust to threshold (32 collapsed at S_floor∈{5,10,15}). Endpoint
collapse fractions: w=0 → 25/25 (100%); w=1 → 7/70 (10%).

### Pre-registrations (committed before the runs they govern)
- `docs/collapse_definition.md` — frozen binary collapse label (§2 above).
- `docs/expB_preregistration.md` — frozen Exp B logistic surface model
  `P(collapse) ~ logit(b0+b1·w+b2·logλ+b3·w·logλ)`, w_crit(λ) estimand, headline
  heatmap, and TILTED/VERTICAL decision rule; amendment A1 (w-grid extension).
- `docs/exp0b_preregistration.md` — frozen Q1/Q2 design and §4 branch table.

### Experiment A — interior-w sweep at 150k (`experiments/expA_interior_w.py`)
220 runs: w∈{0,0.1,…,1.0} (11 levels) × seeds 42–61 (20), λ=1, scaled [0,500]
substrate, 150k steps. Data: `docs/expA_figures/expA_raw_data.csv`.

Collapse fraction by w:
| w | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| collapse | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.85 | 0.55 | 0.20 | 0.10 |

Logistic (`analyze_expA.py`): b1 = −15.78 (p=5.0×10⁻⁹, CI [−21.1,−10.5]);
w_crit = 0.822, bootstrap CI [0.785, 0.858]; transition width Δw = 0.278; quadratic
term n.s. (p=0.21).

### Mechanism diagnostic 1 — critic (`docs/expA_figures/expA_robustness_diagnostics.md`)
From logged final_value_loss / final_explained_variance:
- collapsed @ w=1.0 (n=2): value_loss 5363, expvar 0.092.
- recovered @ w=1.0 (n=18): value_loss 6218, expvar 0.086.
- collapsed @ w≤0.6: value_loss 1761, expvar 0.165.
- at w∈{0.7,0.8,0.9}: collapsed value_loss 2448 / expvar 0.275 vs recovered 4340 /
  0.168.

### Mechanism diagnostic 2 — horizon check (w=0.9)
Same 20 seeds, 400k vs 150k. Collapse 4/20 (150k) → 0/20 (400k); median S 0→158,
median profit 62→418; seeds 48,54,56,60 flipped collapsed→recovered.

### Experiment C — basin/initialization control (`experiments/expC_basin.py`)
144 runs: w∈{0.5,0.85,1.0} × S_init∈{0,50,150,250} × 12 seeds, offset wrapper,
λ=1, 150k. Data: `docs/expC_figures/expC_raw_data.csv`.

converged_S (last-10k log mean) by (w, S_init):
| w \ S_init | 0 | 50 | 150 | 250 |
|---|---|---|---|---|
| 0.50 | 1.2 | 1.1 | 0.8 | 0.8 |
| 0.85 | 18.5 | 16.3 | 13.9 | 14.1 |
| 1.00 | 39.5 | 26.0 | 22.4 | 22.6 |

corr(S_init, converged_S) within w: −0.11 (0.5), −0.38 (0.85), −0.36 (1.0).
Corrected collapse (converged_S<10 AND profit≤−127.1): w=0.5 → 1.00; w=0.85 → 0.00
(except S_init=250 → 0.08); w=1.0 → 0.00. (A labeling bug using the scaled mean_S
formula on the offset substrate was found and fixed; collapsed_fixed column added.)

### Experiment B — stopped
Pre-registered (extended grid + λ); launched, **stopped at 20/900 runs**. Partial
data `docs/expB_figures/expB_raw_data.csv` retained.

### Experiment 0 — convergence study (`experiments/exp0_convergence.py`)
24 runs: w∈{0.7,0.8,0.9} × 8 seeds, scaled substrate, trained to 600k, collapse
logged at 150k/300k/600k. Data: `docs/exp0_figures/exp0_convergence_data.csv`.

Collapse fraction by (w, checkpoint):
| w | 150k | 300k | 600k |
|---|---|---|---|
| 0.7 | 0.75 | 0.12 | 0.12 |
| 0.8 | 0.75 | 0.00 | 0.00 |
| 0.9 | 0.38 | 0.00 | 0.00 |

### Converged w_crit re-estimation at 300k (`expA_interior_w.py --timesteps 300000`)
140 runs: w∈{0.2,…,0.8} × seeds 42–61 (20), single learn() to 300k. Data:
`docs/expA300k_figures/expA300k_raw_data.csv`.

Collapse fraction by w:
| w | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|
| collapse | 1.00 | 1.00 | 1.00 | 0.95 | 0.80 | 0.40 | 0.00 |

Logistic: b1 = −22.05 (p=3.2×10⁻⁷, CI [−30.5,−13.6]); w_crit = 0.666, bootstrap CI
[0.635, 0.694]; Δw = 0.199; quadratic n.s. (p=0.43).
Reconciliation with Exp 0 at w=0.7 @300k: this run (20 seeds) = 0.40; Exp 0 (8 seeds,
seeds 42–49) = 0.12; shared seeds 42–49 disagree on 3/8.

### Experiment 0b — horizon lock + metastability origin (`experiments/exp0b_horizon_meta.py`)
104 runs, procs=5, thread-limited (OMP=1). Data: `docs/exp0b_figures/exp0b_raw_data.csv`.

Piece 1 (Q1 horizon), 20 seeds, single learn() to 600k:
| w | 300k | 600k (Wilson CI) |
|---|---|---|
| 0.6 | 0.80 | 0.35 [0.18, 0.57] |
| 0.7 | 0.40 | 0.20 [0.08, 0.42] |

Piece 2 (Q2 origin), per-seed flip rate segmented-vs-single (seeds 42–49):
| w | 0.3 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|
| flip rate | 0/8 | 0/8 | 0/8 | 0/8 |

Frozen §4 result: Q1 = SLIDING (both drops > 0.15); Q2 = INCONCLUSIVE
(transition flip-rate 0.00, deep flip-rate 0.00). Per the branch table this is not a
STABLE row; Experiment B remained gated.

---

## 5. w_crit by training horizon (assembled across experiments)

| horizon | w_crit estimate | source |
|---|---|---|
| 150k | 0.822 [0.785, 0.858] | Exp A (220 runs) |
| 300k | 0.666 [0.635, 0.694] | 300k re-estimation (140 runs) |
| 600k | not estimated as a single value; w=0.6 collapse 0.35, w=0.7 collapse 0.20 | Exp 0b Piece 1 |

---

## 6. Artifacts

### Code (this session)
`experiments/expA_interior_w.py`, `analyze_expA.py`, `analyze_expB.py`,
`expB_run.py`, `expC_basin.py`, `exp0_convergence.py`, `exp0b_horizon_meta.py`;
edits to `experiments/phase3_8_incentives.py` (CSV saving).

### Data / figures (this session)
`docs/expA_figures/` (expA_raw_data.csv, expA_logistic.png, expA_cellfractions.csv,
mean_S_bimodality.png, expA_robustness_diagnostics.md),
`docs/expA300k_figures/` (raw data, logistic png, results.md),
`docs/expC_figures/` (raw data, basin png, results.md),
`docs/exp0_figures/` (convergence data, png, results.md),
`docs/exp0b_figures/` (raw data, horizon-slide png, results.md),
`docs/expB_figures/` (partial raw data),
pre-registrations: `docs/collapse_definition.md`, `docs/expB_preregistration.md`,
`docs/exp0b_preregistration.md`.

### Prior handoffs
`docs/handoff/handoff_v1.md` (full project-wide inventory),
`docs/handoff/handoff_expA.md` (Exp A session).

### Commits (this session, branch `experiments/w-lambda-collapse-surface`)
Pre-registration + Day-0 (03c3fc6); Exp A runner + phase3_8 fix + Phase 4 stats
(46acab0); Exp A analysis (5471d0d); Exp B pre-registration (96e0a1b); Exp A result
(9d100f9); session handoff (1c42773); critic diagnostic + amendment A1 (5329541);
horizon result + C/B runners (bf71bca); Exp C result + bug fix (80d6958); Exp 0
runner (9ee5e51); Exp 0 result (c7ea65b); converged w_crit @300k (3e20379); Exp 0b
pre-registration (403913f); Exp 0b runner (282fc88); Exp 0b result (f46992c).

### Experiment status summary
| experiment | status |
|---|---|
| Phases 0–5 | complete (prior sessions) |
| Day 0 diagnostic | complete |
| Exp A @150k | complete (w_crit=0.822) |
| critic diagnostic | complete |
| horizon check (w=0.9, 400k) | complete |
| Exp C (basin) | complete |
| Exp B (w×λ surface) | stopped at 20/900 |
| Exp 0 (convergence, 600k) | complete |
| converged w_crit @300k | complete (w_crit=0.666) |
| Exp 0b (horizon lock + origin) | complete (Q1 SLIDING, Q2 INCONCLUSIVE) |
| Exp D (heterogeneous-λ minority) | not run (partial pre-existing phase6 file) |
