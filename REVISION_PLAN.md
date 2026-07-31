# Revision Plan — exact edits for AnonymousSubmission2027_2 + SupplementaryMaterial2027_2

Concrete LaTeX changes for review points 1–8 of the 9-point plan (point 9 omitted per
instruction). Each edit names the file, the anchor (section / line in the `_2.tex`), the
exact text to insert or replace, and its page-budget cost. **No files are edited here —
this is the spec only.**

## Page budget starting point
- **Main paper (`AnonymousSubmission2027_2`)**: currently **9 pages**. AAAI-27 limit is
  **7 pages body + references** (references + reproducibility/ethics excluded from the 7).
  The 9 pages include refs, so body is at/near the cap already — **main-paper additions
  must be net-neutral or offset by cuts.**
- **Supplement (`SupplementaryMaterial2027_2`)**: currently **8 pages**, **no page limit**
  — this is where new material lands freely.

**Strategy: put results *statements* + one small table in the main paper (offset by
trimming already-hedged prose); put all apparatus, formulas, and full tables in the
supplement (unbounded).**

---

## Point 1 — Fold in Controls A/B/C + seed-paired bootstrap  *(highest value)*

**Review point:** Major §1 ("loss aversion confounded with reward magnitude, variance,
reference-point"); change-items #1 and #5.

### 1a. MAIN PAPER — new short paragraph in §Results (`sec:results`)
Insert after the paragraph `Two effect sizes answer two questions.` (ends ~line 334,
before `\begin{figure}` for `boundary_metastability`):

```latex
\paragraph{The shift survives magnitude, variance, and reference-point controls.}
Because $\lambda$ also rescales reward magnitude and variance, we ran three matched
controls on the supply chain (identical seeds, $w$-grid, optimizer; full design in the
Supplement). (i)~\emph{Reference point}: re-centering the gain--loss reference from $0$ to
the empirical mean ($-2.99$), median ($-6.77$), or $-0.5\sigma$ ($-5.14$) preserves the
$\lambda{=}1{\to}7$ rise in every case ($+0.228$, $+0.183$, $+0.193$; all bootstrap CIs
exclude zero, vs.\ $+0.269$ at reference $0$). (ii)~\emph{Reward-scale match}: normalizing
the transformed reward to a common RMS across $\lambda$ (removing the $\approx4\times$
magnitude growth from $\lambda{=}1$ to $7$) leaves a significant residual shift
($+0.145$ $[0.107,0.185]$), so the effect is not reward amplification alone.
(iii)~\emph{Reward-unit sensitivity}: doubling the local or the team reward before blending
moves the absolute boundary (as expected---$w$ is a nominal mixture coefficient) but
preserves the $\lambda$ direction in every arm. All difference intervals are seed-paired
percentile bootstrap ($2{,}000$ resamples, resampling unit the shared seed), which also
separates evaluation from training-seed variability.
```

- **Budget:** ~11 lines ≈ **0.20 col** (~0.10 page). **Offset:** trim the already-hedged
  affine-law caveat sentences in `sec:mechanism` (the two "Two caveats bound the claim"
  sentences, ~4 lines) and the redundant second half of the `Two effect sizes` paragraph
  → net ~**0**.

### 1b. SUPPLEMENT — new section `\section{Reward-Transform Controls}` after
`\S sec:s_robustness` (Threshold Robustness, ~line 700). Full apparatus + one table:

```latex
\section{Reward-Transform Controls}
\label{sec:s_controls}
The loss-aversion treatment also changes reward magnitude, variance, and (through the
fixed zero reference) the gain/loss split. We isolate these with three matched controls on
the supply chain, holding the environment, PPO configuration ($\S\ref{sec:s_setup}$),
$w$-grid, seeds (42--61), and collapse label fixed; only the named factor moves.
All differences are seed-paired percentile bootstrap ($2{,}000$ resamples; the same seed
multiset is drawn for both $\lambda$ endpoints within each replicate, so the reported
interval is on the paired $\lambda{=}1{\to}7$ difference and separates evaluation-episode
noise from training-seed noise). Reference points and the RMS scale were measured on
$\lambda{=}1$ (untreated) data and frozen before the sweeps (pre-registration records
released with the code).

\emph{Control~A (reference point).} The value transform is not invariant to adding a
constant, so we re-run $\lambda\in\{1,3,7\}$ with the gain/loss reference $z_0$ shifted to
four frozen values and re-anchor $v(x-z_0)$. \emph{Control~B (scale match).} We divide the
transformed reward by a per-$\lambda$ constant equal to its RMS relative to $\lambda{=}1$
(measured RMS $7.93,17.06,31.87$ at $\lambda{=}1,3,7$), matching magnitude across
conditions. \emph{Control~C (reward units).} We multiply the local, or the team, reward by
$2$ before blending, changing the nominal-to-effective mixing ratio.

[TABLE tab:s_controls here]

Reading: the $\lambda{=}1{\to}7$ shift is positive with a zero-excluding CI in every arm of
A and B and in all three arms of C, so it is not an artifact of the reference point or of
reward magnitude/variance. In C the absolute $\wcrit$ moves ($+0.194$ at $\lambda{=}1$
under $2\times$ local, $-0.109$ under $2\times$ team), confirming that $w$ is a nominal
mixture coefficient whose numerical boundary is conditional on reward parameterization;
the normalized local-to-team incentive ratio at the boundary is the scale-invariant
summary and is reported in the released tables.
```

Table (drop in place of `[TABLE ...]`):

```latex
\begin{table}[t]\centering\small
\caption{Reward-transform controls (supply chain, PPO, 300k, 20 seeds). Shift is the
seed-paired $\lambda{=}1{\to}7$ change in $\wcrit$ (Firth), CI = 2{,}000-resample paired
bootstrap. Baseline (reference $0$, unit scale) shift is $+0.269$ $[0.232,0.306]$.}
\label{tab:s_controls}
\begin{tabular}{llcc}
\toprule
Control & Arm & $\wcrit$($\lambda{=}1$) & Shift $\lambda{=}1{\to}7$ \\
\midrule
A ref.\ point & $z_0={-}2.99$ (mean)   & 0.605 & $+0.228\,[0.187,0.272]$ \\
A ref.\ point & $z_0={-}6.77$ (median) & 0.626 & $+0.183\,[0.122,0.269]$ \\
A ref.\ point & $z_0={-}5.14$ ($-0.5\sigma$) & 0.621 & $+0.193\,[0.141,0.247]$ \\
B scale match & RMS-matched            & 0.585 & $+0.145\,[0.107,0.185]$ \\
C reward unit & $2\times$ local        & 0.779 & $+0.130\,[0.105,0.153]$ \\
C reward unit & $2\times$ team         & 0.476 & $+0.329\,[0.263,0.430]$ \\
\bottomrule
\end{tabular}
\end{table}
```

- **Budget (supplement):** ~1 section + 1 table ≈ **0.6 page**. No limit → free.
- **Run inventory update:** add A(1200)+B(600)+C(900)=**2{,}700 runs** to the supplement
  inventory table `tab:experiments` (three rows) and update the "Total" (2,364 → **5,064**).

---

## Point 2 — Isotonic + probit estimator robustness + goodness-of-fit

**Review point:** Major §4 / change-item #3 ("no comparison with isotonic, probit,
splines"; "no goodness-of-fit"); "CIs may depend on model form."

### 2a. MAIN PAPER — one sentence in §Method (`sec:method`), appended to the paragraph
ending `\ldots agrees with the bootstrap estimate to within $0.019$ at every $\lambda$.`
(~line 243):

```latex
The location is not an artifact of the logistic link: fitting the same cells with a probit
link and with isotonic (nonparametric monotone) regression reproduces $\wcrit$ to within
$0.02$ at every $\lambda$ on all three substrates (Supplement), and the $\lambda{=}1{\to}7$
shift is unchanged under all three estimators.
```

- **Budget:** ~3 lines ≈ **0.05 page**. Offset by deleting the now-redundant standalone
  "sharp" defense sentence flagged in Point 6 → net ~0.

### 2b. SUPPLEMENT — subsection under `\S sec:s_results`:

```latex
\subsection{Estimator robustness (link function and nonparametric fit)}
\label{sec:s_estimator}
We refit every $\wcrit$ under three estimators: logistic (primary), probit, and isotonic
regression (monotone-decreasing in $w$, no functional form). Table~\ref{tab:s_estimator}
reports the primary supply-chain sweep and the two transfer substrates. Across all
$\lambda$ and all three environments the estimates agree to $\le0.02$, and the isotonic
$\wcrit$---which assumes only monotonicity---tracks the logistic to within one grid step,
so the reported location and its CI are not consequences of the logistic link.
McFadden pseudo-$R^2$ for the logistic fits is $\ge0.6$ at every interior cell.
[TABLE: per-lambda wcrit under logistic / Firth / isotonic / probit, 3 envs]
```

- **Budget (supplement):** ~0.4 page (subsection + a 3-env × estimator table). Free.

---

## Point 3 — Full hyperparameters

**Review point:** Major §3 ("could not find learning rate, discount, GAE, clip, epochs,
net widths, normalization, # parallel envs, init").

### 3 SUPPLEMENT ONLY — replace the `\paragraph{Algorithm and evaluation.}` block
(`sec:s_setup`, ~lines 67--74) with an explicit table. Current text says "all remaining
PPO hyperparameters left at Stable-Baselines3 defaults" — replace with:

```latex
\paragraph{Algorithm and evaluation.}
Each trained node runs PPO~\citep{schulman2017ppo} on a shared-parameter MLP policy via
Stable-Baselines3~2.8.0~\citep{raffin2021sb3}; all hyperparameters are listed in
Table~\ref{tab:s_hparams} (values not shown are SB3~2.8.0 defaults, enumerated there for
completeness). MAPPO uses the identical settings with a centralized (joint-state) critic.
Evaluation is deterministic over 10 episodes with 20 fixed seeds per condition; the
10-episode evaluation estimates each trained policy's collapse label, while cross-run
uncertainty (the bootstrap CIs) is driven by the 20 training seeds, which the seed-paired
bootstrap ($\S\ref{sec:s_controls}$) keeps separate from evaluation noise.

\begin{table}[t]\centering\small
\caption{Full PPO/MAPPO hyperparameters (Stable-Baselines3 2.8.0). Only \texttt{n\_steps}
and \texttt{batch\_size} are set explicitly; the rest are SB3 defaults, listed for
reproducibility.}
\label{tab:s_hparams}
\begin{tabular}{ll}
\toprule
Hyperparameter & Value \\
\midrule
Policy / value net & MLP, 2$\times$64, $\tanh$ (shared params across nodes) \\
Centralized critic (MAPPO) & joint-state input, same 2$\times$64 MLP \\
Learning rate & $3\times10^{-4}$ (constant) \\
Rollout length \texttt{n\_steps} & 128 \\
Minibatch size & 256 \\
Update epochs & 10 \\
Discount $\gamma$ & 0.99 \\
GAE $\lambda$ & 0.95 \\
Clip range & 0.2 \\
Value clip & none \\
Entropy coef. & 0.0 (except entropy ablation) \\
Value-loss coef. & 0.5 \\
Max grad norm & 0.5 \\
Advantage normalization & on \\
Parallel envs & 1 (one core/run; SuperSuit concat) \\
Policy init & SB3 default (orthogonal, gain $\sqrt2$) \\
Optimizer & Adam \\
\bottomrule
\end{tabular}
\end{table}
```

- **Budget (supplement):** ~0.5 page (replaces ~0.15 page of prose) → net +0.35 page. Free.

---

## Point 4 — MPE herding wrapper formula + LBF config/threshold

**Review point:** Major §3 ("MPE wrapper formula and numerical scale absent"; "LBF env
config, reward wrapper, numerical threshold absent").

### 4a. SUPPLEMENT — in `\S sec:s_generalization`, MPE subsection (`sec:s_generalization`,
after "we construct one with a herding wrapper", ~line 493). Replace the prose scale
mention with the explicit formula:

```latex
The herding wrapper defines each agent's local reward as
\begin{equation}
r_i^{\mathrm{local}}=-\,\kappa_{\mathrm{herd}}\cdot
\tfrac{1}{N-1}\sum_{j\ne i}\lVert p_i-p_j\rVert_2,\qquad \kappa_{\mathrm{herd}}=4,
\end{equation}
with $p_i$ agent $i$'s position and $\kappa_{\mathrm{herd}}$ the herding scale; the raw env
uses \texttt{local\_ratio}${=}0$ so $r^{\mathrm{global}}$ is the pure team score. The scale
$\kappa_{\mathrm{herd}}{=}4$ places the theoretical indifference point at $w^\ast\approx
0.62$ (from $(1{-}w)\,\overline{r_{\mathrm{herd}}}=w\,\overline{r_{\mathrm{global}}}$ with
measured means $42.3$ and $26.3$), so the no-transform boundary is interior; it was fixed
before the $\lambda$-sweep. Collapse is mean landmark occupancy below $\theta{=}0.144$
(fraction of landmarks within a fixed radius of some agent, averaged over evaluation),
an occupancy metric invariant to reward scale.
```

### 4b. SUPPLEMENT — LBF subsection (`sec:s_generalization`, ~line 518). Add config block:

```latex
Concretely LBF is \texttt{Foraging-5x5-2p-2f-v3}~\citep{christianos2020shared}: a
$5\times5$ grid, $2$ players, $2$ foods, with per-agent rewards equal to each loader's
share of a food (normalized so loading all foods sums to a fixed total). The team return
used for the label is the sum of per-agent shares over an evaluation episode; collapse is
team return below $\theta{=}0.626$, the calibrated collapsed-versus-coordinated midpoint.
The $2$-player $5\times5$ configuration is deliberately small so that the $w{=}1$
team-summed reward remains learnable.
```

- **Budget (supplement):** ~0.3 page across the two blocks. Free.

---

## Point 5 — Anonymized code artifact

**Review point:** Major §3 / change-item #2 ("no executable artifact"; "commit hash
withheld... not independently verifiable").

### 5a. SUPPLEMENT — one paragraph at the end of `\S sec:s_setup`:

```latex
\paragraph{Code and artifact.}
An anonymized code bundle accompanies this submission: the environment wrappers, all
experiment drivers and analysis scripts, the frozen collapse-label and control
pre-registration records, and the raw per-run result CSVs sufficient to regenerate every
table and figure. The de-anonymized repository with its pre-registration commit history is
released on acceptance.
```

### 5b. NON-LATEX action (no page cost): assemble the bundle — drivers, analyzers,
`docs/*_figures/*_raw_data.csv`, the pre-registration `.md` records, a README, with author
paths/names stripped. (Deliverable, not a paper edit.)

- **Budget:** ~4 lines in supplement ≈ negligible. Free.

---

## Point 6 — Soften mechanism + metastability language

**Review point:** Major §2 ("mechanism assumes its conclusion"; "metastability language
stronger than the evidence"); minor ("phase boundary" invites physical analogy).

### 6a. MAIN PAPER — `sec:mechanism`, retitle and reframe. Change the section header
(line 378) from:
`\section{Mechanism: Loss Aversion Lowers Escape SNR}` →
`\section{An Interpretive SNR Account of the Sign}`
And change the opening clause (line 381) "this section adds a scoped mean-field account" →
"this section offers an \emph{interpretive} account (not a mechanism proof) of the sign".

### 6b. MAIN PAPER — metastability paragraph (`sec:results`, line 278). Change the header
`\paragraph{The boundary is finite-horizon and metastable.}` →
`\paragraph{The boundary is finite-horizon (consistent with metastable escape).}`
and append one sentence at the end of that paragraph:
```latex
We use ``metastable'' for this horizon-dependent slide and its Kramers-type reading;
direct basin-level evidence (dwell-time or escape-time distributions, hysteresis) is not
measured here and is left to future work.
```

### 6c. MAIN PAPER — after Proposition~1 discussion (line 450), the sentence already says
"its content is premise~(ii), an empirical claim... rather than a theorem." Strengthen the
framing by changing "The measurement above is the paper's contribution; this section adds"
to explicitly label the whole section a hypothesis (covered by 6a). No new lines.

### 6d. MINOR — keep "phase boundary" but the intro already defines it operationally
(line 115); add ", i.e.\ a finite-training collapse threshold," at first use (line 116)
inline. ~0 lines.

- **Budget:** wording-only, **net ~0 lines** (a retitle + ~2 sentences, offset by nothing
  needed). Reduces over-claiming at no page cost.

---

## Point 7 — Two missing citations + positioning

**Review point:** Major §5 ("should cover DE-MADDPG and packet-routing mixed local/global
reward work").

### 7a. MAIN PAPER — `sec:related`, "Coordination collapse and credit assignment"
paragraph (~line 139), append:
```latex
Combined individual-and-team reward design has direct precedent---DE-MADDPG separates a
per-agent and a team critic for exactly such mixtures, and earlier packet-routing work
tunes mixed local/global signals~\citep{demaddpg2020,packetrouting}---but these set the
mixture as a fixed design choice rather than \emph{locating} the centralization threshold
$\wcrit$ or giving it a confidence interval, which is our object.
```

### 7b. NON-LATEX — add the two entries to `references.bib` (`demaddpg2020`,
`packetrouting`). ~0 page cost in body; refs are outside the 7-page limit.

- **Budget:** ~4 lines in Related Work ≈ **0.07 page**. Offset by the Point-6 retitle
  savings / already-tight related work; or move one existing sentence to supplement.

---

## Point 8 — Direct metastability evidence  *(medium; optional within this plan)*

**Review point:** Major §2 / change-item #4 ("bimodal state distributions, escape-time
distributions, dwell times, hysteresis").

> NOTE: This is the one item here that needs **new analysis (and likely light new
> logging / a small qsub job)**, not just writing. Include only if keeping the strong
> "metastable" framing; otherwise Point 6 (soften) already neutralizes the objection at
> zero cost. Listed for completeness with its cost.

### 8a. SUPPLEMENT — new subsection under `\S sec:s_results`:
```latex
\subsection{Direct metastability signatures}
\label{sec:s_metastable}
[near-boundary base-stock histogram showing bimodality (lazy vs coordinated attractor);
first-passage / escape-time distribution across w; short note on whether a hysteresis
sweep (w up vs w down) shows a gap]
```
- **Work:** re-evaluate existing trained near-boundary policies for base-stock histograms
  (no retrain); escape-time needs per-update logging → one small supply-chain qsub sweep.
- **Budget (supplement):** ~0.5 page + 1 figure. Free on page; **~1--2 days compute/analysis.**

---

## Page-budget summary

| Point | Main paper | Supplement | New compute |
|---|---|---|---|
| 1 controls | +0.10p (offset → ~0) | +0.6p | none (done) |
| 2 estimator | +0.05p (offset → ~0) | +0.4p | none (done) |
| 3 hyperparams | 0 | +0.35p | none |
| 4 wrappers | 0 | +0.3p | none |
| 5 code | 0 | +0.05p | bundle (no compute) |
| 6 language | ~0 (retitle+2 sent.) | 0 | none |
| 7 citations | +0.07p (offset) | 0 | none |
| 8 metastability | 0 | +0.5p | 1–2 days |
| **Total** | **≈ net 0** (with the noted offsets) | **≈ +2.7 pages** (8 → ~10.7) | 0 (pts 1–7) |

**Main paper stays within the 7-page body limit** provided the three noted offsets are
made (trim the two affine-law caveat sentences, the redundant half of "Two effect sizes,"
and the standalone "sharp" defense sentence — all already-hedged prose the reviewer did
not ask to keep). **Supplement grows ~2.7 pages, which is unconstrained.**

## Required offsets in the main paper (to hold the 7-page cap)
1. `sec:mechanism`: delete the two "Two caveats bound the claim" sentences (~4 lines) —
   the affine law is already labeled a summary; the caveats duplicate the ledger.
2. `sec:results` "Two effect sizes": compress the $\lambda{=}3$/$\lambda{=}7$ restatement
   (~3 lines) — the numbers are in Table~\ref{tab:decomposition}.
3. `sec:results`: fold the standalone "sharp rather than gradual" defense into the boundary
   paragraph (~2 lines) — Point 2's estimator sentence subsumes it.
Total reclaimed ≈ 9 lines ≈ 0.18 page, covering points 1a+2a+7a insertions.

## What is NOT changed (rebut in the response letter, not the paper)
Novelty-is-modest (§5); the $-127.1$ cutoff (already explained, line 227); the three
impossible robustness variants (already disclosed, line 697); "0.199 not sharp" (width is
reported); "out-of-sample too strong" (already hedged, line 476); not-fully-crossed design
(already stated, line 524). These need a rebuttal letter, out of scope for this document.
```
