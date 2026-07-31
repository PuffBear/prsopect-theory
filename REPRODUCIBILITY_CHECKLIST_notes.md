# Reproducibility checklist — how it was filled, and what still gates it

I completed `AuthorKit27/ReproducibilityChecklist.tex` (was 100% blank — every answer was "Type your response here"). Answers are grounded in the paper, supplement, and your actual code (`experiments/`, `agents/mappo_policy.py`, `env_manifest.txt`, the PBS scripts). Below: every answer, why, and the actions that must happen for the "yes"es to be truthful.

## The one hard dependency (do this or several answers become false)

Six answers assume the **anonymized code + data archive is submitted** (2.7, 4.2, 4.3 say code is "included in a code appendix"). AAAI-27's new rule: *materials that "will be released after acceptance" are NOT counted; if not included at submission, reviewers assume they won't exist.*

➡️ **You must package `experiments/` + `agents/` + the wrappers + the `docs/*_raw_data.csv` results + analysis scripts into an anonymized ZIP and upload it as the "Code and Data" supplement by Aug 1 (IST).** If you do not, change 2.7 / 4.2 / 4.3 from **yes** to **no**. This is the single most important action.

## Answers and rationale

**1. General paper structure**
- 1.1 conceptual outline / pseudocode — **partial.** The four-step protocol is given as a clear conceptual outline in the Method, but there is no boxed pseudocode. (If you add a small algorithm box, upgrade to yes.)
- 1.2 opinions vs facts delineated — **yes.** The claim ledger, "consistent with," and scoped-claim language do exactly this.
- 1.3 pedagogical references — **yes** (prospect theory, Kramers, PPO/MAPPO, CTDE all cited).

**2. Theoretical contributions** — makes them (**yes**), via Proposition 1.
- 2.1 assumptions stated formally — **yes** (Prop 1 (i)–(iii)).
- 2.2 novel claims formal — **yes.**
- 2.3 proofs included — **yes** (main text + supplement).
- 2.4 proof sketches/intuition — **yes.**
- 2.5 citations to theoretical tools — **yes** (Kramers, mean-field refs, IFT).
- 2.6 theoretical claims demonstrated empirically — **partial**, and this is deliberately honest: the *sign* result (loss aversion raises $\wcrit$) is shown across four settings, but the $\partial_\lambda\mathcal S<0$ premise is not independently localized (two pre-registered probes were negative). Answering "yes" here would contradict the honesty edits we made to the paper — keep it partial.
- 2.7 code to disprove claims included — **yes** (the negative SNR probes are in `experiments/` — *conditional on submitting the archive*).

**3. Dataset usage** — relies on datasets/environments (**yes**; the sim environments + generated run data).
- 3.1 motivation for choice — **yes.**
- 3.2 novel datasets in a data appendix — **NA.** You introduce no benchmark dataset; environments are from prior work. (The generated run-outcome CSVs ship in the code/data archive, but that's results, not a novel dataset.)
- 3.3 novel datasets made public — **NA** (same reason).
- 3.4 literature datasets cited — **yes** (or-gym/Hubbs 2020, MPE/Lowe 2017, LBF/Christianos 2020).
- 3.5 literature datasets public — **yes** (all three are open-source).
- 3.6 non-public datasets described — **NA** (all public).

**4. Computational experiments** — **yes.**
- 4.1 hyperparameter ranges tried + selection criterion — **partial.** The reward-design sweeps ($w$-grid, $\lambda\in\{1..7\}$, $\alpha=\beta\in\{1,0.88\}$, horizons) and the criterion for $\alpha=\beta=0.88$ (canonical PT value) are stated; you did not search PPO optimizer hyperparameters (you used SB3 defaults), so there is no "range tried" for those. Partial is the honest call.
- 4.2 pre-processing code included — **yes** (env/base-stock/herding wrappers — *conditional on archive*).
- 4.3 all experiment/analysis code included — **yes** (*conditional on archive*).
- 4.4 code public upon publication — **yes.**
- 4.5 code commented w/ paper references — **partial** (typical; comments exist but aren't systematically cross-referenced to paper sections).
- 4.6 seed method described — **yes** (20 fixed seeds per cell; seeds set in the `PPO(..., seed=seed)` calls).
- 4.7 computing infrastructure — **partial.** I just added a "Computing infrastructure" paragraph to the supplement (CPU-only, 1 core/run, Linux PBS cluster, and exact library versions: SB3 2.8.0, PettingZoo 1.26.1, SuperSuit 3.11.0, Gymnasium 1.2.2, mpe2 1.1.0, PyTorch 2.12.0, NumPy 2.4.6, pandas 3.0.3). Still "partial" because the exact **CPU model and RAM** aren't stated — if you know them (cluster node spec), add them and upgrade to **yes**.
- 4.8 evaluation metrics described + motivated — **yes** (collapse labels, $\wcrit$, transition width — all defined and motivated).
- 4.9 number of runs per result — **yes** (20 seeds/cell; full inventory, Supplement Table 1 = 2,364 runs).
- 4.10 beyond single-dim summaries — **yes** (bootstrap 95% CIs everywhere).
- 4.11 significance via appropriate tests — **yes** (Firth logistic-slope test, bootstrap CIs, reported $p$-values).
- 4.12 all final hyperparameters listed — **yes** (now fully specified in the supplement: MlpPolicy 2×64, n_steps=128, batch_size=256, remaining PPO settings at SB3 2.8.0 defaults; MAPPO = same + centralized critic).

## What I changed in the supplement to support these answers
Added to `SupplementaryMaterial2027_2.tex` (Algorithm/evaluation section):
1. Full hyperparameter statement — "SB3 default MlpPolicy, two hidden layers of 64 units, all remaining PPO hyperparameters at Stable-Baselines3 defaults; MAPPO = same with a centralized joint-state critic" (supports 4.12, 4.1).
2. A new **Computing infrastructure** paragraph — CPU-only, 1 core/run, Linux PBS cluster, 3–12 h walltimes, and the exact library versions (supports 4.7).

## Remaining actions before you submit the checklist (Aug 1 IST)
1. **Build and upload the anonymized code+data archive** — this is what makes 2.7/4.2/4.3 truthful. Strip your name/email/paths and any GitHub URL first (`grep -ri "agriya\|agriyayadav\|/Users/\|github.com" experiments agents env`).
2. (Optional, to upgrade 4.7 → yes) add the cluster node's **CPU model + RAM** to the Computing-infrastructure paragraph.
3. (Optional, to upgrade 1.1 → yes) add a small pseudocode box for the boundary-estimation protocol.
4. Re-read 2.6 and keep it **partial** — it is the honest answer and matches the paper's own hedging; do not "upgrade" it.
