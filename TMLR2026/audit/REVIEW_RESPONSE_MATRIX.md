# TMLR revision response matrix

Updated 2026-10-15. This matrix separates completed manuscript changes from evidence that still requires execution.

| Review concern | Evidence/action | Manuscript disposition | Status |
|---|---|---|---|
| Is the crossing only an optimizer stopping contour? | Added an independent 150-policy cohort evaluated every 150k transitions through 3.6M. Reported absolute crossings, paired multiplier-minus-curvature contrasts, within-policy recovery/relapse, and 10-versus-50-episode sensitivity. | Claims persistence only over tested budgets; explicitly disclaims an infinite-horizon asymptote and absorbing collapse. | Complete |
| Does loss aversion differ from scalar scaling? | Added a 240-policy three-arm comparison at a common empirical-mean reference: curvature baseline B, losses-only multiplier A, and symmetric multiplier S. A-S is -0.062 [-0.157, 0.028], while S-B exceeds A-B in point estimate. | Abstract, results, discussion, and conclusion now state that gain-loss asymmetry is not causally identified. | Complete; result is negative |
| Could critic-dominated global clipping explain the scaling result? | Audited every optimizer minibatch. All were clipped; critic pre-clip norms greatly exceed actor norms. Frozen a broader optimizer/normalization factorial with four reward arms, including an RMS-matched symmetric arm, and pre/post actor-gradient-share logging. | Presented as a concrete but untested pathway and future experiment, never as a causal diagnosis. | Factorial v2 design and source frozen; execution blocked because the compute account is unavailable and the local analysis environment has no PyTorch |
| Does the horizon result approach a nonzero asymptote? | Extended the same policies to 3.6M and applied a declared practical-stability criterion. No condition meets it. | No asymptote claim. | Complete within observed horizon; infinite-horizon question remains open |
| Are observed cells visible and do they bracket the midpoint? | Main figures show observed proportions and Wilson intervals. Appendix tables give all primary cells and confirmation cells; raw exports retain every bootstrap result. | Bracketing is a separate diagnostic from fitted localization. | Complete |
| Is estimator agreement overstated? | Primary supply-PPO maximum localized spread is 0.018. The text also discloses the 0.158 spread found in an excluded exploratory transfer audit. | Removed the universal agreement claim. | Complete |
| Are transfer claims robust to wrapper/label calibration? | Removed MPE, LBF, supply-MAPPO, action-discretization, and heterogeneous-node results from the TMLR claims. | Quantitative core is explicitly single-substrate. | Complete |
| Is LBF's loss branch valid? | LBF is no longer used as confirmatory evidence. | No LBF mechanism or transfer claim remains. | Complete |
| Are MPE configurations traceable? | Removed all MPE estimates and provenance commentary from the paper. | No untraceable MPE result remains. | Complete |
| Is the contribution confused with a new logistic midpoint? | Added dose-response and psychometric-threshold lineage and narrowed novelty to MARL experimental design, behavioral labels, seed-block uncertainty, and paired reward-design comparisons. | Contribution statement rewritten. | Complete |
| Are IRAT and CLOT covered? | Added both and distinguished boundary measurement from reward balancing. | Related work revised. | Complete |
| Does the paper overclaim monotonicity? | Retains selected contrasts and full observed sweep counts without asserting a universal monotone relation. | Monotonicity claim removed. | Complete |
| Are the absolute long-horizon crossings available? | Added 1.8M and 3.6M crossings and intervals to the main text, including the coarse-grid zero-width explanation. | Main result no longer depends on appendix-only values. | Complete |
| Can transform shifts be expressed in budget units? | Refit the four cross-sectional linear horizons on each of 2,000 seed-block replicates and inverted the local log-budget curve. The table reports 685k [535,942] for curvature and 98k [67,128] for lambda 7, marking the latter as an extrapolation below the observed range. | Presented only as a descriptive translation, never as causal equivalence. | Complete |
| Are the experimental ingredients visible in one place? | Added a main-text setup table covering environment, rewards, actor/critic sharing, label, budgets, transform arms, and uncertainty. | Removes ambiguity about the six-node/five-trainable-node setup and joint clipping. | Complete |
| Does every abstract number come from checked exports? | Added a deterministic exporter for the primary crossing, multiplier shift, training budget, crossing probability, and maximum confirmation horizon. | The abstract imports generated macros rather than hand-entered values. | Complete |
| Is reproducibility overstated? | Built an anonymous artifact with raw new-cohort records, audited legacy per-run rows, exact new-cohort executed-source snapshots and hashes, analysis scripts, generated evidence, and a one-command rebuild. Clean extraction passes the 390-run audit and manuscript build. | Explicitly states that the older primary sweep lacks a preserved execution environment. | Complete |
| Is the prose defensive or internally inconsistent? | Rewrote the abstract and contributions around positive measurement findings plus one direct limitation; removed “historical/provisional” result annotations and stale transfer sections. | Main paper is eight pages before references. | Complete |
| Companion CPT citation | User instructed that it is not required. | No companion-paper citation added. | Intentionally omitted |

## Submission blockers

1. The frozen optimizer/normalization factorial cannot run until the compute account is restored (or a suitable PyTorch training environment is supplied). The current paper remains scientifically honest without it, but completing it would directly test the strongest remaining causal alternative.
2. TMLR prohibits simultaneous submission to another archival peer-reviewed venue. Do not upload this manuscript while the AAMAS submission remains under review; wait for its final disposition or withdraw it first.
3. Before upload, replace the placeholder OpenReview identifier/date fields if the submission system requires them and run the final anonymity gate once more.

## Checklist experiments intentionally not represented as completed evidence

- LBF curvature, the separate-actor lambda-1 baseline, and per-node reward-RMS accounting have not been rerun in the verified pipeline. They remain outside the TMLR claims rather than being filled with unverified historical estimates.
- The missing implementation-choice ablation for sticky collapse remains open. The paper reports the observed persistence and recovery data and explicitly says the causal implementation choice is unidentified.
- The audited anonymous archive and reconstruction sources are included on the dedicated GitHub revision branch. Venue timing and anonymity still govern when that branch or archive should be cited from a submission.
