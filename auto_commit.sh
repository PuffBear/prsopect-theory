#!/bin/bash
set -euo pipefail

# ============================================================
#  MODIFIED FILES
# ============================================================

git add NEW_RUNS_RUNBOOK.md
git commit -m "Update runbook with SNR reprobe details

Marks the three reviewer-response experiments (points 2, 3, 4) as done
and adds the design, localizer definitions, and pre-registered verdict
criteria for the new SNR reprobe (WA→A) experiment targeting the
survivor-bias hypothesis."

git add AuthorKit27/AnonymousSubmission2027.tex
git commit -m "Update AAAI anonymous submission with new experiment results

Integrates findings from the continuous-lambda cascade, finer high-lambda
grid, and extra horizon experiments into the main technical body to address
reviewer points 2, 3, and 4."

git add AuthorKit27/AnonymousSubmission2027.aux AuthorKit27/AnonymousSubmission2027.bbl \
  AuthorKit27/AnonymousSubmission2027.blg AuthorKit27/AnonymousSubmission2027.log \
  AuthorKit27/AnonymousSubmission2027.pdf
git commit -m "Update AAAI anonymous submission build artifacts

Refreshed compilation artifacts for the updated anonymous submission."

git add AuthorKit27/SupplementaryMaterial2027.tex
git commit -m "Update AAAI supplementary material with new experiment details

Expands the appendix to include comprehensive details, methodology, and
extended results for the new reviewer-response experiments."

git add AuthorKit27/SupplementaryMaterial2027.aux AuthorKit27/SupplementaryMaterial2027.log \
  AuthorKit27/SupplementaryMaterial2027.pdf
git commit -m "Update AAAI supplementary material build artifacts

Refreshed compilation artifacts for the updated supplementary material."

git add AuthorKit27/references.bib
git commit -m "Update AAAI bibliography

Refreshes the project bibliography with any new citations referenced in
the updated submission or supplementary material."

git add AuthorKit27/Figures/boundary_metastability.pdf AuthorKit27/Figures/boundary_metastability.png
git commit -m "Update boundary metastability figures

Refreshes the boundary metastability figures with data from the extra
horizon experiment."

git add auto_commit.sh
git commit -m "Update auto_commit.sh utility script

Refreshes the batch commit script to include the latest batch of files."

# ============================================================
#  UNTRACKED FILES
# ============================================================

git add AuthorKit27/Figures/cascade_tier_dose.pdf AuthorKit27/Figures/cascade_tier_dose.png
git commit -m "Add cascade tier dose figures (Reviewer Point 3)

Visualizes the position-dependent cascade effects from the continuous
lambda cascade experiment, showing differential sensitivity between
upstream and downstream nodes."

git add AuthorKit27/Figures/metastability_lnT.pdf AuthorKit27/Figures/metastability_lnT.png
git commit -m "Add log-time metastability figures (Reviewer Point 2)

Visualizes the metastable slide of the collapse boundary against the
logarithm of the training horizon, using the denser time-series data."

git add docs/expCascade_continuous_figures/cascade_continuous_raw_data.csv
git commit -m "Add continuous-lambda cascade raw data

Tabular results from the continuous lambda × position cascade sweep,
addressing reviewer concerns about discrete lambda values."

git add docs/expHighLambda_finegrid_figures/highlambda_finegrid_raw_data.csv
git commit -m "Add high-lambda fine w-grid raw data

Tabular results from the fine-grained w sweep (step 0.025) at high
lambda values (5, 6, 7), providing higher resolution for w_crit estimation."

git add docs/expHorizon_extra_figures/horizon_extra_raw_data.csv
git commit -m "Add extra matched-horizon raw data

Tabular results from the intermediate training horizon runs (200k, 400k,
500k), filling the gap in the metastability analysis."

git add experiments/analyze_new_runs.py
git commit -m "Add analysis script for new reviewer-response runs

Processes the raw data from the cascade, fine-grid, and extra horizon
experiments to generate the corresponding figures and summary metrics."

git add experiments/exp_snr_reprobe.py
git commit -m "Add SNR reprobe experiment script (WA→A)

Addresses the survivor bias in the original SNR probe by initializing
the actor's mean output near the cooperative optimum (S=150), allowing
clean measurement of the restoring drift."

git add experiments/analyze_snr_reprobe.py
git commit -m "Add SNR reprobe analysis script

Processes the trajectories from the SNR reprobe experiment using the
restoring-drift crossing and basin-flow localizers to test if they
contain the empirical boundary."

git add run_snr_reprobe.pbs
git commit -m "Add PBS script for SNR reprobe experiment

Array job (0-179) for running the SNR reprobe sweep on the HPC cluster."

echo "=== All commits done. Pushing... ==="
git push
