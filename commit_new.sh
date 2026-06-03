#!/bin/bash

# Handoff docs
git add docs/handoff/handoff_v1.md
git commit -m "Add project handoff document v1
Provides a comprehensive overview of the prospect theory supply chain project state, including phase 4 replication updates and guidelines for future researchers or reviewers."

# Final Archive
git add docs/final_archive/final_statistics.txt
git commit -m "Add final phase statistics
Contains summarized statistical test results and metrics from the final replication phases to formally conclude the experimental evaluation."

git add docs/final_archive/master_seed_dataset.csv
git commit -m "Add master seed dataset for final archive
Aggregates all training seeds, trajectories, and parameters into a single master dataset for long-term archiving and reproducibility."

# Phase 4 Figures & Data
for file in docs/phase4_figures/*.png docs/phase4_figures/*.txt docs/phase4_figures/*.csv; do
  if [ -f "$file" ]; then
    git add "$file"
    git commit -m "Add Phase 4 asset: $(basename "$file")
Includes interaction data, replication confidence intervals, and regression results for the Phase 4 prospect theory behavioral experiments."
  fi
done

# Experiments
git add experiments/phase4_5_interaction.py
git commit -m "Add phase 4.5 interaction script
Analyzes the interaction effects between different prospect theory parameters (like loss aversion and probability weighting) on agent ordering behaviors."

git add experiments/phase4_regression_analysis.py
git commit -m "Add phase 4 regression analysis script
Performs formal statistical regression to quantify the explanatory power of RL behavioral divergence against established operations research benchmarks."

git add experiments/phase5_final_archiving.py
git commit -m "Add phase 5 final archiving script
Automates the collection, zipping, and organization of all necessary artifacts, models, and logs into a final reproducible archive package."

# Other files
git add commit_all.sh
git commit -m "Add batch commit script
Helper script to automate individual commits with comprehensive messages for each project file, ensuring clean history."

# Push
git push
