#!/bin/bash

# Docs
git add "docs/Literature Review for Novelty Kill.md"
git commit -m "Add literature review regarding novelty
Provides an in-depth review of existing literature to defend the novelty of the research claims against potential reviewer pushback regarding previous work in behavioral operations."

git add docs/experiment_spec_v1.md
git commit -m "Add experiment specification document v1
Outlines the detailed experimental design, hyperparameter specifications, and expected outcomes for the upcoming behavioral simulation phases."

# Phase 6
git add docs/phase6_figures/behavioral_phase_transition_ci.png
git commit -m "Add phase 6 behavioral transition figure
Visualizes the confidence intervals for the behavioral phase transitions observed during the phase 6 experiments."

git add docs/phase6_figures/phase6_raw_data.csv
git commit -m "Add phase 6 raw experimental data
Contains the raw tabular data and metrics gathered during the execution of the phase 6 behavioral claim experiments."

git add experiments/phase6_behavioral_claim.py
git commit -m "Add phase 6 behavioral claim script
Executes the phase 6 experiment to formally test and validate specific behavioral claims about the RL agents under complex supply chain dynamics."

# Logs
git add logs/exp0_run.log
git commit -m "Add experiment 0 run log
Contains the stdout execution logs and diagnostic output from running the baseline experiment 0."

git add logs/expA_horizon.log
git commit -m "Add experiment A horizon log
Contains the execution logs for experiment A, detailing the performance and stability over extended time horizons."

git add logs/expCB_run.log
git commit -m "Add experiment C/B run log
Contains the execution logs for experiments focusing on specific interventions and location sensitivity."

# Other files
git add commit_new.sh
git commit -m "Add supplemental commit script
Helper script to automate individual commits with comprehensive messages for newly added phase 4 and 5 files."

# Push
git push
