#!/bin/bash

# Agents
git add agents/cpt_wrapper.py
git commit -m "Add Cumulative Prospect Theory wrapper for RL agents
Implements CPT utilities and probability weighting functions to integrate human-like decision biases (loss aversion, reference dependence) into reinforcement learning policies for supply chain models."

# Docs
git add docs/defense.md
git commit -m "Add formal defense strategy for academic reviewer critiques
Details methodological justifications, literature context, and diagnostic focus regarding the observed failures of continuous PPO in the supply chain environments, framing it as a diagnostic contribution rather than a failed solution."

git add docs/litreview.md
git commit -m "Add literature review on Prospect Theory and RL in Supply Chains
Summarizes existing research on behavioral operations, bullwhip effect, and application of reinforcement learning with prospect theory elements in multi-echelon inventory management."

# Docs figures - Phase 1
for file in docs/phase1_figures/*.png; do
  git add "$file"
  git commit -m "Add phase 1 diagnostic figure: $(basename "$file")
Visualizes experimental results and diagnostic metrics (order quantities, lost sales, bullwhip effect) from Phase 1 of the supply chain RL agent evaluation."
done

# Docs figures - Phase 2
for file in docs/phase2_figures/*.png; do
  git add "$file"
  git commit -m "Add phase 2 divergence figure: $(basename "$file")
Visualizes policy divergence, inventory levels, and profit curves during phase 2 experiments, highlighting suboptimal behavioral patterns in the agents."
done

# Docs figures - Phase 2a
for file in docs/phase2a_figures/*.png; do
  git add "$file"
  git commit -m "Add phase 2a replication figure: $(basename "$file")
Visualizes the results of phase 2a replication experiments, comparing alternative algorithm or hyperparameter configurations."
done

# Docs figures - Phase 3
for dir in docs/phase3_*_figures; do
  if [ -d "$dir" ]; then
    for file in "$dir"/*.png; do
      if [ -f "$file" ]; then
        git add "$file"
        git commit -m "Add phase 3 analysis figure: $(basename "$file")
Visualizes advanced agent dynamics including reward landscapes, basin of attraction, action histograms, and lead time causality from phase 3 experiments."
      fi
    done
  fi
done

# Docs assets
for file in docs/assets/*.png; do
  git add "$file"
  git commit -m "Add documentation asset: $(basename "$file")
Includes visual assets such as pt curves, tipping points, and baseline rational behaviors for README and markdown documentation."
done

# Docs results (csv, json, zip)
for file in docs/results/*.csv docs/results/metadata/*.json docs/results/*.zip docs/phase3_9_figures/*.csv; do
  if [ -f "$file" ]; then
    git add "$file"
    git commit -m "Add experimental result data: $(basename "$file")
Stores quantitative output, metadata, and serialized models generated during the multi-phase reinforcement learning supply chain experiments."
  fi
done

# Env wrappers
git add env/base_stock_wrapper.py
git commit -m "Add Base Stock policy environment wrapper
Provides an environment wrapper to enforce or benchmark against standard base stock (order-up-to) inventory policies in the OR-Gym environments."

git add env/diagnostic_wrapper.py
git commit -m "Add diagnostic environment wrapper
Instruments the base supply chain environment to track and log detailed behavioral metrics, gradients, and step-by-step state transitions for deep analysis of agent failure modes."

git add env/intervention_wrapper.py
git commit -m "Add intervention environment wrapper
Allows injecting specific state or reward interventions (e.g., freezing inventory, overriding orders) during episodes to test causal hypotheses about agent learning dynamics."

git add env/marl_or_gym_wrapper.py
git commit -m "Add Multi-Agent RL wrapper for OR-Gym
Adapts single-agent multi-echelon supply chain environments into a proper multi-agent formulation compatible with independent or centralized MARL algorithms."

git add env/offset_base_stock_wrapper.py
git commit -m "Add Offset Base Stock wrapper
Implements an inventory policy wrapper that introduces systematic offsets to base stock levels, used to evaluate agent robustness against biased reference points."

git add env/scaled_base_stock_wrapper.py
git commit -m "Add Scaled Base Stock wrapper
Implements a base stock wrapper that scales order quantities, providing a structural benchmark for evaluating magnitude-based policy errors."

# Experiments
git add experiments/ablation_scaling.py
git commit -m "Add ablation scaling experiment script
Tests the impact of scaling various reward or state components to isolate the drivers of suboptimal agent behavior in supply chain environments."

git add experiments/parameter_sweep.py
git commit -m "Add parameter sweep utility script
Automates hyperparameter tuning and exploration across different learning rates, discount factors, and penalty terms for the PPO algorithms."

git add experiments/phase0_validation.py
git commit -m "Add phase 0 validation script
Establishes a baseline by validating the pure un-modified OR-Gym supply chain environment mechanics and rational optimal solutions."

git add experiments/phase1_5_5_cure.py
git commit -m "Add phase 1.5.5 'cure' experiment script
Evaluates specific interventions designed to 'cure' the observed pathologies (like extreme ordering) in standard RL supply chain agents."

git add experiments/phase1_5_diagnosis.py
git commit -m "Add phase 1.5 diagnosis script
Analyzes preliminary failure patterns identified in Phase 1, logging detailed trajectories to isolate policy collapse triggers."

git add experiments/phase1_double_sweep.py
git commit -m "Add phase 1 double sweep experiment script
Conducts a two-dimensional hyperparameter sweep to map out the interaction between key algorithm parameters and supply chain stability."

git add experiments/phase1_intervention.py
git commit -m "Add phase 1 intervention script
Applies structural interventions (e.g., state masking, reward clipping) to the supply chain environment to test hypotheses regarding policy instability."

git add experiments/phase1_runner.py
git commit -m "Add phase 1 main runner script
Orchestrates the initial batch of diagnostic training runs for PPO agents on the supply chain environment, logging base performance and failure modes."

git add experiments/phase2_divergence.py
git commit -m "Add phase 2 divergence analysis script
Focuses on tracking and plotting the exact episodes where agent policies systematically diverge from rational behavior towards pathological extremes."

git add experiments/phase2a_replication.py
git commit -m "Add phase 2a replication script
Replicates phase 2 findings with alternative random seeds and slight environmental variations to ensure the observed divergence is a robust phenomenon."

git add experiments/phase3_5_challenges.py
git commit -m "Add phase 3.5 challenges experiment
Introduces specific adversarial scenarios and edge cases in demand to test the limits of agent policies that survived earlier phases."

git add experiments/phase3_6_audit.py
git commit -m "Add phase 3.6 action audit script
Performs a deep-dive audit of the action distributions produced by the trained agents, identifying structural biases such as uniform maximum ordering."

git add experiments/phase3_6_replication.py
git commit -m "Add phase 3.6 replication script
Ensures reproducibility of the action distribution anomalies found in the phase 3.6 audit across multiple training runs."

git add experiments/phase3_7_verification.py
git commit -m "Add phase 3.7 verification script
Verifies the theoretical limits of performance achievable by the network architecture compared to the empirically observed agent policies."

git add experiments/phase3_8_incentives.py
git commit -m "Add phase 3.8 incentives analysis script
Explores how changes to the localized reward structures (e.g., backlog sharing, penalty shaping) influence the emergence of cooperative vs competitive ordering."

git add experiments/phase3_95_reward_landscape.py
git commit -m "Add phase 3.95 reward landscape analysis
Maps the objective function landscape around the converged policy parameters to determine if the agents are falling into sharp local optima."

git add experiments/phase3_96_basin_attraction.py
git commit -m "Add phase 3.96 basin of attraction script
Analyzes the stability of different policy regions, measuring how easily agents can escape suboptimal regions (like zero-ordering or max-ordering)."

git add experiments/phase3_9_leadtime.py
git commit -m "Add phase 3.9 leadtime causality script
Investigates the causal impact of deterministic vs stochastic lead times on the agent's ability to maintain stable inventory levels."

git add experiments/phase3_validation.py
git commit -m "Add phase 3 validation script
Synthesizes and validates the core diagnostic findings across phase 3 experiments regarding reward shaping, exploration, and policy collapse."

git add experiments/phase4_pt_replication.py
git commit -m "Add phase 4 prospect theory replication script
Evaluates whether injecting Prospect Theory elements (loss aversion, reference dependence) directly replicates or exacerbates the observed human-like bullwhip behaviors."

git add experiments/phase_transition.py
git commit -m "Add phase transition analysis script
Identifies the critical hyperparameter or environmental thresholds (tipping points) where agent behavior rapidly shifts from rational to pathological."

git add experiments/sb3_phase_transition.py
git commit -m "Add Stable Baselines 3 phase transition script
Adapts the phase transition analysis specifically for algorithms implemented in the Stable Baselines 3 library to rule out implementation bugs."

git add experiments/visualize_policy.py
git commit -m "Add policy visualization script
Generates graphical representations of the learned policy mappings from state observations to continuous action outputs."

git add experiments/visualize_supply_chain.py
git commit -m "Add supply chain visualization script
Creates visual dashboards or plots tracking inventory flow, backlog buildup, and order quantities across multiple echelons over time."

# Logs
for file in logs/agent_traces/*.csv; do
  if [ -f "$file" ]; then
    git add "$file"
    git commit -m "Add agent trace log: $(basename "$file")
Contains raw step-by-step diagnostic trace data for specific agent training runs."
  fi
done

# or-gym
git add or-gym/
git commit -m "Add or-gym submodule or directory
Includes the base implementation of Operations Research environments (like the multi-echelon supply chain) used as the foundation for the RL experiments."

# Root python files
git add or_gym_network.py
git commit -m "Add custom OR-Gym network architecture
Defines the neural network topologies used by RL agents interacting with OR-Gym environments, potentially tailored for structural state spaces."

git add test_env.py
git commit -m "Add environment unit test script
Contains basic validation tests to ensure the supply chain environment initializes, steps, and resets correctly according to Markov property expectations."

git add test_lead.py
git commit -m "Add lead time test script
Validates the correct implementation of order delays, intransit inventory accounting, and lead time stochasticity in the supply chain models."

git add test_rllib.py
git commit -m "Add RLlib integration test
Ensures that the customized OR-Gym environments are fully compatible with Ray RLlib's API for distributed reinforcement learning."

git add test_short.py
git commit -m "Add quick validation test script
Provides a short, fast-running test suite for rapid sanity checking during active development of environments and wrappers."

git add test_single.py
git commit -m "Add single-agent test script
Tests the simplified single-echelon version of the supply chain environment to isolate node-level decision dynamics."

git add test_wrapper.py
git commit -m "Add environment wrapper unit tests
Validates that state transformations, reward shaping, and action modifications implemented by custom gym wrappers function as intended."

# Utils
git add utils/learning_diagnostics.py
git commit -m "Add learning diagnostics utility
Provides functions to calculate and track gradient norms, value function losses, and entropy over time to diagnose RL training stability."

git add utils/metrics.py
git commit -m "Add core metrics utility
Defines calculation logic for key supply chain performance indicators such as the Bullwhip Effect, service level, and average inventory costs."

git add utils/phase1_metrics.py
git commit -m "Add phase 1 specific metrics utility
Contains customized metric definitions explicitly tailored for the phase 1 diagnostic experiments and pathological behavior isolation."

git add utils/plot_ablation.py
git commit -m "Add ablation plotting utility
Generates visualizations comparing the performance impact of selectively removing or scaling various components of the RL algorithm or environment."

git add utils/plot_cure.py
git commit -m "Add cure intervention plotting utility
Creates graphs demonstrating the effectiveness of specific interventions ('cures') on mitigating suboptimal agent ordering behaviors."

git add utils/plot_diagnosis.py
git commit -m "Add diagnosis plotting utility
Produces detailed visual diagnostics of policy divergence, highlighting where and how agent behavior deviates from theoretical optima."

git add utils/plot_intervention.py
git commit -m "Add intervention plotting utility
Visualizes the before-and-after effects of structural environmental interventions on multi-echelon inventory levels and order stability."

git add utils/plot_phase0.py
git commit -m "Add phase 0 plotting utility
Generates baseline performance graphs for the un-modified environments to serve as a rational benchmark."

git add utils/plot_phase1.py
git commit -m "Add phase 1 plotting utility
Visualizes the initial findings of phase 1, emphasizing the onset of extreme ordering and policy collapse in continuous PPO agents."

git add utils/plot_phase2.py
git commit -m "Add phase 2 plotting utility
Graphs the divergence patterns of phase 2, analyzing how non-stationary demand or specific network topologies trigger pathological behavior."

git add utils/plot_phase2a.py
git commit -m "Add phase 2a plotting utility
Provides visual validation of the phase 2a replication studies to ensure robustness of the observed failure modes."

git add utils/plot_phase3_5_success.py
git commit -m "Add phase 3.5 success plotting utility
Visualizes the rare scenarios or specific hyperparameter configurations where agents successfully managed the supply chain without collapse."

git add utils/plot_tipping_point.py
git commit -m "Add tipping point plotting utility
Graphs the behavioral phase transitions, visualizing the sharp thresholds where agent policies shift from stable to unstable."

# Push
git push
