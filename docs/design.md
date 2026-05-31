# Prospect Theory MARL - Design Document

## Objective
Simulate **Behavioral Phase Transitions** in supply chains using Multi-Agent Reinforcement Learning (MARL) where agents exhibit heterogenous Prospect-Theoretic biases.

## Core Hypothesis
A minor fraction of highly loss-averse agents can trigger a macroscopic system collapse (e.g., severe Bullwhip Effect, inventory starvation, or cascading stockouts) in an otherwise stable supply chain network. We seek to identify the critical loss-aversion threshold ($\lambda_{crit}$) that initiates these cascades.

## Environment: OR-Gym Multi-Echelon Inventory
We are using the `OR-Gym` benchmark, specifically `NetInvMgmt-v1`, which simulates a multi-echelon supply chain without backlogs.
- Normally, this environment uses a centralized controller.
- We will implement a **Multi-Agent Wrapper** around it. Each node/echelon in the supply chain will be controlled by an independent RL agent.

## Agent Architecture
- **Learning Algorithm:** Independent Q-Learning (IQL) / DQN for each node.
- **Reward Function Shaping:** Each agent has a Prospect Theory value function applied to its raw profit/loss:
  $$v(x) = \begin{cases} x^\alpha & \text{if } x \ge 0 \\ -\lambda(-x)^\beta & \text{if } x < 0 \end{cases}$$
- **Heterogeneity:** Agents will have randomized or sweeping parameter assignments for $\lambda$ (loss aversion) and $\alpha/\beta$ (risk attitude).

## Experiments & Metrics
We will sweep the fraction of loss-averse agents and track macroscopic stability:
1. **Bullwhip Effect:** Variance of orders vs. Variance of demand at the consumer level.
2. **Systemic Starvation:** Frequency and magnitude of lost sales across the entire chain.
3. **Phase Transition Visualization:** Plotting stability metrics against the distribution of $\Lambda$ to find $\lambda_{crit}$.
