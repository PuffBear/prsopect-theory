import os
import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
from agents.cpt_wrapper import CPTRewardWrapper

def env_creator(config):
    env = MultiAgentNetInvMgmt()
    env = CPTRewardWrapper(env, config.get("agent_params", {}))
    return ParallelPettingZooEnv(env)

ray.tune.registry.register_env("cpt_supply_chain", env_creator)

def run_experiment(lambda_value, fraction_loss_averse, num_iterations=1):
    ray.init(ignore_reinit_error=True)
    
    # Define agent parameters
    env_temp = MultiAgentNetInvMgmt()
    agents = env_temp.agents
    
    # Assign high lambda to a fraction of agents, lambda=1 to the rest
    num_loss_averse = int(len(agents) * fraction_loss_averse)
    agent_params = {}
    for i, agent in enumerate(agents):
        if i < num_loss_averse:
            agent_params[agent] = {"lambda": lambda_value, "alpha": 1.0, "beta": 1.0}
        else:
            agent_params[agent] = {"lambda": 1.0, "alpha": 1.0, "beta": 1.0}
            
    # Setup policies (one for each agent)
    policies = {
        agent: (None, env_temp.observation_space(agent), env_temp.action_space(agent), {})
        for agent in agents
    }
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return agent_id
        
    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment("cpt_supply_chain", env_config={"agent_params": agent_params})
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=1000,
            minibatch_size=128,
            num_epochs=10
        )
        .env_runners(num_env_runners=0)
    )
    
    algo = config.build()
    
    for i in range(num_iterations):
        result = algo.train()
        reward = result.get("env_runners", {}).get("episode_reward_mean", result.get("episode_reward_mean", "N/A"))
        print(f"Iteration {i}: mean reward = {reward}")
        
    # Evaluate policy to get metrics
    from utils.metrics import calculate_bullwhip_effect, calculate_systemic_lost_sales
    
    test_env = env_creator({"agent_params": agent_params})
    obs, info = test_env.reset()
    
    while True:
        actions = {}
        for agent in test_env.agents:
            if agent in obs:
                action, _, _ = algo.get_policy(agent).compute_single_action(obs[agent], explore=False)
                actions[agent] = action
            
        obs, rewards, term, trunc, info = test_env.step(actions)
        if len(test_env.agents) == 0:
            break
            
    # Unpack the wrappers: test_env is ParallelPettingZooEnv -> env is CPTRewardWrapper -> env is MultiAgentNetInvMgmt
    base_env = test_env.env.env
    bw = calculate_bullwhip_effect(base_env)
    ls = calculate_systemic_lost_sales(base_env)
    print(f"*** FINAL EVALUATION (Lambda={lambda_value}, Fraction High-Lambda={fraction_loss_averse:.0%}) ***")
    print(f"Bullwhip Effect: {bw:.2f}")
    print(f"Systemic Lost Sales: {ls:.2%}")
    print("*" * 50)
    
    return algo

if __name__ == "__main__":
    print("Running Baseline (Homogeneous Rational Agents, lambda=1.0)...")
    run_experiment(lambda_value=1.0, fraction_loss_averse=0.0, num_iterations=2)
    
    print("\nRunning High Loss Aversion Cascade Simulation (lambda=5.0 for 30% of agents)...")
    run_experiment(lambda_value=5.0, fraction_loss_averse=0.3, num_iterations=2)
    
    ray.shutdown()
