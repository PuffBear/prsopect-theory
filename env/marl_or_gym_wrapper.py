import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
import or_gym

class MultiAgentNetInvMgmt(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "marl_net_inv_mgmt_v1"}

    def __init__(self, *args, **kwargs):
        self.render_mode = None
        self.env = or_gym.make("NetworkManagement-v1", *args, **kwargs)
        self.unwrapped_env = self.env.unwrapped
        
        # main_nodes are the nodes that place orders and hold inventory
        # e.g., distrib and factory nodes
        self.agents = [f"node_{j}" for j in self.unwrapped_env.main_nodes]
        self.possible_agents = self.agents[:]
        
        # Maps agent name to node ID
        self.agent_name_to_id = {f"node_{j}": j for j in self.unwrapped_env.main_nodes}
        
        # Determine which indices in the global action array belong to which agent
        self.agent_action_indices = {}
        for agent in self.agents:
            node_id = self.agent_name_to_id[agent]
            # reorder_links are (supplier, purchaser)
            indices = [i for i, link in enumerate(self.unwrapped_env.reorder_links) if link[1] == node_id]
            self.agent_action_indices[agent] = indices

        # Setup action and observation spaces
        self.action_spaces = {}
        self.observation_spaces = {}
        
        obs_dim = self.env.observation_space.shape[0]
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)
        
        global_action_high = self.env.action_space.high
        
        for agent in self.agents:
            # We give each agent the FULL global observation for simplicity
            self.observation_spaces[agent] = obs_space
            
            # Action space is just the order quantities to its specific suppliers
            agent_indices = self.agent_action_indices[agent]
            highs = global_action_high[agent_indices]
            self.action_spaces[agent] = gym.spaces.Box(low=0.0, high=highs, dtype=np.float64)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        self.agents = self.possible_agents[:]
        
        observations = {agent: obs for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos

    def step(self, actions):
        # Construct global action array
        global_action = np.zeros(self.env.action_space.shape[0], dtype=np.float64)
        
        for agent, action in actions.items():
            indices = self.agent_action_indices[agent]
            global_action[indices] = action

        obs, global_reward, terminated, truncated, info = self.env.step(global_action)
        
        # Extract local rewards from the unwrapped environment's profit dataframe
        # The env updates self.period during step, so the profit for the current step
        # is at self.period - 1
        t = self.unwrapped_env.period - 1
        
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent in self.agents:
            node_id = self.agent_name_to_id[agent]
            observations[agent] = obs
            
            # Local reward is the profit at this specific node
            rewards[agent] = self.unwrapped_env.P.loc[t, node_id]
            
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {}

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self, mode='human'):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
