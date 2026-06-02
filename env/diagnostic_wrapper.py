import numpy as np
from pettingzoo.utils import BaseParallelWrapper

class DiagnosticWrapper(BaseParallelWrapper):
    """
    Hides specified scripted_nodes from the RL algorithm and injects a 
    constant Base-Stock (Pull) policy for them.
    """
    def __init__(self, env, scripted_nodes):
        super().__init__(env)
        self.scripted_nodes = scripted_nodes
        
        # Target total order quantity per node to maintain flow
        self.target_qty = {
            "node_1": 20.0,
            "node_2": 10.0,
            "node_3": 10.0,
            "node_4": 10.0,
            "node_5": 10.0,
            "node_6": 10.0
        }
        
        self.possible_agents = [a for a in self.env.possible_agents if a not in self.scripted_nodes]
        self.agents = self.possible_agents[:]
        
        self.observation_spaces = {a: self.env.observation_space(a) for a in self.possible_agents}
        self.action_spaces = {a: self.env.action_space(a) for a in self.possible_agents}
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        
        for sa in self.scripted_nodes:
            if sa in obs:
                del obs[sa]
            if sa in info:
                del info[sa]
            
        return obs, info
        
    def step(self, actions):
        # Inject scripted actions
        for sa in self.scripted_nodes:
            # Reconstruct the original action shape
            shape = self.env.action_space(sa).shape[0]
            val = self.target_qty[sa] / shape
            actions[sa] = np.ones(shape, dtype=np.float64) * val
            
        obs, rewards, term, trunc, info = self.env.step(actions)
        
        # Filter outputs
        for sa in self.scripted_nodes:
            if sa in obs: del obs[sa]
            if sa in rewards: del rewards[sa]
            if sa in term: del term[sa]
            if sa in trunc: del trunc[sa]
            if sa in info: del info[sa]
            
        if any(term.values()) or any(trunc.values()):
            self.agents = []
            
        return obs, rewards, term, trunc, info

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
