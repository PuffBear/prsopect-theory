import numpy as np
from pettingzoo.utils import BaseParallelWrapper

class InterventionWrapper(BaseParallelWrapper):
    """
    Hides node_1 from the RL algorithm and injects a scripted base-stock policy for it.
    This tests the "MARL Relative Overgeneralization" hypothesis.
    """
    def __init__(self, env):
        super().__init__(env)
        self.scripted_agent = "node_1"
        
        self.possible_agents = [a for a in self.env.possible_agents if a != self.scripted_agent]
        self.agents = self.possible_agents[:]
        
        self.observation_spaces = {a: self.env.observation_space(a) for a in self.possible_agents}
        self.action_spaces = {a: self.env.action_space(a) for a in self.possible_agents}
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        
        if self.scripted_agent in obs:
            del obs[self.scripted_agent]
        if self.scripted_agent in info:
            del info[self.scripted_agent]
            
        return obs, info
        
    def step(self, actions):
        # Inject scripted action for node_1
        # node_1 orders from node_2 and node_3. 
        # Mean market demand is 20, so ordering 10 from each perfectly balances flow.
        actions[self.scripted_agent] = np.array([10.0, 10.0], dtype=np.float64)
        
        obs, rewards, term, trunc, info = self.env.step(actions)
        
        if self.scripted_agent in obs:
            del obs[self.scripted_agent]
        if self.scripted_agent in rewards:
            del rewards[self.scripted_agent]
        if self.scripted_agent in term:
            del term[self.scripted_agent]
        if self.scripted_agent in trunc:
            del trunc[self.scripted_agent]
        if self.scripted_agent in info:
            del info[self.scripted_agent]
            
        if any(term.values()) or any(trunc.values()):
            self.agents = []
            
        return obs, rewards, term, trunc, info

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
