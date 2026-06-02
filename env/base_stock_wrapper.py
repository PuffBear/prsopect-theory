import numpy as np
import gymnasium as gym
from pettingzoo.utils import BaseParallelWrapper

class BaseStockWrapper(BaseParallelWrapper):
    """
    Translates the PPO continuous action from a raw 'Order Quantity' to a 'Target Base-Stock Level' (S).
    Order Quantity = max(0, S - Inventory_Position)
    """
    def __init__(self, env):
        super().__init__(env)
        
        # The base environment defines action space as Box(low=0, high=capacity)
        # We change it to represent target Base-Stock levels up to a high limit (e.g. 500)
        self.action_spaces = {
            agent: gym.spaces.Box(low=0.0, high=500.0, shape=self.env.action_space(agent).shape, dtype=np.float64)
            for agent in self.possible_agents
        }
        
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        # We need to compute the Order Quantity from the S action.
        # base_env is the underlying or_gym NetInvMgmtMasterEnv
        if hasattr(self.env, 'unwrapped_env'):
            base_env = self.env.unwrapped_env
        else:
            base_env = self.env.unwrapped.unwrapped_env
            
        t = base_env.period
        
        order_actions = {}
        for agent, S_action in actions.items():
            # S_action is a numpy array of target base-stock levels (one per supplier)
            node_id = self.env.agent_name_to_id[agent]
            
            # 1. On-Hand Inventory
            on_hand = base_env.X.loc[t, node_id]
            
            # 2. Pipeline Inventory (Incoming orders not yet arrived)
            pipeline = 0
            for link in base_env.reorder_links:
                if link[1] == node_id:
                    pipeline += base_env.Y.loc[t, link]
                    
            # 3. Backlog (Unfulfilled retail demand, only applies to retailers)
            backlog = 0
            if node_id in base_env.retail and t > 0:
                for link in base_env.retail_links:
                    if link[0] == node_id:
                        backlog += base_env.U.loc[t-1, link]
                        
            # Total Inventory Position
            IP = on_hand + pipeline - backlog
            
            # Calculate Replenishment Order
            order_qty = np.maximum(0.0, S_action - IP)
            order_actions[agent] = order_qty
            
        return self.env.step(order_actions)
