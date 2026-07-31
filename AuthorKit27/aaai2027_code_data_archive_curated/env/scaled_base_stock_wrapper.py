import numpy as np
import gymnasium as gym
from pettingzoo.utils import BaseParallelWrapper

class ScaledBaseStockWrapper(BaseParallelWrapper):
    """
    Translates the PPO continuous action from the normalized range [-1, 1] 
    to a 'Target Base-Stock Level' (S) in the range [low_bound, high_bound].
    Order Quantity = max(0, S - Inventory_Position)
    """
    def __init__(self, env, low_bound=200.0, high_bound=700.0, log_file=None):
        super().__init__(env)
        self.low_bound = float(low_bound)
        self.high_bound = float(high_bound)
        self.log_file = log_file
        self.step_counter = 0
        
        # PPO requires action space to be strictly [-1, 1] for Gaussian output mapping
        self.action_spaces = {
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=self.env.action_space(agent).shape, dtype=np.float64)
            for agent in self.possible_agents
        }
        
        if self.log_file and getattr(self, '_cleared', False) == False:
            import os
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    f.write("step,agent,S,IP,Q\n")
            self._cleared = True
        
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        base_env = self.unwrapped.unwrapped_env
        t = base_env.period
        
        order_actions = {}
        for agent, action in actions.items():
            # 1. Unscale action from [-1, 1] to [low_bound, high_bound]
            raw_action = np.clip(action[0], -1.0, 1.0)
            S = self.low_bound + 0.5 * (raw_action + 1.0) * (self.high_bound - self.low_bound)
            
            node_id = self.env.agent_name_to_id[agent]
            
            # 2. On-Hand Inventory
            on_hand = base_env.X.loc[t, node_id]
            
            # 3. Pipeline Inventory (Incoming orders not yet arrived)
            pipeline = 0
            for link in base_env.reorder_links:
                if link[1] == node_id:
                    pipeline += base_env.Y.loc[t, link]
                    
            # 4. Backlog (Unfulfilled retail demand, only applies to retailers)
            backlog = 0
            if node_id in base_env.retail and t > 0:
                for link in base_env.retail_links:
                    if link[0] == node_id:
                        backlog += base_env.U.loc[t-1, link]
                        
            # Total Inventory Position
            IP = on_hand + pipeline - backlog
            
            # Calculate Replenishment Order
            order_qty = np.maximum(0.0, S - IP)
            order_actions[agent] = np.array([order_qty], dtype=np.float64)
            
            if self.log_file:
                if not hasattr(self, 'log_buffer'):
                    self.log_buffer = []
                self.log_buffer.append(f"{self.step_counter},{agent},{S},{IP},{order_qty}\n")
                if len(self.log_buffer) >= 10000:
                    with open(self.log_file, "a") as f:
                        f.writelines(self.log_buffer)
                    self.log_buffer.clear()
            
        self.step_counter += 1
        return self.env.step(order_actions)
    
    def close(self):
        if hasattr(self, 'log_buffer') and self.log_buffer and self.log_file:
            with open(self.log_file, "a") as f:
                f.writelines(self.log_buffer)
            self.log_buffer.clear()
        super().close()
