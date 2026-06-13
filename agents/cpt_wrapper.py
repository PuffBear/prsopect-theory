from pettingzoo.utils import BaseParallelWrapper

class CPTRewardWrapper(BaseParallelWrapper):
    """
    Applies the Prospect Theory value function to raw rewards.
    v(x) = x^alpha if x >= 0
           -lambda * (-x)^beta if x < 0
    """
    def __init__(self, env, agent_params, reward_scale=1.0, global_reward_weight=0.0):
        super().__init__(env)
        self.agent_params = agent_params
        self.reward_scale = reward_scale
        self.global_reward_weight = global_reward_weight
        
    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        
        global_reward = sum(rewards.values())
        
        cpt_rewards = {}
        for agent, reward in rewards.items():
            blended_reward = (1.0 - self.global_reward_weight) * reward + self.global_reward_weight * global_reward
            
            params = self.agent_params.get(agent, {"lambda": 1.0, "alpha": 1.0, "beta": 1.0})
            lam = params["lambda"]
            alpha = params["alpha"]
            beta = params["beta"]
            
            scaled_reward = blended_reward / self.reward_scale
            
            if scaled_reward >= 0:
                cpt_reward = scaled_reward ** alpha
            else:
                cpt_reward = -lam * ((-scaled_reward) ** beta)
                
            cpt_rewards[agent] = cpt_reward
            
            if agent not in info:
                info[agent] = {}
            info[agent]["raw_local_reward"] = reward
            info[agent]["raw_global_reward"] = global_reward
            info[agent]["blended_reward"] = blended_reward
            info[agent]["scaled_reward"] = scaled_reward
            info[agent]["cpt_reward"] = cpt_reward
            # Preserve true_reward set by an inner wrapper (e.g. MPELocalRewardWrapper
            # sets this to the raw env reward before w-blending).  Only fall back to
            # the blended reward if no inner wrapper has set it.
            if "true_reward" not in info[agent]:
                info[agent]["true_reward"] = reward
                
        return obs, cpt_rewards, term, trunc, info
