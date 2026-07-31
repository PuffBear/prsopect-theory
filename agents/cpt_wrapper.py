from pettingzoo.utils import BaseParallelWrapper

class CPTRewardWrapper(BaseParallelWrapper):
    """
    Applies the Prospect Theory value function to raw rewards, about a reference point.
    Let u = (blended_reward - reference_point) / reward_scale. Then
        v(u) = u^alpha              if u >= 0   (gain, relative to reference_point)
               -lambda * (-u)^beta  if u < 0    (loss, relative to reference_point)

    reference_point (z0) is the gain-loss reference. Default 0.0 reproduces the original
    behavior exactly (the paper's condition); Control A sweeps it to test whether the
    loss-aversion boundary shift survives plausible alternative reference points.
    """
    def __init__(self, env, agent_params, reward_scale=1.0, global_reward_weight=0.0,
                 reference_point=0.0, local_scale=1.0, global_scale=1.0,
                 output_scale=1.0):
        super().__init__(env)
        self.agent_params = agent_params
        self.reward_scale = reward_scale
        self.global_reward_weight = global_reward_weight
        self.reference_point = reference_point
        # Control C: independent reward-unit factors applied BEFORE the (1-w)/w blend,
        # so they change the local-to-global incentive ratio. Both default 1.0 =>
        # identical to the original behavior (every other experiment is unaffected).
        self.local_scale = local_scale
        self.global_scale = global_scale
        # Control B: divides the FINAL transformed (CPT) reward by a per-condition
        # constant, to std/RMS-match magnitudes across lambda (scale-matched control).
        # Default 1.0 => identical to the original behavior.
        self.output_scale = output_scale
        
    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        
        global_reward = sum(rewards.values())
        
        cpt_rewards = {}
        for agent, reward in rewards.items():
            blended_reward = ((1.0 - self.global_reward_weight) * (self.local_scale * reward)
                              + self.global_reward_weight * (self.global_scale * global_reward))
            
            params = self.agent_params.get(agent, {"lambda": 1.0, "alpha": 1.0, "beta": 1.0})
            lam = params["lambda"]
            alpha = params["alpha"]
            beta = params["beta"]
            
            # Shift by the gain-loss reference point BEFORE the value transform, then
            # scale. reference_point=0 (default) => identical to the original code.
            scaled_reward = (blended_reward - self.reference_point) / self.reward_scale

            if scaled_reward >= 0:
                cpt_reward = scaled_reward ** alpha
            else:
                cpt_reward = -lam * ((-scaled_reward) ** beta)

            # Control B: post-transform RMS/std match (output_scale=1.0 => unchanged).
            cpt_reward = cpt_reward / self.output_scale

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
