import or_gym
from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
env = MultiAgentNetInvMgmt(env_config={'uniform_lead_time': 0})
print("Pipeline length (L=0):", env.unwrapped_env.pipeline_length)
env = MultiAgentNetInvMgmt(env_config={'uniform_lead_time': 8})
print("Pipeline length (L=8):", env.unwrapped_env.pipeline_length)
env = MultiAgentNetInvMgmt(uniform_lead_time=8)
print("Pipeline length (kwargs directly, L=8):", env.unwrapped_env.pipeline_length)
