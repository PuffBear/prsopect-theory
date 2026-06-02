from env.marl_or_gym_wrapper import MultiAgentNetInvMgmt
import numpy as np

env = MultiAgentNetInvMgmt()
obs, info = env.reset()
print("Agents:", env.agents)
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
print("Actions:", actions)
obs, rewards, term, trunc, info = env.step(actions)
print("Rewards:", rewards)
