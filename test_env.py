import or_gym
env = or_gym.make("NetworkManagement-v1")
print("Num nodes:", env.unwrapped.num_nodes)
print("Node indices:", env.unwrapped.nodes)
print("Edges:", env.unwrapped.main_graph.edges)
print("Step test:", env.step(env.action_space.sample()))
