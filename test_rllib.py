import ray
from ray.tune.registry import register_env
from experiments.phase_transition import env_creator
from ray.rllib.algorithms.ppo import PPOConfig

register_env("cpt_supply_chain", env_creator)

config = (
    PPOConfig()
    .environment("cpt_supply_chain")
    .training(num_epochs=1)
    .env_runners(num_env_runners=0)  # Try with local worker only
)
try:
    algo = config.build()
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
