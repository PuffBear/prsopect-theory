import time
from experiments.phase3_9_leadtime import _run_single_config
import supersuit as ss
from stable_baselines3 import PPO

def test_speed():
    start = time.time()
    # patch the total_timesteps in _run_single_config by mocking PPO.learn
    original_learn = PPO.learn
    def mock_learn(self, total_timesteps, **kwargs):
        return original_learn(self, total_timesteps=1500, **kwargs)
    PPO.learn = mock_learn
    
    args = (42, 8, 1.0)
    print("Starting L=8, alpha=1.0, seed=42 (1500 steps)")
    res = _run_single_config(args)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")

test_speed()
