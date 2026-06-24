"""
Local viability GATE for MAgent2 battle as the 4th substrate (run before any
integration/sweep work -- the same gate-then-build discipline used for LBF/RWARE).

QUESTION this answers: does the MAgent battle task run end-to-end through the CPT
w-dial with a CNN policy, AND does a high-w (team-reward) red team learn to
coordinate and beat the random baseline? If yes, MAgent is a viable 4th substrate
and the full CNN-MAPPO sweep is worth building. If no, stop.

Design:
  - battle_v4, small map, 2v2. We CONTROL the red team; blue acts randomly
    (fixed baseline opponent for the gate).
  - red reward is run through CPTRewardWrapper with global_reward_weight=w:
    w=0 -> selfish (own kills), w=1 -> red-team-sum (coordinated). The gate
    trains at w=1 (the coordinated end) -- if the team can't coordinate even at
    w=1, no interior boundary can exist (same logic as the LBF/RWARE smokes).
  - CNN policy (SB3 CnnPolicy) since obs is a (13,13,5) spatial stack.

  PART 1 (no --train): random red-vs-random-blue, report red team reward baseline.
  PART 2 (--train): PPO CnnPolicy at w=1, report trained red team reward.
    PASS: trained red team reward clearly above the random baseline (coordination
    learned). This is the viability check, not the full result.

    python experiments/run_magent_smoke_local.py            # PART 1 (~1 min)
    python experiments/run_magent_smoke_local.py --train    # PART 2 (~10-20 min)
"""
import os, sys, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from magent2.environments import battle_v4
from pettingzoo.utils import BaseParallelWrapper
from agents.cpt_wrapper import CPTRewardWrapper

MAP_SIZE, MAX_CYCLES = 16, 200   # 6 red agents (richer coordination); re-gate size


def _small_cnn():
    """SB3's default NatureCNN (8x8/4x4/3x3 strided convs) is built for 84x84
    Atari frames and collapses a 13x13 map below its kernel size. This is a small
    3x3 conv stack sized for the (5,13,13) MAgent obs. Defined lazily so PART 1
    (no torch/SB3) stays import-light."""
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym

    class SmallCNN(BaseFeaturesExtractor):
        def __init__(self, obs_space: gym.Space, features_dim: int = 128):
            super().__init__(obs_space, features_dim)
            c = obs_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 16, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten())
            import torch
            with torch.no_grad():
                n = self.cnn(torch.zeros(1, *obs_space.shape)).shape[1]
            self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

        def forward(self, x):
            return self.linear(self.cnn(x))

    return SmallCNN


class RedTeamWrapper(BaseParallelWrapper):
    """Expose ONLY the red team to the learner; step blue with random actions
    internally. Yields a PettingZoo parallel env whose agents are red_* only, so
    CPTRewardWrapper's global_reward = sum(red rewards) is the red TEAM reward."""
    def __init__(self, env):
        super().__init__(env)
        self._all = env.possible_agents
        self.possible_agents = [a for a in self._all if a.startswith("red")]
        self.agents = list(self.possible_agents)
        # MAgent obs is (H,W,C) channels-last; SB3 NatureCNN wants channels-first
        # (C,H,W). Transpose here so train AND eval are consistent (supersuit in
        # this version has no transpose wrapper).
        import gymnasium as gym
        self._obs_spaces = {}
        for a in self.possible_agents:
            s = env.observation_space(a)
            h, w, c = s.shape
            self._obs_spaces[a] = gym.spaces.Box(
                low=np.transpose(s.low, (2, 0, 1)),
                high=np.transpose(s.high, (2, 0, 1)),
                shape=(c, h, w), dtype=s.dtype)

    def observation_space(self, a): return self._obs_spaces[a]
    def action_space(self, a): return self.env.action_space(a)

    @staticmethod
    def _cf(o):                                   # channels-last -> channels-first
        return np.transpose(np.asarray(o), (2, 0, 1))

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = [a for a in self.env.agents if a.startswith("red")]
        return ({a: self._cf(obs[a]) for a in self.agents},
                {a: info.get(a, {}) for a in self.agents})

    def step(self, red_actions):
        acts = dict(red_actions)
        for a in self.env.agents:
            if a.startswith("blue"):
                acts[a] = self.env.action_space(a).sample()
        obs, rew, term, trunc, info = self.env.step(acts)
        self.agents = [a for a in self.env.agents if a.startswith("red")]
        keep = lambda d: {a: d[a] for a in self.agents if a in d}
        kept_obs = {a: self._cf(obs[a]) for a in self.agents if a in obs}
        return kept_obs, keep(rew), keep(term), keep(trunc), keep(info)


def build_env(w):
    base = battle_v4.parallel_env(map_size=MAP_SIZE, max_cycles=MAX_CYCLES,
                                  extra_features=False)
    red = RedTeamWrapper(base)
    params = {a: {"lambda": 1.0, "alpha": 1.0, "beta": 1.0}
              for a in red.possible_agents}
    return CPTRewardWrapper(red, params, reward_scale=1.0, global_reward_weight=w)


def team_reward(env, policy=None, n_ep=8, base_seed=0):
    """Mean red-team TRUE (raw) reward summed over an episode. Obs is already
    channels-first (RedTeamWrapper transposes), so it feeds model.predict directly."""
    tot = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_r = 0.0
        while env.agents:
            if policy is None:
                acts = {a: env.action_space(a).sample() for a in env.agents}
            else:
                acts = {a: int(policy.predict(np.asarray(obs[a]), deterministic=True)[0])
                        for a in env.agents}
            obs, rew, term, trunc, info = env.step(acts)
            ep_r += sum(i.get("true_reward", 0.0) for i in info.values())
        tot.append(ep_r)
    return float(np.mean(tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--timesteps", type=int, default=300_000)
    args = ap.parse_args()

    print("=== PART 1: random red vs random blue ===")
    base_r = team_reward(build_env(1.0), policy=None, n_ep=8)
    print(f"  random red team reward/episode = {base_r:.4f}")
    print("  PART 1: PASS (stack runs end-to-end through CPT + CNN obs)")

    if not args.train:
        print("\n(probe only; rerun with --train for the coordination gate)")
        return

    import supersuit as ss
    from stable_baselines3 import PPO
    print(f"\n=== PART 2: PPO CnnPolicy @ w=1, {args.timesteps} steps ===")
    env = build_env(1.0)
    vec = ss.black_death_v3(env)               # pad dead agents -> fixed agent set
    # obs is already channels-first (RedTeamWrapper transposes). normalize_images
    # =False below because it's a small bounded feature stack, not 0-255 pixels.
    vec = ss.pettingzoo_env_to_vec_env_v1(vec)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class="stable_baselines3")
    vec.seed = lambda s: None
    model = PPO("CnnPolicy", vec, verbose=0, n_steps=256, batch_size=512, seed=42,
                policy_kwargs=dict(normalize_images=False,
                                   features_extractor_class=_small_cnn(),
                                   features_extractor_kwargs=dict(features_dim=128)))
    model.learn(total_timesteps=args.timesteps)

    trained_r = team_reward(build_env(1.0), policy=model, n_ep=8, base_seed=100)
    print(f"  trained red team reward/episode = {trained_r:.4f}  (random {base_r:.4f})")
    passed = trained_r > base_r + abs(base_r) * 0.5 + 1e-3
    print(f"  PART 2: {'PASS' if passed else 'FAIL'} "
          f"(coordination {'learned -> MAgent viable, build the sweep' if passed else 'NOT learned -> reconsider'})")


if __name__ == "__main__":
    main()
