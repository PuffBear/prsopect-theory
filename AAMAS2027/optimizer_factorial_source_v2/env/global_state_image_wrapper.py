"""
Image (CNN) global-state augmentation for centralized-critic MAPPO on spatial
observations (MAgent2). The vector GlobalStateWrapper flattens obs with .ravel(),
which destroys the spatial structure a CNN needs -- this is the channels-stacked
analogue that keeps every observation a (C,H,W) image.

Each agent's CHANNELS-FIRST image obs (C,H,W) is augmented to:

    own_img(i)        = obs(i)                         (C,  H, W)
    global_img        = concat_j own_img(j)            (N*C, H, W)  fixed order
    augmented_obs(i)  = concat[ own_img(i), global_img ]  ((1+N)*C, H, W)

- ACTOR  reads the first  C  channels (own image, decentralized execution).
- CRITIC reads the trailing N*C channels (joint image, centralized training).

The local channel count `local_channels` (= C) is read by the MAPPO image policy
to slice actor vs critic input on the channel axis -- the spatial (CNN) mirror of
the flat [own | global] slicing in env/global_state_wrapper.py.

No agent-id one-hot is needed here: MAgent agents already differ by their
egocentric spatial view (each sees itself centered with distinct neighbours), so
the symmetry that forced the one-hot in the flat MPE case does not arise.

Sits OUTERMOST in the parallel stack (after reward wrappers) so it only touches
observations; rewards / info pass through unchanged. Assumes obs is already
channels-first (C,H,W) -- pair it with a wrapper that transposes MAgent's native
channels-last (H,W,C) first (see RedTeamWrapper / the MAgent driver).
"""
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper


class GlobalStateImageWrapper(BaseParallelWrapper):
    """Append the channel-stacked joint image (global state) to every agent's
    own image, on the channel axis, preserving spatial structure for a CNN."""

    def __init__(self, env):
        super().__init__(env)
        self._agents = list(self.possible_agents)
        self._n = len(self._agents)
        base = self.env.observation_space(self._agents[0])
        assert len(base.shape) == 3, "expected channels-first image obs (C,H,W)"
        self._c, self._h, self._w = (int(x) for x in base.shape)
        self.local_channels = self._c            # actor input channel count
        aug_c = self._c * (1 + self._n)          # own + N agents
        # bounds: replicate the base low/high across the stacked channels
        lo = float(np.min(base.low)) if np.isfinite(np.min(base.low)) else 0.0
        hi = float(np.max(base.high)) if np.isfinite(np.max(base.high)) else 2.0
        self._aug_space = spaces.Box(low=lo, high=hi,
                                     shape=(aug_c, self._h, self._w),
                                     dtype=np.float32)

    def observation_space(self, agent):
        return self._aug_space

    def action_space(self, agent):
        return self.env.action_space(agent)

    def _global_img(self, obs_dict):
        """Channel-concat every agent's image in fixed order (zeros if absent)."""
        z = np.zeros((self._c, self._h, self._w), dtype=np.float32)
        return np.concatenate(
            [np.asarray(obs_dict.get(a, z), dtype=np.float32) for a in self._agents],
            axis=0)

    def _augment(self, obs_dict):
        g = self._global_img(obs_dict)
        return {a: np.concatenate([np.asarray(o, dtype=np.float32), g], axis=0)
                for a, o in obs_dict.items()}

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs), info

    def step(self, actions):
        obs, rewards, term, trunc, info = self.env.step(actions)
        return self._augment(obs), rewards, term, trunc, info
