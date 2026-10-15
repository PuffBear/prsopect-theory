"""
Centralized-critic MAPPO policy for IMAGE observations (CNN CTDE), the spatial
analogue of agents/mappo_policy.py. Pairs with
env.global_state_image_wrapper.GlobalStateImageWrapper, which stacks observations
on the channel axis:

    obs = [ own_img (C ch) | global_img (N*C ch) ]   shape ((1+N)*C, H, W)

- ACTOR  CNN reads obs[:, :C]            (own image, decentralized execution)
- CRITIC CNN reads obs[:, C:]            (joint image, centralized training)

Both extractors emit the same feature_dim so SB3's MlpExtractor accepts either.
Mirrors mappo_policy.py exactly: force share_features_extractor=False, install a
local-slice CNN actor extractor, then swap in a global-slice CNN critic extractor
and rebuild the optimizer so the critic's params train.

The CNN is a small 3x3 stack sized for MAgent's 13x13 maps (SB3's NatureCNN
collapses below kernel size on small grids). Validate with the wiring check in
exp_mappo_magent.py before cluster time -- it asserts the critic params are in
the optimizer AND that the critic reads the global channel slice.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

FEATURES_DIM = 128


def _conv_stack(in_ch):
    """3x3 conv stack sized for ~13x13 maps (stride 1 then 2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 16, 3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
    )


class _LocalCNNActorExtractor(BaseFeaturesExtractor):
    """Actor features from the OWN-image channel slice (first local_channels)."""

    def __init__(self, observation_space, local_channels, features_dim=FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        self.local_channels = int(local_channels)
        _c, h, w = observation_space.shape
        self.cnn = _conv_stack(self.local_channels)
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, self.local_channels, h, w)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs[:, :self.local_channels]))


class _CentralCNNCriticExtractor(BaseFeaturesExtractor):
    """Critic features from the GLOBAL-image channel slice (all agents)."""

    def __init__(self, observation_space, local_channels, features_dim=FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        self.local_channels = int(local_channels)
        total_c, h, w = observation_space.shape
        self.global_channels = int(total_c) - self.local_channels
        self.cnn = _conv_stack(self.global_channels)
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, self.global_channels, h, w)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs[:, self.local_channels:]))


class CentralizedCriticImagePolicy(ActorCriticPolicy):
    """ActorCriticPolicy with a CNN centralized critic over the global channel
    slice. local_channels (= per-agent obs channels) passed via policy_kwargs."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 local_channels=None, **kwargs):
        if local_channels is None:
            raise ValueError("CentralizedCriticImagePolicy requires local_channels "
                             "(pass via policy_kwargs).")
        self._local_channels = int(local_channels)
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = _LocalCNNActorExtractor
        kwargs["features_extractor_kwargs"] = dict(
            local_channels=self._local_channels, features_dim=FEATURES_DIM)
        # MAgent obs is a small bounded feature stack, not 0-255 pixels
        kwargs.setdefault("normalize_images", False)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.vf_features_extractor = _CentralCNNCriticExtractor(
            observation_space, self._local_channels, features_dim=FEATURES_DIM
        ).to(self.device)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1.0), **(self.optimizer_kwargs or {})
        )
