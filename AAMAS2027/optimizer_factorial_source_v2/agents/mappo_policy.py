"""
Centralized-critic MAPPO policy for Stable-Baselines3 (CTDE).

Works on observations augmented by env.global_state_wrapper.GlobalStateWrapper:

    obs = [ own_obs (local_dim) | global_state (n_agents * local_dim) ]

- ACTOR  reads only obs[:, :local_dim]            (decentralized execution)
- CRITIC reads only obs[:, local_dim:]            (centralized training)

Both feature extractors emit the SAME feature dimension so SB3's shared
MlpExtractor (single feature_dim) accepts either path. We use SB3's native
share_features_extractor=False machinery -- which already calls
pi_features_extractor(obs) and vf_features_extractor(obs) separately -- and
swap in a global-slice critic extractor after construction.

NOTE: this is a from-scratch implementation; validate with the MAPPO smoke
(run_mappo_smoke_local.py) before committing cluster time. The smoke asserts
the critic's parameters are actually in the optimizer.
"""
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

FEATURES_DIM = 128


class _LocalActorExtractor(BaseFeaturesExtractor):
    """Actor features from the OWN-obs slice only."""

    def __init__(self, observation_space, local_dim, features_dim=FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        self.local_dim = int(local_dim)
        self.net = nn.Sequential(
            nn.Linear(self.local_dim, 128), nn.ReLU(),
            nn.Linear(128, features_dim), nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs[:, :self.local_dim])


class _CentralCriticExtractor(BaseFeaturesExtractor):
    """Critic features from the GLOBAL-state slice (all agents' obs)."""

    def __init__(self, observation_space, local_dim, features_dim=FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        self.local_dim = int(local_dim)
        global_dim = int(observation_space.shape[0]) - self.local_dim
        self.net = nn.Sequential(
            nn.Linear(global_dim, 128), nn.ReLU(),
            nn.Linear(128, features_dim), nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs[:, self.local_dim:])


class CentralizedCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with a centralized critic over the global-state slice."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 local_dim=None, **kwargs):
        if local_dim is None:
            raise ValueError("CentralizedCriticPolicy requires local_dim "
                             "(pass via policy_kwargs).")
        self._local_dim = int(local_dim)
        # Force separate extractors; actor extractor = local slice.
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = _LocalActorExtractor
        kwargs["features_extractor_kwargs"] = dict(
            local_dim=self._local_dim, features_dim=FEATURES_DIM)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        # Replace the (default-built) vf extractor with the centralized critic
        # over the global slice, then rebuild the optimizer so its params train.
        self.vf_features_extractor = _CentralCriticExtractor(
            observation_space, self._local_dim, features_dim=FEATURES_DIM
        ).to(self.device)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1.0), **(self.optimizer_kwargs or {})
        )
