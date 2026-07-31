"""
Self-contained QMIX / IQL learner on the existing PettingZoo parallel stack.

WHY THIS EXISTS (and why not EPyMARL)
-------------------------------------
The Value-Based plan asks for the value-based half of the credit-assignment x
learning-paradigm 2x2:

    QMIX  = centralized value-based   (monotonic mixing network over the joint state)
    IQL   = independent value-based   (per-agent Q, no mixer)

The whole PPO/MAPPO pipeline in this repo is SB3 + SuperSuit + PettingZoo with no
sacred/EPyMARL. Rather than vendoring EPyMARL (heavy, needs a second HPC env, and
hard to smoke-test on a laptop), this is a compact, audited QMIX/IQL that reuses
the EXACT wrapper stack the policy-gradient cells use:

    ... -> CPTRewardWrapper -> (action shell) -> GlobalStateWrapper (OUTERMOST)

GlobalStateWrapper already produces, per agent, an augmented observation
    [ own_part(i) | global_state ]
where own_part(i) = [ base_obs(i) | one-hot id(i) ] has length `local_dim`, and
global_state = concat_j own_part(j). That is precisely the QMIX input contract:
  - per-agent Q-network reads own_part(i)        (the first local_dim entries)
  - the mixer reads the global_state             (the trailing entries)
IQL is the same minus the mixer (sum of per-agent chosen-action Q-values).

This file holds the algorithm only; environment builders / drivers live in
experiments/exp_qmix_*.py. The driver's --wiring_check asserts the mixer sees the
joint state and IQL does not (mirrors the MAPPO mislabel we got burned by).

DESIGN NOTES
------------
- Parameter sharing across agents (one Q-network with the one-hot id in obs),
  matching the MAPPO actor. The id is what lets a shared net specialize.
- Monotone mixer: mixing weights are produced by hypernetworks of the global
  state and passed through abs() so dQ_tot/dQ_i >= 0 (Rashid et al. 2018). This
  is the property the --wiring_check and the toy sanity check verify.
- Discrete actions only (value-based). Supply chain gets a Discrete(K) shell;
  MPE simple_spread is natively Discrete(5).
- Iso-sample budget: training is measured in ENVIRONMENT STEPS (env.step calls x
  n_agents transitions), not gradient updates, so it is comparable to PPO/MAPPO.
- All value-based knobs (lr, buffer, target-update, epsilon anneal) are frozen
  defaults here and must NOT vary with lambda (see plan section 3).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Frozen value-based hyperparameters (do NOT vary with lambda) ------------
DEFAULTS = dict(
    lr=5e-4,
    gamma=0.99,
    buffer_size=5000,            # episodes
    batch_size=32,               # episodes per gradient batch
    target_update_interval=200,  # gradient steps between hard target syncs
    eps_start=1.0,
    eps_finish=0.05,
    eps_anneal_steps=50_000,     # env steps over which epsilon anneals
    hidden_dim=64,
    mixer_embed_dim=32,
    train_every=1,               # gradient step every N completed episodes
    grad_clip=10.0,
    learning_starts=32,          # episodes in buffer before training
)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class RunningNorm:
    """Welford running mean/std of the scalar team reward.

    The supply-chain CPT rewards are O(100s); without conditioning the TD target
    on their scale the mixer's MSE explodes (observed value loss ~1e12 in the
    smoke test) and learning is unstable. This standardizes the regression target
    (reward and bootstrapped value are scaled by the same running std) -- a
    LEARNING-side normalization that leaves the reward PIPELINE byte-identical to
    the PPO/MAPPO runs (it is the QMIX analogue of SB3's value normalization).
    MPE rewards are O(1) so the scale stays ~1 there and this is a no-op.
    """

    def __init__(self):
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 1e-4

    def update(self, x):
        # Chan et al. parallel variance merge: fold a whole batch in one shot
        # (vectorized -- no per-element Python loop in the hot training path).
        b = np.atleast_1d(np.asarray(x, dtype=np.float64))
        n_b = b.size
        if n_b == 0:
            return
        mean_b = b.mean()
        m2_b = ((b - mean_b) ** 2).sum()
        delta = mean_b - self.mean
        tot = self.count + n_b
        self.mean += delta * n_b / tot
        self.m2 += m2_b + delta * delta * self.count * n_b / tot
        self.count = tot

    @property
    def std(self):
        var = self.m2 / self.count if self.count > 1 else 1.0
        return float(max(np.sqrt(var), 1e-6))


# --- Networks ----------------------------------------------------------------

class AgentQNet(nn.Module):
    """Shared per-agent Q-network: own_part(i) -> Q-values over n_actions.

    A feed-forward net (no recurrence) keeps it comparable to the MLP PPO actor
    and avoids hidden-state bookkeeping; the one-hot id in own_part supplies the
    agent identity that recurrence would otherwise have to track.
    """

    def __init__(self, local_dim, n_actions, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, own_part):
        return self.net(own_part)


class QMixer(nn.Module):
    """Monotonic mixing network (Rashid et al. 2018, QMIX).

    Q_tot = w2 . elu(W1 @ q + b1) + b2, where W1, w2 are produced by
    hypernetworks of the global state and forced non-negative via abs(), giving
    the monotonicity dQ_tot/dQ_i >= 0 that lets decentralized argmax recover the
    centralized argmax.
    """

    def __init__(self, n_agents, state_dim, embed_dim):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        # hypernet -> first layer weights (|.|) and bias
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        # hypernet -> second layer weights (|.|) and bias (bias via 2-layer net)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, state):
        """agent_qs: [B, n_agents] chosen-action Q per agent. state: [B, state_dim]."""
        B = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        qs = agent_qs.view(B, 1, self.n_agents)
        hidden = F.elu(torch.bmm(qs, w1) + b1)            # [B,1,embed]
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2                # [B,1,1]
        return q_tot.view(B, 1)


# --- Learner -----------------------------------------------------------------

class EpisodeBuffer:
    """Stores whole episodes as padded transition tensors. Episodes are short
    (supply 30-50 steps, MPE 50), so per-episode storage is cheap and lets us
    do the QMIX joint-state TD target cleanly."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self._pos = 0

    def add(self, episode):
        if len(self.data) < self.capacity:
            self.data.append(episode)
        else:
            self.data[self._pos] = episode
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size, rng):
        idx = rng.choice(len(self.data), size=min(batch_size, len(self.data)),
                         replace=False)
        return [self.data[i] for i in idx]

    def __len__(self):
        return len(self.data)


class QLearner:
    """QMIX (use_mixer=True) or IQL (use_mixer=False).

    The ONLY structural difference between the two cells is `use_mixer`; every
    other knob is shared and frozen, so the cell comparison is clean.
    """

    def __init__(self, n_agents, local_dim, state_dim, n_actions,
                 use_mixer, device="cpu", seed=0, **kw):
        cfg = dict(DEFAULTS); cfg.update(kw)
        self.cfg = cfg
        self.n_agents = n_agents
        self.local_dim = local_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.use_mixer = use_mixer
        self.device = torch.device(device)
        self.rng = np.random.RandomState(seed)

        self.q = AgentQNet(local_dim, n_actions, cfg["hidden_dim"]).to(self.device)
        self.q_target = AgentQNet(local_dim, n_actions, cfg["hidden_dim"]).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        params = list(self.q.parameters())

        if use_mixer:
            self.mixer = QMixer(n_agents, state_dim, cfg["mixer_embed_dim"]).to(self.device)
            self.mixer_target = QMixer(n_agents, state_dim, cfg["mixer_embed_dim"]).to(self.device)
            self.mixer_target.load_state_dict(self.mixer.state_dict())
            params += list(self.mixer.parameters())
        else:
            self.mixer = self.mixer_target = None

        self.opt = torch.optim.Adam(params, lr=cfg["lr"])
        self.buffer = EpisodeBuffer(cfg["buffer_size"])
        self.rnorm = RunningNorm()  # conditions the TD target on the reward scale
        self.env_steps = 0          # ENV-STEP budget counter (iso-sample)
        self.grad_steps = 0
        self.last_value_loss = float("nan")

    # --- epsilon-greedy action selection (exploration noise is FROZEN) -------
    def epsilon(self):
        c = self.cfg
        frac = min(1.0, self.env_steps / max(1, c["eps_anneal_steps"]))
        return c["eps_start"] + frac * (c["eps_finish"] - c["eps_start"])

    def select_actions(self, own_parts, explore=True):
        """own_parts: dict agent->own_part vector. Returns dict agent->int action."""
        eps = self.epsilon() if explore else 0.0
        out = {}
        with torch.no_grad():
            for a, op in own_parts.items():
                if explore and self.rng.rand() < eps:
                    out[a] = int(self.rng.randint(self.n_actions))
                else:
                    t = torch.as_tensor(op, dtype=torch.float32, device=self.device).unsqueeze(0)
                    out[a] = int(self.q(t).argmax(dim=1).item())
        return out

    # --- training step over sampled episodes ---------------------------------
    def train(self):
        c = self.cfg
        if len(self.buffer) < c["learning_starts"]:
            return
        batch = self.buffer.sample(c["batch_size"], self.rng)
        for ep in batch:
            self.rnorm.update(ep["reward"])
        loss = self._td_loss(batch)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for g in self.opt.param_groups for p in g["params"]], c["grad_clip"])
        self.opt.step()
        self.grad_steps += 1
        self.last_value_loss = float(loss.item())
        if self.grad_steps % c["target_update_interval"] == 0:
            self.q_target.load_state_dict(self.q.state_dict())
            if self.use_mixer:
                self.mixer_target.load_state_dict(self.mixer.state_dict())

    def _td_loss(self, episodes):
        """Standard (Q)MIX TD(0) loss summed over the agents' chosen actions.

        Each episode is a dict of stacked arrays:
          own_parts [T, n_agents, local_dim], actions [T, n_agents],
          reward [T] (team reward), state [T, state_dim],
          next_own_parts [T, n_agents, local_dim], next_state [T, state_dim],
          done [T]
        """
        dev = self.device
        total = torch.zeros((), device=dev)
        n_trans = 0
        for ep in episodes:
            T = ep["actions"].shape[0]
            op = torch.as_tensor(ep["own_parts"], dtype=torch.float32, device=dev)      # [T,N,L]
            nop = torch.as_tensor(ep["next_own_parts"], dtype=torch.float32, device=dev)
            act = torch.as_tensor(ep["actions"], dtype=torch.long, device=dev)          # [T,N]
            rew = torch.as_tensor(ep["reward"], dtype=torch.float32, device=dev)        # [T]
            done = torch.as_tensor(ep["done"], dtype=torch.float32, device=dev)         # [T]

            q_all = self.q(op.view(T * self.n_agents, -1)).view(T, self.n_agents, -1)
            chosen = q_all.gather(2, act.unsqueeze(2)).squeeze(2)                       # [T,N]

            with torch.no_grad():
                nq = self.q_target(nop.view(T * self.n_agents, -1)).view(T, self.n_agents, -1)
                nmax = nq.max(dim=2).values                                            # [T,N]

            if self.use_mixer:
                st = torch.as_tensor(ep["state"], dtype=torch.float32, device=dev)      # [T,state_dim]
                nst = torch.as_tensor(ep["next_state"], dtype=torch.float32, device=dev)
                q_tot = self.mixer(chosen, st).squeeze(1)                              # [T]
                with torch.no_grad():
                    nq_tot = self.mixer_target(nmax, nst).squeeze(1)                   # [T]
            else:
                q_tot = chosen.sum(dim=1)                                              # [T]
                with torch.no_grad():
                    nq_tot = nmax.sum(dim=1)                                           # [T]

            # Standardize the regression target by the running team-reward std so
            # the loss is well-conditioned regardless of reward magnitude. Q_tot
            # and the target live in the same standardized space (Q is learned to
            # predict the scaled return); greedy argmax is scale-invariant so
            # action selection / eval are unaffected.
            scale = self.rnorm.std
            target = rew / scale + self.cfg["gamma"] * (1.0 - done) * nq_tot
            # Huber is robust to the residual heavy tail at the collapse boundary.
            total = total + F.smooth_l1_loss(q_tot, target, reduction="sum")
            n_trans += T
        return total / max(1, n_trans)
