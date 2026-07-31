"""
Structural credit assignment for the multi-echelon substrate.

Replaces the scalar-w blend r_i = (1-w) local_i + w * global with a
Shapley-value / difference-reward decomposition.  Each agent is credited with
its *marginal contribution* to system profit, estimated by counterfactual
rollouts in which non-coalition agents place zero orders (the "lazy" baseline,
i.e. the same reference point as the lazy-agent attractor studied in the paper).

Two modes
---------
- mode="difference"  (a.k.a. Wonderful-Life / Aristocrat utility, COMA-style):
      phi_i = G(a) - G(a_{-i}, a_i = 0)
  One counterfactual rollout per agent per step.  This is the single-term
  leave-one-out Shapley approximation and is exact for the grand coalition.

- mode="shapley"     (Monte-Carlo Shapley over coalitions):
      phi_i = E_pi [ G(S_pi^i U {i}) - G(S_pi^i) ]
  averaged over `n_perm` random agent orderings.  Exact Shapley as n_perm -> inf
  (or set n_perm large enough to enumerate all N! orders for small N).

Design notes
------------
* The wrapper sits where CPTRewardWrapper sits (it receives ORDER actions and
  wraps the diagnostic/base env).  It can be composed *under* a CPT value
  function by passing `cpt_params` (so loss aversion can be layered on the
  Shapley-credited reward for the credit-assignment control).
* Counterfactual rollouts deep-copy the base or-gym env and pin the global RNG
  state so that the demand realization is held fixed across coalitions
  (ceteris paribus).  This is the one integration assumption to validate with a
  single run on the real env before launching the full sweep (see README).
"""
import copy
import math
import itertools
import numpy as np
from pettingzoo.utils import BaseParallelWrapper


class ShapleyRewardWrapper(BaseParallelWrapper):
    def __init__(self, env, mode="difference", n_perm=8, default_order=0.0,
                 cpt_params=None, reward_scale=1.0, seed=0, baseline="mean"):
        super().__init__(env)
        assert mode in ("difference", "shapley")
        assert baseline in ("zero", "mean")
        self.mode = mode
        self.n_perm = int(n_perm)
        self.default_order = float(default_order)
        # Counterfactual baseline for non-coalition agents:
        #   "zero" -> order `default_order` (extreme; high variance, lazy-biased)
        #   "mean" -> order the population-mean action (mild, low-variance; the
        #             agent is replaced by a "typical" agent rather than removed)
        self.baseline = baseline
        self.cpt_params = cpt_params or {}
        self.reward_scale = float(reward_scale)
        self._rng = np.random.RandomState(seed)

    # -- helpers ---------------------------------------------------------
    @property
    def _base(self):
        # the unwrapped or-gym NetInvMgmt env (has .P profit dataframe, .period)
        return self.env.unwrapped_env

    def _coalition_global_profit(self, actions, coalition):
        """Global profit (sum of node profits) for the one step in which agents
        in `coalition` place their real orders and all others place
        `default_order`.  Demand RNG is pinned so the realization matches the
        real step.  Returns a scalar; does NOT advance the real env."""
        rng_state = np.random.get_state()
        env_clone = copy.deepcopy(self.env)
        try:
            if self.baseline == "mean":
                # per-dimension population-mean action across all agents
                stack = np.array([np.asarray(a, dtype=np.float64) for a in actions.values()])
                baseline_act = stack.mean(axis=0)
            modified = {}
            for a, act in actions.items():
                if a in coalition:
                    modified[a] = act
                elif self.baseline == "mean":
                    modified[a] = baseline_act.copy()
                else:
                    modified[a] = np.array(
                        [self.default_order] * np.asarray(act).shape[-1], dtype=np.float64)
            # restore RNG so the clone samples the same demand as the real step
            np.random.set_state(rng_state)
            env_clone.step(modified)
            t = env_clone.unwrapped_env.period - 1
            P = env_clone.unwrapped_env.P
            g = float(sum(P.loc[t, env_clone.agent_name_to_id[a]] for a in actions))
        finally:
            np.random.set_state(rng_state)  # leave global RNG untouched
        return g

    def _credit(self, actions):
        agents = list(actions.keys())
        if self.mode == "difference":
            G_full = self._coalition_global_profit(actions, set(agents))
            phi = {}
            for i in agents:
                G_without = self._coalition_global_profit(
                    actions, set(agents) - {i})
                phi[i] = G_full - G_without
            return phi
        # ---- Monte-Carlo Shapley ----
        phi = {a: 0.0 for a in agents}
        N = len(agents)
        # enumerate all permutations if small enough, else sample
        if N <= 5 and self.n_perm >= math.factorial(N):
            perms = list(itertools.permutations(agents))
        else:
            perms = [list(self._rng.permutation(agents)) for _ in range(self.n_perm)]
        cache = {}
        def Gc(coal):
            key = frozenset(coal)
            if key not in cache:
                cache[key] = self._coalition_global_profit(actions, set(coal))
            return cache[key]
        for perm in perms:
            S = []
            G_prev = Gc(S)
            for i in perm:
                S2 = S + [i]
                G_new = Gc(S2)
                phi[i] += (G_new - G_prev)
                S, G_prev = S2, G_new
        for a in agents:
            phi[a] /= len(perms)
        return phi

    def _apply_cpt(self, agent, x):
        p = self.cpt_params.get(agent, {"lambda": 1.0, "alpha": 1.0, "beta": 1.0})
        lam, alpha, beta = p["lambda"], p["alpha"], p["beta"]
        x = x / self.reward_scale
        return x ** alpha if x >= 0 else -lam * ((-x) ** beta)

    # -- gym API ---------------------------------------------------------
    def step(self, actions):
        phi = self._credit(actions)               # counterfactual credit (pre-step)
        obs, rewards, term, trunc, info = self.env.step(actions)  # real step
        out = {}
        for a in rewards:
            shaped = self._apply_cpt(a, phi.get(a, 0.0))
            out[a] = shaped
            info.setdefault(a, {})
            info[a].update(raw_local_reward=rewards[a],
                           shapley_credit=phi.get(a, 0.0),
                           shaped_reward=shaped,
                           true_reward=info[a].get("true_reward", rewards[a]))
        return obs, out, term, trunc, info
