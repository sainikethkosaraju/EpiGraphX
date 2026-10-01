"""Vaccination policies on the dynamic network, plus a PPO meta-policy.

Each step a policy may move up to ``budget`` *susceptible* nodes to R.
(Picking nodes that are already infected or immune wastes the budget -- the
first version of this project did that, re-selecting the same top-degree
hubs every step.)
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .config import Config
from .graphs import generate_dynamic_graphs
from .simulation import R, S, adjacency, initial_state, is_infectious, step


def _top_susceptible(score: dict, states, budget):
    cand = [v for v in sorted(score, key=score.get, reverse=True) if states[v] == S]
    return cand[:budget]


def no_policy(G, states, budget, rng):
    return []


def random_policy(G, states, budget, rng):
    sus = np.flatnonzero(states == S)
    return list(rng.choice(sus, size=min(budget, len(sus)), replace=False)) if len(sus) else []


def degree_policy(G, states, budget, rng):
    return _top_susceptible(dict(G.degree()), states, budget)


def pagerank_policy(G, states, budget, rng):
    return _top_susceptible(nx.pagerank(G), states, budget)


def acquaintance_policy(G, states, budget, rng):
    """Vaccinate a random neighbour of a random node (Cohen et al., 2003).

    Needs no global knowledge of the network, yet tends to find hubs.
    """
    chosen, sus = set(), np.flatnonzero(states == S)
    for _ in range(20 * budget):
        if len(chosen) >= budget or not len(sus):
            break
        nbrs = list(G.neighbors(int(rng.integers(G.number_of_nodes()))))
        if nbrs:
            v = nbrs[rng.integers(len(nbrs))]
            if states[v] == S:
                chosen.add(v)
    return list(chosen)


POLICIES = {
    "No intervention": no_policy,
    "Random": random_policy,
    "Acquaintance": acquaintance_policy,
    "Degree": degree_policy,
    "PageRank": pagerank_policy,
}


def run_policy(policy, graphs, cfg: Config, budget: int, rng: np.random.Generator):
    """Infectious count per step and cumulative infections under ``policy``."""
    n = graphs[0].number_of_nodes()
    state = initial_state(n, cfg.epidemic, rng)
    curve, ever = [], is_infectious(state.states).copy()
    for G in graphs:
        for v in policy(G, state.states, budget, rng):
            state.states[v] = R
        state = step(state, adjacency(G, n), cfg.epidemic, rng)
        ever |= is_infectious(state.states)
        curve.append(int(is_infectious(state.states).sum()))
    return np.array(curve), int(ever.sum())


def compare_policies(cfg: Config, budget: int = 5, n_runs: int = 30, seed: int = 0):
    """Every policy faces the same network and the same random seed in each run."""
    rows, curves = [], {name: [] for name in POLICIES}
    for r in range(n_runs):
        graphs = generate_dynamic_graphs(cfg.graph, np.random.default_rng([seed, r, 0]))
        for name, pol in POLICIES.items():
            curve, total = run_policy(pol, graphs, cfg, budget, np.random.default_rng([seed, r, 1]))
            curves[name].append(curve)
            rows.append({"policy": name, "run": r, "total_infected": total,
                         "peak_infectious": int(curve.max()), "peak_step": int(curve.argmax())})
    return rows, {k: np.stack(v) for k, v in curves.items()}


# ---------------------------------------------------------------- RL (PPO) ---
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                                  # RL is optional
    gym = None


if gym is not None:
    class MetaPolicyEnv(gym.Env):
        """Each step the agent picks *which* targeting rule spends today's budget.

        Choosing among a few rules (instead of among all N nodes) keeps the
        action space small enough for PPO to learn in minutes. Observation:
        compartment fractions and progress through the horizon. Reward:
        minus the number of new infectious nodes.
        """
        ACTIONS = ("Degree", "PageRank", "Acquaintance", "Random")

        def __init__(self, cfg: Config, budget: int = 5, seed: int = 0):
            super().__init__()
            self.cfg, self.budget = cfg, budget
            self.action_space = spaces.Discrete(len(self.ACTIONS))
            self.observation_space = spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)
            self._rng = np.random.default_rng(seed)

        def _obs(self):
            s, n = self.state.states, len(self.state.states)
            fr = [np.mean(s == c) for c in range(5)]
            return np.array(fr + [self.t / len(self.graphs)], dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self.graphs = generate_dynamic_graphs(self.cfg.graph, self._rng)
            self.n = self.graphs[0].number_of_nodes()
            self.state = initial_state(self.n, self.cfg.epidemic, self._rng)
            self.t = 0
            return self._obs(), {}

        def step(self, action):
            G = self.graphs[self.t]
            for v in POLICIES[self.ACTIONS[int(action)]](G, self.state.states, self.budget, self._rng):
                self.state.states[v] = R
            before = is_infectious(self.state.states)
            self.state = step(self.state, adjacency(G, self.n), self.cfg.epidemic, self._rng)
            new_cases = int((is_infectious(self.state.states) & ~before).sum())
            self.t += 1
            done = self.t >= len(self.graphs)
            return self._obs(), -float(new_cases), done, False, {}
