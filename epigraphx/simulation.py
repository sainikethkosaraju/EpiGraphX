"""SEIR++ stochastic simulator on dynamic graphs.

Compartments: S (susceptible), E (exposed), I_s / I_a (symptomatic /
asymptomatic infectious), R (recovered, may wane back to S).

A susceptible node with ``k`` infectious neighbours becomes exposed with
probability ``1 - (1 - beta)^k`` -- the closed form of trying each infectious
contact independently. Incubation and infectious periods are Poisson draws.

The same ``step`` function drives the simulator, the intervention
experiments and the RL environment, so every part of the project shares one
definition of the disease dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .config import EpidemicConfig

S, E, I_SYM, I_ASYM, R = 0, 1, 2, 3, 4
COMPARTMENTS = ("S", "E", "I_sym", "I_asym", "R")


def is_infectious(states: np.ndarray) -> np.ndarray:
    return (states == I_SYM) | (states == I_ASYM)


@dataclass
class EpidemicState:
    states: np.ndarray           # int8 compartment per node
    timer: np.ndarray            # steps left in the current E or I period

    def copy(self) -> "EpidemicState":
        return EpidemicState(self.states.copy(), self.timer.copy())


def adjacency(G: nx.Graph, n: int) -> sp.csr_matrix:
    if G.number_of_edges() == 0:
        return sp.csr_matrix((n, n))
    e = np.asarray(G.edges(), dtype=np.int64)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def initial_state(n: int, cfg: EpidemicConfig, rng: np.random.Generator) -> EpidemicState:
    states = np.full(n, S, dtype=np.int8)
    timer = np.zeros(n, dtype=np.int32)
    seeds = rng.choice(n, size=cfg.initial_infected, replace=False)
    asymp = rng.random(len(seeds)) < cfg.asymp_prob
    states[seeds] = np.where(asymp, I_ASYM, I_SYM)
    # Seed cases need a real infectious period; without it they recover
    # immediately and the outbreak never starts (a bug in the first version).
    timer[seeds] = np.maximum(1, rng.poisson(cfg.mean_infectious, len(seeds)))
    return EpidemicState(states, timer)


def step(state: EpidemicState, A: sp.csr_matrix, cfg: EpidemicConfig,
         rng: np.random.Generator) -> EpidemicState:
    """Advance the epidemic by one time step on contact matrix ``A``."""
    s, timer = state.states, state.timer
    new_s, new_t = s.copy(), timer.copy()
    n = len(s)

    # S -> E
    k_inf = A @ is_infectious(s).astype(float)
    p_inf = 1.0 - np.power(1.0 - cfg.beta, k_inf)
    newly_exposed = (s == S) & (rng.random(n) < p_inf)
    new_s[newly_exposed] = E
    new_t[newly_exposed] = np.maximum(1, rng.poisson(cfg.mean_incubation, newly_exposed.sum()))

    # E -> I
    exposed = s == E
    new_t[exposed] -= 1
    onset = exposed & (new_t <= 0)
    asymp = rng.random(n) < cfg.asymp_prob
    new_s[onset & asymp] = I_ASYM
    new_s[onset & ~asymp] = I_SYM
    new_t[onset] = np.maximum(1, rng.poisson(cfg.mean_infectious, onset.sum()))

    # I -> R
    inf = is_infectious(s)
    new_t[inf] -= 1
    recover = inf & ((new_t <= 0) | (rng.random(n) < cfg.gamma))
    new_s[recover] = R
    new_t[recover] = 0

    # R -> S (waning immunity)
    wane = (s == R) & (rng.random(n) < cfg.waning)
    new_s[wane] = S

    return EpidemicState(new_s, new_t)


def simulate(graphs: list[nx.Graph], cfg: EpidemicConfig,
             rng: np.random.Generator) -> np.ndarray:
    """Run SEIR++ over the snapshot sequence; returns states of shape [T, N]."""
    n = graphs[0].number_of_nodes()
    state = initial_state(n, cfg, rng)
    out = np.empty((len(graphs), n), dtype=np.int8)
    out[0] = state.states
    for t in range(len(graphs) - 1):
        state = step(state, adjacency(graphs[t], n), cfg, rng)
        out[t + 1] = state.states
    return out


def compartment_counts(states: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "susceptible": (states == S).sum(1),
        "exposed": (states == E).sum(1),
        "infectious": is_infectious(states).sum(1),
        "recovered": (states == R).sum(1),
    }
