"""Dynamic contact networks.

A base network evolves by *rewiring* (an existing edge is removed and a new
random edge added, so the number of contacts stays constant), and each step may
include a temporary super-spreader gathering whose extra edges exist only in
that step's snapshot.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .config import GraphConfig


def _base_graph(cfg: GraphConfig, rng: np.random.Generator) -> nx.Graph:
    seed = int(rng.integers(2**31 - 1))
    n, m = cfg.n_nodes, cfg.base_m
    if cfg.topology == "barabasi":
        return nx.barabasi_albert_graph(n, m, seed=seed)
    if cfg.topology == "watts":
        return nx.watts_strogatz_graph(n, 2 * m, 0.1, seed=seed)
    if cfg.topology == "erdos":
        # match the BA mean degree (~2m) so topologies are comparable
        return nx.gnp_random_graph(n, 2 * m / (n - 1), seed=seed)
    raise ValueError(f"unknown topology: {cfg.topology}")


def generate_dynamic_graphs(cfg: GraphConfig, rng: np.random.Generator) -> list[nx.Graph]:
    """Return ``cfg.n_steps`` snapshots of an evolving contact network."""
    n = cfg.n_nodes
    base = _base_graph(cfg, rng)
    snapshots = []
    for t in range(cfg.n_steps):
        if t > 0:
            edges = list(base.edges())
            k = int(round(cfg.rewire_frac * len(edges)))
            for idx in rng.choice(len(edges), size=min(k, len(edges)), replace=False):
                base.remove_edge(*edges[idx])
                while True:
                    u, v = rng.integers(n, size=2)
                    if u != v and not base.has_edge(u, v):
                        base.add_edge(int(u), int(v))
                        break
        snap = base.copy()
        if rng.random() < cfg.event_prob:
            group = rng.choice(n, size=min(cfg.event_size, n), replace=False)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if rng.random() < cfg.event_p:
                        snap.add_edge(int(group[i]), int(group[j]))
        snapshots.append(snap)
    return snapshots
