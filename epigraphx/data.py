"""Turn simulated epidemics into graph-learning datasets.

Each snapshot at time ``t`` holds node features computed from the contact
network and *observable* infection status at ``t`` (only infectious nodes are
observable -- exposed nodes are hidden, as in a real outbreak), and labels
``infectious at t + h``.

Train / validation / test splits are **separate simulated epidemics**, never
different time steps of the same run, so no information about a test outbreak
leaks into training.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from .config import Config
from .graphs import generate_dynamic_graphs
from .simulation import is_infectious, simulate

FEATURES = ("degree", "clustering", "pagerank", "infected_now")


@dataclass
class Epidemic:
    graphs: list[nx.Graph]
    states: np.ndarray            # [T, N]


def make_epidemics(cfg: Config, n: int, rng: np.random.Generator) -> list[Epidemic]:
    out = []
    for _ in range(n):
        graphs = generate_dynamic_graphs(cfg.graph, rng)
        out.append(Epidemic(graphs, simulate(graphs, cfg.epidemic, rng)))
    return out


def node_features(G: nx.Graph, states_t: np.ndarray, drop: str | None = None) -> np.ndarray:
    n = G.number_of_nodes()
    deg = np.array([d for _, d in sorted(G.degree())], dtype=float)
    clust = np.array([c for _, c in sorted(nx.clustering(G).items())], dtype=float)
    pr = np.array([p for _, p in sorted(nx.pagerank(G).items())], dtype=float)
    cols = {
        "degree": deg / max(deg.max(), 1.0),
        "clustering": clust,
        "pagerank": pr / max(pr.max(), 1e-12),
        "infected_now": is_infectious(states_t).astype(float),
    }
    if drop is not None:
        cols.pop(drop)
    return np.stack(list(cols.values()), axis=1).astype(np.float32) if n else np.zeros((0, len(cols)))


def edge_index(G: nx.Graph) -> torch.Tensor:
    if G.number_of_edges() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    e = torch.tensor(list(G.edges()), dtype=torch.long).t()
    return torch.cat([e, e.flip(0)], dim=1)


def build_snapshots(ep: Epidemic, horizons=(1,), drop: str | None = None) -> list[Data]:
    """One ``Data`` per time step that has labels for every horizon.

    ``data.y`` has shape [N, len(horizons)]; ``data.infected_now`` marks nodes
    already infectious at ``t`` (used to score *new* infections separately).
    """
    T = len(ep.graphs)
    hmax = max(horizons)
    snaps = []
    for t in range(T - hmax):
        x = node_features(ep.graphs[t], ep.states[t], drop)
        y = np.stack([is_infectious(ep.states[t + h]) for h in horizons], axis=1)
        snaps.append(Data(
            x=torch.from_numpy(x),
            edge_index=edge_index(ep.graphs[t]),
            y=torch.from_numpy(y.astype(np.float32)),
            infected_now=torch.from_numpy(is_infectious(ep.states[t])),
            t=t,
        ))
    return snaps


def windows(snaps: list[Data], window: int) -> list[list[Data]]:
    """Sliding windows ending at (and including) each snapshot ``t``."""
    return [snaps[t - window + 1: t + 1] for t in range(window - 1, len(snaps))]


@dataclass
class Split:
    train: list
    val: list
    test: list


def build_split(cfg: Config, seed: int, horizons=(1,), drop: str | None = None,
                epidemics: tuple | None = None) -> tuple[Split, tuple]:
    """Simulate train/val/test epidemics for one seed and convert to snapshots.

    Returns the split (lists of per-epidemic snapshot lists) and the raw
    epidemics so callers can reuse them (e.g. for feature ablations).
    """
    if epidemics is None:
        rng = np.random.default_rng(seed)
        tc = cfg.train
        epidemics = (make_epidemics(cfg, tc.n_train_sims, rng),
                     make_epidemics(cfg, tc.n_val_sims, rng),
                     make_epidemics(cfg, tc.n_test_sims, rng))
    split = Split(*[[build_snapshots(ep, horizons, drop) for ep in group] for group in epidemics])
    return split, epidemics
