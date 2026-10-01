"""Non-learned reference predictors, scored on exactly the same nodes as the models."""
from __future__ import annotations

import numpy as np
import torch

from .data import windows


def _scores(snaps, window, score_fn):
    ps, ys, inf = [], [], []
    for s in snaps[window - 1:]:
        ps.append(score_fn(s))
        ys.append(s.y.numpy())
        inf.append(s.infected_now.numpy())
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(inf)


def persistence(s) -> np.ndarray:
    """"Whoever is infectious now will still be infectious later." """
    p = s.infected_now.numpy().astype(float)
    return np.repeat(p[:, None], s.y.shape[1], axis=1)


def make_exposure(beta: float):
    """Probability of catching it from today's infectious contacts: 1 - (1 - beta)^k.

    Infectious nodes score 1. Uses the true transmission rate, so it is a
    strong mechanistic reference rather than a strawman.
    """
    def exposure(s) -> np.ndarray:
        inf = s.infected_now.float()
        k = torch.zeros_like(inf).index_add_(0, s.edge_index[1], inf[s.edge_index[0]])
        p = np.where(s.infected_now.numpy(), 1.0, 1.0 - (1.0 - beta) ** k.numpy())
        return np.repeat(p[:, None], s.y.shape[1], axis=1)
    return exposure


def score_baseline(epidemics, window, score_fn):
    outs = [_scores(snaps, window, score_fn) for snaps in epidemics]
    return tuple(np.concatenate(parts) for parts in zip(*outs))


__all__ = ["persistence", "make_exposure", "score_baseline", "windows"]
