"""Evaluation metrics, including calibration and temperature scaling."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe(fn, y, p):
    return float(fn(y, p)) if 0 < y.sum() < len(y) else float("nan")


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_curve(y, p, n_bins=10):
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.sum() >= 20:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def classification_metrics(y: np.ndarray, p: np.ndarray, infected_now: np.ndarray) -> dict:
    """``auc``/``ap`` over all nodes and ``*_new`` over nodes not yet infectious.

    The ``*_new`` scores isolate the hard part of the task -- predicting who
    *becomes* infectious -- which a "still infected tomorrow" rule cannot do.
    """
    new = ~infected_now
    return {
        "auc": _safe(roc_auc_score, y, p),
        "ap": _safe(average_precision_score, y, p),
        "auc_new": _safe(roc_auc_score, y[new], p[new]),
        "ap_new": _safe(average_precision_score, y[new], p[new]),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
        "prevalence": float(y.mean()),
    }


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single temperature minimising validation NLL (Guo et al., 2017)."""
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / log_T.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.detach().exp())


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
