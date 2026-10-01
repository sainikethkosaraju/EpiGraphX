"""Training, prediction and MC-dropout uncertainty for static and temporal models."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .config import TrainConfig
from .data import windows
from .metrics import average_precision_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _samples(model, epidemics, window):
    """Training units: snapshots for static models, windows for temporal ones."""
    if model.temporal:
        return [w for snaps in epidemics for w in windows(snaps, window)]
    return [s for snaps in epidemics for s in snaps]


def _forward(model, batch_items):
    """Forward a list of samples as one disjoint-union batch; returns logits and the last snapshot batch."""
    if model.temporal:
        steps = [Batch.from_data_list([w[i] for w in batch_items]) for i in range(len(batch_items[0]))]
        xs = [b.x.to(DEVICE) for b in steps]
        es = [b.edge_index.to(DEVICE) for b in steps]
        return model(xs, es), steps[-1]
    b = Batch.from_data_list(batch_items)
    return model(b.x.to(DEVICE), b.edge_index.to(DEVICE)), b


@torch.no_grad()
def predict_logits(model, epidemics, cfg: TrainConfig, mc_samples: int = 0):
    """Logits [M, H] (or MC-dropout logits [S, M, H]) plus labels and infected-now mask.

    Static models are scored on the same time steps as temporal ones (the
    first ``window - 1`` steps are skipped) so every model is evaluated on an
    identical set of nodes.
    """
    model.eval()
    if mc_samples:
        model.train()                          # keep dropout active
    outs, ys, inf = [], [], []
    for snaps in epidemics:
        items = windows(snaps, cfg.window) if model.temporal else snaps[cfg.window - 1:]
        for i in range(0, len(items), 8):
            chunk = items[i:i + 8]
            reps = []
            for _ in range(max(mc_samples, 1)):
                logits, last = _forward(model, chunk)
                reps.append(logits.cpu().numpy())
            outs.append(np.stack(reps) if mc_samples else reps[0])
            ys.append(last.y.numpy())
            inf.append(last.infected_now.numpy())
    model.eval()
    axis = 1 if mc_samples else 0
    return np.concatenate(outs, axis=axis), np.concatenate(ys), np.concatenate(inf)


def train_model(model, train_eps, val_eps, cfg: TrainConfig, batch_size: int = 4, verbose=False):
    """Adam + BCE-with-logits, early stopping on validation average precision."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    samples = _samples(model, train_eps, cfg.window)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(cfg.max_epochs):
        model.train()
        random.shuffle(samples)
        losses = []
        for i in range(0, len(samples), batch_size):
            logits, b = _forward(model, samples[i:i + batch_size])
            loss = F.binary_cross_entropy_with_logits(logits, b.y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        z, y, _ = predict_logits(model, val_eps, cfg)
        val_ap = float(np.nanmean([average_precision_score(y[:, j], z[:, j]) for j in range(y.shape[1])]))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_ap": val_ap})
        if verbose:
            print(f"  epoch {epoch + 1:2d} loss={np.mean(losses):.4f} val_ap={val_ap:.4f}")
        if val_ap > best + 1e-4:
            best, best_state, bad = val_ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return history
