"""All experiments behind the README tables.

    python -m epigraphx.experiments                 # full run, all sections
    python -m epigraphx.experiments --quick         # tiny smoke run
    python -m epigraphx.experiments --only main horizons

Each section writes CSV/JSON into ``--out`` (default ``results/``); figures are
produced from those files by ``python -m epigraphx.plots``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import make_exposure, persistence, score_baseline
from .config import Config, quick_config
from .data import FEATURES, build_split
from .metrics import classification_metrics, fit_temperature, reliability_curve, sigmoid, ece
from .models import build_model
from .train import predict_logits, set_seed, train_model

STATIC = ["MLP", "GCN", "GraphSAGE", "GAT", "GraphTransformer"]
TEMPORAL = ["GCN-GRU", "Transformer-GRU"]


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fit_and_score(name, split, cfg, seed, n_out=1, in_dim=len(FEATURES)):
    set_seed(seed)
    model = build_model(name, in_dim, cfg.train.hidden, n_out, cfg.train.dropout)
    hist = train_model(model, split.train, split.val, cfg.train)
    z_val, y_val, _ = predict_logits(model, split.val, cfg.train)
    z, y, inf = predict_logits(model, split.test, cfg.train)
    return model, hist, (z_val, y_val), (z, y, inf)


def run_main(cfg: Config, out: Path):
    """Next-step infection forecasting: baselines vs static vs temporal models."""
    rows, calib = [], {}
    for seed in cfg.seeds:
        split, _ = build_split(cfg, seed)
        for bname, fn in [("Persistence", persistence), ("Exposure (true β)", make_exposure(cfg.epidemic.beta))]:
            p, y, inf = score_baseline(split.test, cfg.train.window, fn)
            rows.append({"model": bname, "kind": "baseline", "seed": seed,
                         **classification_metrics(y[:, 0], p[:, 0], inf)})
        for name in STATIC + TEMPORAL:
            t0 = time.time()
            model, hist, (zv, yv), (z, y, inf) = _fit_and_score(name, split, cfg, seed)
            T = fit_temperature(zv[:, 0], yv[:, 0])
            m = classification_metrics(y[:, 0], sigmoid(z[:, 0]), inf)
            m["ece_temp_scaled"] = ece(y[:, 0], sigmoid(z[:, 0] / T))
            rows.append({"model": name, "kind": "temporal" if name in TEMPORAL else "static",
                         "seed": seed, "epochs": len(hist), "temperature": T, **m})
            if seed == cfg.seeds[0] and name == "GraphSAGE":
                xs, ys, ns = reliability_curve(y[:, 0], sigmoid(z[:, 0]))
                xt, yt, nt = reliability_curve(y[:, 0], sigmoid(z[:, 0] / T))
                calib = {"raw": [xs.tolist(), ys.tolist()], "scaled": [xt.tolist(), yt.tolist()],
                         "temperature": T}
                # MC-dropout uncertainty for the same model
                zmc, _, _ = predict_logits(model, split.test, cfg.train, mc_samples=cfg.train.mc_samples)
                pm = sigmoid(zmc[..., 0])
                mean, std = pm.mean(0), pm.std(0)
                wrong = (mean > 0.5) != (y[:, 0] > 0.5)
                calib["mc_dropout"] = {
                    "ece_mc_mean": ece(y[:, 0], mean),
                    "mean_std_correct": float(std[~wrong].mean()),
                    "mean_std_wrong": float(std[wrong].mean()) if wrong.any() else None,
                    "std_hist": np.histogram(std, bins=30, range=(0, max(std.max(), 1e-6)))[0].tolist(),
                    "std_max": float(std.max()),
                }
            _log(f"main seed={seed} {name:16s} ap={m['ap']:.3f} ap_new={m['ap_new']:.3f} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "main_results.csv", index=False)
    (out / "calibration.json").write_text(json.dumps(calib, indent=2))
    return df


def run_horizons(cfg: Config, out: Path):
    """Multi-horizon forecasting with one multi-output ST-LSTM vs a static GNN."""
    H = list(cfg.train.horizons)
    rows = []
    for seed in cfg.seeds:
        split, _ = build_split(cfg, seed, horizons=H)
        preds = {}
        for bname, fn in [("Persistence", persistence), ("Exposure (true β)", make_exposure(cfg.epidemic.beta))]:
            preds[bname] = score_baseline(split.test, cfg.train.window, fn)
        for name in ["GraphSAGE", "ST-LSTM"]:
            _, _, _, (z, y, inf) = _fit_and_score(name, split, cfg, seed, n_out=len(H))
            preds[name] = (sigmoid(z), y, inf)
        for name, (p, y, inf) in preds.items():
            for j, h in enumerate(H):
                m = classification_metrics(y[:, j], p[:, j], inf)
                rows.append({"model": name, "seed": seed, "horizon": h,
                             "rmse": float(np.sqrt(m["brier"])), **m})
        _log(f"horizons seed={seed} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "multi_horizon.csv", index=False)
    return df


def run_ablation(cfg: Config, out: Path, model="GraphSAGE"):
    """Drop one input feature at a time."""
    rows = []
    for seed in cfg.seeds:
        _, eps = build_split(cfg, seed)
        for drop in [None, *FEATURES]:
            split, _ = build_split(cfg, seed, drop=drop, epidemics=eps)
            _, _, _, (z, y, inf) = _fit_and_score(model, split, cfg, seed,
                                                  in_dim=len(FEATURES) - (drop is not None))
            m = classification_metrics(y[:, 0], sigmoid(z[:, 0]), inf)
            rows.append({"dropped": drop or "none (all features)", "seed": seed, **m})
        _log(f"ablation seed={seed} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation.csv", index=False)
    return df


def run_topology(cfg: Config, out: Path):
    rows = []
    for topo in ["barabasi", "watts", "erdos"]:
        tcfg = Config(**{**cfg.__dict__})
        tcfg.graph = type(cfg.graph)(**{**cfg.graph.__dict__, "topology": topo})
        for seed in cfg.seeds:
            split, _ = build_split(tcfg, seed)
            p, y, inf = score_baseline(split.test, cfg.train.window, make_exposure(cfg.epidemic.beta))
            rows.append({"topology": topo, "model": "Exposure (true β)", "seed": seed,
                         **classification_metrics(y[:, 0], p[:, 0], inf)})
            _, _, _, (z, y, inf) = _fit_and_score("GraphSAGE", split, tcfg, seed)
            rows.append({"topology": topo, "model": "GraphSAGE", "seed": seed,
                         **classification_metrics(y[:, 0], sigmoid(z[:, 0]), inf)})
        _log(f"topology {topo} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "topology.csv", index=False)
    return df


def run_interventions(cfg: Config, out: Path, n_runs=30, budget=5):
    from .interventions import compare_policies
    rows, curves = compare_policies(cfg, budget=budget, n_runs=n_runs)
    pd.DataFrame(rows).to_csv(out / "interventions.csv", index=False)
    np.savez_compressed(out / "intervention_curves.npz", **curves)
    _log("interventions done")


def run_epidemic_curves(cfg: Config, out: Path, n_runs=20):
    from .graphs import generate_dynamic_graphs
    from .simulation import compartment_counts, simulate
    rows = []
    for r in range(n_runs):
        rng = np.random.default_rng([99, r])
        counts = compartment_counts(simulate(generate_dynamic_graphs(cfg.graph, rng), cfg.epidemic, rng))
        for t in range(cfg.graph.n_steps):
            rows.append({"run": r, "t": t, **{k: int(v[t]) for k, v in counts.items()}})
    pd.DataFrame(rows).to_csv(out / "epidemic_curves.csv", index=False)


SECTIONS = {"curves": run_epidemic_curves, "main": run_main, "horizons": run_horizons,
            "ablation": run_ablation, "topology": run_topology, "interventions": run_interventions}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny configuration for smoke testing")
    ap.add_argument("--only", nargs="*", choices=list(SECTIONS), help="run only these sections")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    cfg = quick_config() if args.quick else Config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=list))
    for name in args.only or SECTIONS:
        _log(f"=== {name} ===")
        kwargs = {"n_runs": 3} if args.quick and name in ("interventions", "curves") else {}
        SECTIONS[name](cfg, out, **kwargs)


if __name__ == "__main__":
    main()
