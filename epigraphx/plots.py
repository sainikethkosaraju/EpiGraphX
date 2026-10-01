"""Figures for the README, generated from the CSV/JSON files in ``results/``.

    python -m epigraphx.plots [--results results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red);
# baselines are drawn in neutral grey so color always means "a learned model".
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": MUTED,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe8", "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 2, "axes.axisbelow": True,
})


def epidemic_curves(res: Path, figs: Path):
    df = pd.read_csv(res / "epidemic_curves.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, col in enumerate(["susceptible", "exposed", "infectious", "recovered"]):
        g = df.groupby("t")[col]
        mean, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
        ax.fill_between(mean.index, lo, hi, color=SERIES[i], alpha=0.15, lw=0)
        ax.plot(mean.index, mean, color=SERIES[i], label=col.capitalize())
    ax.set_xlabel("time step")
    ax.set_ylabel("nodes")
    ax.set_title(f"SEIR++ on a dynamic network (mean and 10–90% band, {df.run.nunique()} runs)", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(figs / "epidemic_curves.png")
    plt.close(fig)


def model_comparison(res: Path, figs: Path):
    df = pd.read_csv(res / "main_results.csv")
    order = df.groupby("model")["ap"].mean().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, [("ap", "All nodes: average precision"),
                                       ("ap_new", "New infections only: average precision")]):
        g = df.groupby("model")[col]
        mean, std = g.mean().reindex(order), g.std().reindex(order).fillna(0)
        kinds = df.drop_duplicates("model").set_index("model")["kind"].reindex(order)
        colors = [GREY if k == "baseline" else (SERIES[0] if k == "static" else SERIES[1]) for k in kinds]
        ax.barh(range(len(order)), mean, xerr=std, color=colors, height=0.62,
                error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        for y, (m, s) in enumerate(zip(mean, std)):
            ax.text(m + s + mean.max() * 0.02, y, f"{m:.3f}", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, mean.max() * 1.25 + std.max())
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), order)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, SERIES[0], SERIES[1])]
    fig.legend(handles, ["Baseline (no training)", "Static GNN / MLP", "Temporal GNN"],
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Next-step infection forecasting on held-out epidemics "
                 f"(mean ± sd over {df.seed.nunique()} seeds)", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / "model_comparison.png")
    plt.close(fig)


def multi_horizon(res: Path, figs: Path):
    df = pd.read_csv(res / "multi_horizon.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, (col, title) in zip(axes, [("auc", "ROC-AUC, all nodes"), ("auc_new", "ROC-AUC, new infections")]):
        for i, model in enumerate(["ST-LSTM", "GraphSAGE", "Exposure (true β)", "Persistence"]):
            g = df[df.model == model].groupby("horizon")[col]
            color = SERIES[[1, 0][i]] if i < 2 else GREY
            style = "-" if i < 2 else ("--" if i == 2 else ":")
            ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), color=color, ls=style, marker="o",
                        ms=5, capsize=2, label=model)
        ax.set_xticks(sorted(df.horizon.unique()))
        ax.set_xlabel("forecast horizon (steps ahead)")
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "multi_horizon.png")
    plt.close(fig)


def calibration(res: Path, figs: Path):
    c = json.loads((res / "calibration.json").read_text())
    if not c:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="Perfect calibration")
    ax.plot(*c["raw"], marker="o", color=SERIES[0], label="GraphSAGE, raw")
    ax.plot(*c["scaled"], marker="s", color=SERIES[1], label=f"Temperature-scaled (T={c['temperature']:.2f})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram (test)", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1]
    mc = c["mc_dropout"]
    edges = np.linspace(0, mc["std_max"], len(mc["std_hist"]) + 1)
    ax.bar(edges[:-1], mc["std_hist"], width=np.diff(edges), align="edge", color=SERIES[0], edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("MC-dropout std of predicted probability")
    ax.set_ylabel("nodes (log scale)")
    ax.set_title("Predictive uncertainty (test)", loc="left")
    fig.tight_layout()
    fig.savefig(figs / "calibration.png")
    plt.close(fig)


def interventions(res: Path, figs: Path):
    curves = np.load(res / "intervention_curves.npz")
    df = pd.read_csv(res / "interventions.csv")
    order = ["No intervention", "Random", "Acquaintance", "PageRank", "Degree"]
    colors = dict(zip(order, [GREY, SERIES[3], SERIES[2], SERIES[0], SERIES[1]]))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for name in order:
        c = curves[name]
        m = c.mean(0)
        se = c.std(0) / np.sqrt(len(c))
        axes[0].fill_between(range(len(m)), m - 1.96 * se, m + 1.96 * se, color=colors[name], alpha=0.15, lw=0)
        axes[0].plot(m, color=colors[name], label=name, ls="--" if name == "No intervention" else "-")
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("infectious nodes")
    axes[0].set_title("Infectious over time (mean, 95% CI)", loc="left")
    axes[0].legend(fontsize=8)
    g = df.groupby("policy")["total_infected"]
    mean, se = g.mean().reindex(order), (g.std() / np.sqrt(g.count())).reindex(order)
    axes[1].barh(range(len(order)), mean, xerr=1.96 * se, color=[colors[o] for o in order], height=0.62,
                 error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
    for y, m in enumerate(mean):
        axes[1].text(m * 0.98, y, f"{m:.0f}", va="center", ha="right", fontsize=8, color="white",
                     fontweight="semibold")
    axes[1].set_yticks(range(len(order)), order)
    axes[1].grid(axis="y", visible=False)
    axes[1].set_xlabel("people ever infected")
    axes[1].set_title(f"Outbreak size ({df.run.nunique()} runs)", loc="left")
    fig.tight_layout()
    fig.savefig(figs / "interventions.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    res = Path(args.results)
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for fn, needs in [(epidemic_curves, "epidemic_curves.csv"), (model_comparison, "main_results.csv"),
                      (multi_horizon, "multi_horizon.csv"), (calibration, "calibration.json"),
                      (interventions, "interventions.csv")]:
        if (res / needs).exists():
            fn(res, figs)
            print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
