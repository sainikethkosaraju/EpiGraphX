"""Print the README result tables (mean ± sd over seeds) from results/*.csv."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def pm(g, col, nd=3):
    m, s = g[col].mean(), g[col].std()
    return m.combine(s, lambda a, b: f"{a:.{nd}f} ± {b:.{nd}f}")


def table(df, cols):
    return df[cols].to_markdown(index=False)


def main_table():
    df = pd.read_csv(RES / "main_results.csv")
    g = df.groupby(["model", "kind"], sort=False)
    out = pd.DataFrame({"ROC-AUC": pm(g, "auc"), "AP": pm(g, "ap"),
                        "AUC (new)": pm(g, "auc_new"), "AP (new)": pm(g, "ap_new"),
                        "ECE": pm(g, "ece"), "ECE (temp-scaled)": pm(g, "ece_temp_scaled")}).reset_index()
    out["ECE (temp-scaled)"] = out["ECE (temp-scaled)"].str.replace("nan ± nan", "–")
    out = out.rename(columns={"model": "Model", "kind": "Type"})
    print(out.to_markdown(index=False))
    print(f"\nprevalence (test): {df.prevalence.mean():.3f}; seeds: {sorted(df.seed.unique())}")


def horizon_table():
    df = pd.read_csv(RES / "multi_horizon.csv")
    g = df.groupby(["model", "horizon"], sort=False)
    out = pd.DataFrame({"ROC-AUC": pm(g, "auc"), "AUC (new)": pm(g, "auc_new"),
                        "AP (new)": pm(g, "ap_new"), "RMSE": pm(g, "rmse")}).reset_index()
    print(out.pivot(index="model", columns="horizon", values="AUC (new)").to_markdown())
    print()
    print(out.pivot(index="model", columns="horizon", values="ROC-AUC").to_markdown())
    print()
    print(out.pivot(index="model", columns="horizon", values="RMSE").to_markdown())


def ablation_table():
    df = pd.read_csv(RES / "ablation.csv")
    g = df.groupby("dropped", sort=False)
    print(pd.DataFrame({"AP": pm(g, "ap"), "AP (new)": pm(g, "ap_new"),
                        "AUC (new)": pm(g, "auc_new")}).reset_index().to_markdown(index=False))


def topology_table():
    df = pd.read_csv(RES / "topology.csv")
    g = df.groupby(["topology", "model"], sort=False)
    print(pd.DataFrame({"AP": pm(g, "ap"), "AUC (new)": pm(g, "auc_new"),
                        "prevalence": pm(g, "prevalence")}).reset_index().to_markdown(index=False))


def interventions_table():
    df = pd.read_csv(RES / "interventions.csv")
    g = df.groupby("policy")
    base = g.total_infected.mean()["No intervention"]
    out = pd.DataFrame({"ever infected": pm(g, "total_infected", 1),
                        "peak infectious": pm(g, "peak_infectious", 1),
                        "reduction": (1 - g.total_infected.mean() / base).map(lambda x: f"{x:.0%}")})
    print(out.sort_values("ever infected").to_markdown())
    p = RES / "ppo.json"
    if p.exists():
        print(json.dumps(json.loads(p.read_text()), indent=1))


def calibration_info():
    c = json.loads((RES / "calibration.json").read_text())
    print({k: v for k, v in c.items() if k in ("temperature",)}, {k: v for k, v in c["mc_dropout"].items()
                                                                   if k != "std_hist"})


for fn in (main_table, horizon_table, ablation_table, topology_table, interventions_table, calibration_info):
    try:
        print(f"\n### {fn.__name__}\n")
        fn()
    except FileNotFoundError as e:
        print("missing:", e.filename)
