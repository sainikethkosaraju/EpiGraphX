"""Train a PPO meta-policy that chooses which vaccination rule to use each step,
then compare it with the fixed rules on the same held-out epidemics.

    python scripts/train_ppo.py --timesteps 30000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from epigraphx.config import Config
from epigraphx.graphs import generate_dynamic_graphs
from epigraphx.interventions import POLICIES, MetaPolicyEnv, run_policy


def evaluate(model, cfg, budget, n_runs, seed=1000):
    """Mean total infections of PPO and each fixed rule on identical runs."""
    totals = {"PPO meta-policy": [], **{k: [] for k in POLICIES}}
    for r in range(n_runs):
        env = MetaPolicyEnv(cfg, budget=budget)
        obs, _ = env.reset(seed=seed + r)
        graphs = env.graphs
        ever, done = None, False
        from epigraphx.simulation import is_infectious
        ever = is_infectious(env.state.states).copy()
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            ever |= is_infectious(env.state.states)
        totals["PPO meta-policy"].append(int(ever.sum()))
        for name, pol in POLICIES.items():
            _, total = run_policy(pol, graphs, cfg, budget, np.random.default_rng([seed, r]))
            totals[name].append(total)
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=30_000)
    ap.add_argument("--budget", type=int, default=5)
    ap.add_argument("--eval-runs", type=int, default=20)
    ap.add_argument("--nodes", type=int, default=400, help="smaller graphs keep training fast")
    ap.add_argument("--out", default="results/ppo.json")
    args = ap.parse_args()

    cfg = Config()
    cfg.graph.n_nodes = args.nodes
    cfg.epidemic.initial_infected = max(5, args.nodes // 40)
    env = MetaPolicyEnv(cfg, budget=args.budget, seed=0)
    model = PPO("MlpPolicy", env, n_steps=cfg.graph.n_steps * 8, batch_size=60, seed=0, verbose=0)
    model.learn(total_timesteps=args.timesteps)
    res = evaluate(model, cfg, args.budget, args.eval_runs)
    res["_setup"] = {"timesteps": args.timesteps, "nodes": args.nodes, "budget": args.budget,
                     "eval_runs": args.eval_runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        if not k.startswith("_"):
            print(f"{k:18s} {v['mean']:7.1f} ± {v['std']:.1f} total infected")


if __name__ == "__main__":
    main()
