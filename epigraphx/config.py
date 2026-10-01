"""Experiment configuration.

Every knob that affects results lives here so a run is fully described by one
``Config`` object (it is also written to ``results/config.json``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class GraphConfig:
    n_nodes: int = 800
    n_steps: int = 60
    topology: str = "barabasi"      # barabasi | watts | erdos
    base_m: int = 3                 # BA attachment / half the WS ring degree
    rewire_frac: float = 0.05       # fraction of edges rewired per step (edge count stays constant)
    event_prob: float = 0.06        # probability of a super-spreader gathering per step
    event_size: int = 30            # people at a gathering
    event_p: float = 0.4            # contact probability between two attendees


@dataclass
class EpidemicConfig:
    beta: float = 0.045             # per-contact, per-step transmission probability
    mean_incubation: float = 3.0    # Poisson mean of the E period (steps)
    mean_infectious: float = 7.0    # Poisson mean of the I period (steps)
    gamma: float = 0.01             # extra per-step chance of early recovery
    asymp_prob: float = 0.25        # share of infections that are asymptomatic
    waning: float = 0.0004          # per-step chance R -> S (reinfection)
    initial_infected: int = 20


@dataclass
class TrainConfig:
    n_train_sims: int = 6           # independent epidemics used for training
    n_val_sims: int = 2             # ... for early stopping and temperature scaling
    n_test_sims: int = 2            # ... held out, only touched for final metrics
    window: int = 5                 # history length for temporal models
    horizons: tuple = (1, 3, 7, 14)
    hidden: int = 64
    dropout: float = 0.3
    lr: float = 2e-3
    weight_decay: float = 5e-4
    max_epochs: int = 20
    patience: int = 4
    mc_samples: int = 30


@dataclass
class Config:
    graph: GraphConfig = field(default_factory=GraphConfig)
    epidemic: EpidemicConfig = field(default_factory=EpidemicConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seeds: tuple = (0, 1, 2)

    def to_dict(self) -> dict:
        return asdict(self)


def quick_config() -> Config:
    """Tiny configuration for smoke tests and CI (runs in well under a minute)."""
    cfg = Config()
    cfg.graph.n_nodes = 150
    cfg.graph.n_steps = 30
    cfg.epidemic.initial_infected = 8
    cfg.train.n_train_sims, cfg.train.n_val_sims, cfg.train.n_test_sims = 2, 1, 1
    cfg.train.max_epochs = 3
    cfg.train.horizons = (1, 3)
    cfg.train.mc_samples = 5
    cfg.seeds = (0,)
    return cfg
