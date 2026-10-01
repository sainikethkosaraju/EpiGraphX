import numpy as np
import pytest
import torch

from epigraphx.config import quick_config
from epigraphx.data import FEATURES, build_split, windows
from epigraphx.graphs import generate_dynamic_graphs
from epigraphx.interventions import POLICIES, degree_policy, run_policy
from epigraphx.metrics import ece, fit_temperature
from epigraphx.models import build_model
from epigraphx.simulation import E, I_ASYM, I_SYM, R, S, initial_state, is_infectious, simulate


@pytest.fixture(scope="module")
def cfg():
    return quick_config()


def test_rewiring_keeps_edge_count(cfg):
    cfg.graph.event_prob = 0.0
    graphs = generate_dynamic_graphs(cfg.graph, np.random.default_rng(0))
    counts = {g.number_of_edges() for g in graphs}
    cfg.graph.event_prob = 0.06
    assert len(counts) == 1, "rewiring must not densify the network"


def test_seed_cases_are_infectious_for_more_than_one_step(cfg):
    st = initial_state(cfg.graph.n_nodes, cfg.epidemic, np.random.default_rng(0))
    assert is_infectious(st.states).sum() == cfg.epidemic.initial_infected
    assert (st.timer[is_infectious(st.states)] >= 1).all()


def test_simulation_states_and_conservation(cfg):
    rng = np.random.default_rng(1)
    graphs = generate_dynamic_graphs(cfg.graph, rng)
    states = simulate(graphs, cfg.epidemic, rng)
    assert states.shape == (cfg.graph.n_steps, cfg.graph.n_nodes)
    assert set(np.unique(states)) <= {S, E, I_SYM, I_ASYM, R}
    # S -> E is the only way in; nobody jumps straight from S to I
    jumped = (states[:-1] == S) & is_infectious(states[1:])
    assert not jumped.any()


def test_zero_beta_means_no_new_infections(cfg):
    cfg.epidemic.beta, old = 0.0, cfg.epidemic.beta
    rng = np.random.default_rng(2)
    states = simulate(generate_dynamic_graphs(cfg.graph, rng), cfg.epidemic, rng)
    cfg.epidemic.beta = old
    assert (states == E).sum() == 0


def test_split_uses_separate_epidemics(cfg):
    split, eps = build_split(cfg, seed=0, horizons=(1, 3))
    assert len(split.train) == cfg.train.n_train_sims
    assert len(split.test) == cfg.train.n_test_sims
    s = split.train[0][0]
    assert s.x.shape[1] == len(FEATURES) and s.y.shape[1] == 2
    w = windows(split.train[0], cfg.train.window)
    assert all(len(x) == cfg.train.window for x in w)
    # the window must end at the prediction time, not one step before it
    assert w[0][-1].t == cfg.train.window - 1


@pytest.mark.parametrize("name", ["MLP", "GCN", "GraphSAGE", "GAT", "GraphTransformer",
                                  "GCN-GRU", "SAGE-GRU", "Transformer-GRU", "ST-LSTM"])
def test_models_output_shape(cfg, name):
    split, _ = build_split(cfg, seed=0)
    m = build_model(name, len(FEATURES), 16, 3, 0.1)
    snaps = split.train[0][:cfg.train.window]
    if m.temporal:
        out = m([s.x for s in snaps], [s.edge_index for s in snaps])
    else:
        out = m(snaps[0].x, snaps[0].edge_index)
    assert out.shape == (cfg.graph.n_nodes, 3)
    assert torch.isfinite(out).all()


def test_ece_and_temperature():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(float)
    assert ece(y, p) < 0.05                               # calibrated by construction
    z = np.log(p / (1 - p)) * 3.0                         # overconfident logits
    assert abs(fit_temperature(z, y) - 3.0) < 0.5


def test_policies_only_vaccinate_susceptibles(cfg):
    graphs = generate_dynamic_graphs(cfg.graph, np.random.default_rng(3))
    states = np.full(cfg.graph.n_nodes, R, dtype=np.int8)
    states[:10] = S
    chosen = degree_policy(graphs[0], states, 5, np.random.default_rng(0))
    assert len(chosen) == 5 and all(states[v] == S for v in chosen)
    for pol in POLICIES.values():
        curve, total = run_policy(pol, graphs, cfg, 3, np.random.default_rng(0))
        assert len(curve) == len(graphs) and total >= cfg.epidemic.initial_infected


def test_rl_environment_contract(cfg):
    gym = pytest.importorskip("gymnasium")
    from epigraphx.interventions import MetaPolicyEnv
    env = MetaPolicyEnv(cfg, budget=3, seed=0)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    done, steps = False, 0
    while not done:
        obs, r, done, trunc, _ = env.step(env.action_space.sample())
        assert r <= 0
        steps += 1
    assert steps == cfg.graph.n_steps
