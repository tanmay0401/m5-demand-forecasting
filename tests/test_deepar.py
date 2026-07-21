"""DeepAR tests on synthetic data — small net, CPU, seconds not minutes.

The load-bearing test is test_learns_scale: two series with 10x different
levels must get 10x different forecasts THROUGH THE SAME network — that's
the per-series scale handling (nu) working end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf

from m5forecast.models.deepar.dataset import WindowDataset, build_arrays
from m5forecast.models.deepar.model import DeepARForecaster
from m5forecast.models.deepar.network import DeepARNet

N_DAYS = 200
TRAIN_END = 180

CFG = OmegaConf.create(
    {
        "name": "deepar",
        "likelihood": "negative_binomial",
        "context_length": 56,
        "hidden_size": 24,
        "num_layers": 1,
        "dropout": 0.0,
        "embedding_dim": 4,
        "num_samples": 80,
        "epochs": 4,
        "batch_size": 64,
        "lr": 5e-3,
        "windows_per_series_per_epoch": 8,
        "predict_chunk": 64,
        "quantiles": [0.25, 0.5, 0.75],
    }
)


def make_features(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    levels = {"ID_A": 20.0, "ID_B": 2.0, "ID_C": 8.0, "ID_D": 0.6}
    for sid, lvl in levels.items():
        for d in range(1, N_DAYS + 1):
            rows.append((sid, "IT_" + sid, "DEPT_0", "ST_0", d, rng.poisson(lvl),
                         d % 7, (d % 28) + 1, ((d // 28) % 12) + 1, 0, 0, 4.0))
    df = pd.DataFrame(rows, columns=["id", "item_id", "dept_id", "store_id", "d", "sales",
                                     "dow", "dom", "month", "snap", "is_event", "sell_price"])
    for c in ["id", "item_id", "dept_id", "store_id"]:
        df[c] = df[c].astype("category")
    df["sales"] = df["sales"].astype("int16")
    return df


@pytest.fixture(scope="module")
def fitted():
    torch.manual_seed(0)
    feats = make_features()
    history = feats[feats["d"] <= TRAIN_END][["id", "d", "sales"]]
    return DeepARForecaster(CFG).fit(history, feats), feats


def test_dataset_windows_respect_cutoff():
    feats = make_features()
    arrays = build_arrays(feats)
    ds = WindowDataset(arrays, context=56, horizon=28, max_day=TRAIN_END, windows_per_series=4, seed=0)
    assert (ds.starts + 28 <= TRAIN_END).all()  # no target beyond the cutoff
    item = ds[0]
    assert item["past_scaled"].shape == (56,)
    assert item["target"].shape == (28,)
    assert item["nu"] > 0


def test_negbin_head_mean_matches_mu():
    net = DeepARNet(n_items=4, n_depts=1, n_stores=1, cov_dim=6, hidden=8, layers=1, embed_dim=2)
    h = torch.randn(3, 5, 8)
    nu = torch.tensor([1.0, 4.0, 9.0])
    dist = net._distribution(h, nu)
    mu = torch.nn.functional.softplus(net.head_mu(h).squeeze(-1)) * nu.unsqueeze(-1) + 1e-6
    assert torch.allclose(dist.mean, mu, rtol=1e-4)


def test_training_reduces_loss():
    torch.manual_seed(1)
    feats = make_features()
    arrays = build_arrays(feats)
    # 4 series x 64 windows / batch 32 = 8 steps/epoch x 4 epochs = 32 steps —
    # enough optimization to fairly demand a 10% NLL drop (16 windows gave 6
    # steps total and the first version of this test failed for that reason)
    ds = WindowDataset(arrays, context=56, horizon=28, max_day=TRAIN_END, windows_per_series=64, seed=1)
    net = DeepARNet(n_items=4, n_depts=1, n_stores=1, cov_dim=6, hidden=24, layers=1, embed_dim=4)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=32, shuffle=True)
    # measure on a FIXED batch before vs after training: epoch-total comparisons
    # fail here because the model converges within the first epoch's 8 steps
    eval_batch = next(iter(DataLoader(ds, batch_size=64, shuffle=False)))
    with torch.no_grad():
        before = float(net.loss(eval_batch))
    for _ in range(4):
        for b in dl:
            loss = net.loss(b)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        after = float(net.loss(eval_batch))
    assert after < before * 0.9  # trained NLL >10% better than untrained


def test_predict_shapes_and_nonneg(fitted):
    model, feats = fitted
    future = feats[(feats["d"] > TRAIN_END) & (feats["d"] <= TRAIN_END + 28)][["id", "d"]]
    preds = model.predict(future)
    assert len(preds) == len(future)
    assert (preds["yhat"] >= 0).all()
    q = model.quantiles_
    assert {"q0.25", "q0.5", "q0.75"} <= set(q.columns)
    assert (q["q0.25"] <= q["q0.75"] + 1e-6).all()  # quantiles ordered


def test_learns_scale(fitted):
    """Series built 20/day vs 0.6/day must forecast ~proportionally via one shared net."""
    model, feats = fitted
    future = feats[(feats["d"] > TRAIN_END) & (feats["d"] <= TRAIN_END + 28)][["id", "d"]]
    preds = model.predict(future)
    mean_a = preds[preds["id"] == "ID_A"]["yhat"].mean()
    mean_d = preds[preds["id"] == "ID_D"]["yhat"].mean()
    assert mean_a > 10 * max(mean_d, 0.05)
    assert 10 < mean_a < 40  # in the right ballpark of 20/day