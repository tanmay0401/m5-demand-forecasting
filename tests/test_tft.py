"""TFT-style network tests on synthetic data (small, CPU, seconds)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf

from m5forecast.models.tft.model import TFTForecaster
from m5forecast.models.tft.network import GRN, TFTNet

# reuse the DeepAR synthetic generator
from tests.test_deepar import TRAIN_END, make_features

QUANTILES = [0.1, 0.5, 0.9]
CFG = OmegaConf.create(
    {
        "name": "tft",
        "context_length": 56,
        "hidden_size": 24,
        "attention_heads": 2,
        "dropout": 0.0,
        "embedding_dim": 4,
        "quantiles": QUANTILES,
        "epochs": 5,
        "batch_size": 64,
        "lr": 1e-2,
        "windows_per_series_per_epoch": 8,
        "predict_chunk": 64,
    }
)


def test_grn_shapes_and_residual():
    grn = GRN(8, 16, 8)
    x = torch.randn(3, 5, 8)
    assert grn(x).shape == (3, 5, 8)
    grn_ctx = GRN(8, 16, 12, ctx_dim=4)
    assert grn_ctx(x, torch.randn(3, 4)).shape == (3, 5, 12)


def test_attention_mask_is_causal_over_decoder():
    net = TFTNet(4, 1, 1, cov_dim=6, context_len=10, horizon=4, quantiles=QUANTILES, hidden=8, heads=2)
    m = net._attn_mask(10, 4, torch.device("cpu"))  # [H, C+H], False = allowed
    # first decoder query (i=0, abs time 10) attends to keys 0..10, not 11..13
    assert (~m[0, :11]).all() and m[0, 11:].all()
    # last decoder query (i=3, abs time 13) attends to everything
    assert (~m[3]).all()


def test_forward_and_attention_shapes():
    net = TFTNet(4, 1, 1, cov_dim=6, context_len=56, horizon=28, quantiles=QUANTILES, hidden=16, heads=2)
    b = 5
    preds, attn = net(
        torch.randn(b, 56), torch.randn(b, 56, 6), torch.randn(b, 28, 6),
        torch.zeros(b, 3, dtype=torch.long), return_attn=True,
    )
    assert preds.shape == (b, 28, 3)          # [B, H, Q]
    assert attn.shape == (b, 28, 56 + 28)     # [B, queries, keys]


def test_pinball_loss_positive_and_zero_at_perfect():
    net = TFTNet(4, 1, 1, cov_dim=6, context_len=8, horizon=4, quantiles=[0.5], hidden=8, heads=2)
    target = torch.randn(3, 4)
    perfect = target.unsqueeze(-1)  # [B,H,1], predicting the exact value
    assert torch.allclose(net.quantile_loss(perfect, target), torch.tensor(0.0), atol=1e-6)
    worse = perfect + 1.0
    assert net.quantile_loss(worse, target) > 0


@pytest.fixture(scope="module")
def fitted():
    torch.manual_seed(0)
    feats = make_features()
    history = feats[feats["d"] <= TRAIN_END][["id", "d", "sales"]]
    return TFTForecaster(CFG).fit(history, feats), feats


def test_predict_quantiles_ordered_and_nonneg(fitted):
    model, feats = fitted
    future = feats[(feats["d"] > TRAIN_END) & (feats["d"] <= TRAIN_END + 28)][["id", "d"]]
    preds = model.predict(future)
    assert (preds["yhat"] >= 0).all()
    q = model.quantiles_
    assert (q["q0.1"] <= q["q0.5"] + 1e-6).all()
    assert (q["q0.5"] <= q["q0.9"] + 1e-6).all()
    assert model.attention_.shape == (56,)
    # attention over the 56 context days holds most (but not all) mass — the
    # remainder is decoder self-attention over the horizon positions
    assert 0.5 < model.attention_.sum() <= 1.01


def test_learns_scale(fitted):
    model, feats = fitted
    future = feats[(feats["d"] > TRAIN_END) & (feats["d"] <= TRAIN_END + 28)][["id", "d"]]
    preds = model.predict(future)
    mean_a = preds[preds["id"] == "ID_A"]["yhat"].mean()  # built ~20/day
    mean_d = preds[preds["id"] == "ID_D"]["yhat"].mean()  # built ~0.6/day
    assert mean_a > 5 * max(mean_d, 0.05)
