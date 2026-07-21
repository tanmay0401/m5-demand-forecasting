"""Windowed dataset for DeepAR-style training.

Each training example is one window of one series:
    context (84 days of known past)  +  horizon (28 days to predict)

Per the DeepAR paper's two scale tricks (its §3.3, the part most
reimplementations skip):
  1. **Input scaling**: the network sees sales / nu, where
     nu = 1 + mean(context sales). A 500-units/day series and a
     0.2-units/day series then look alike to the shared LSTM; the raw
     scale re-enters through the likelihood head (network.py).
  2. **Scale-weighted sampling**: windows are drawn with probability
     proportional to nu, so the few high-volume series (which dominate
     revenue and WRMSSE weight) are seen more often than a uniform draw
     over 30,490 mostly-quiet series would allow.

Covariates per timestep (all *known in the future*, Phase 7's taxonomy):
    dow/7, dom/31, month/12, snap, is_event, log-price change vs context mean
Static per series: item_id, dept_id, store_id codes -> embeddings.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

TIME_FEATURES = 6


def build_arrays(panel, id_col="id"):
    """Panel (sorted id, d) -> dense arrays keyed by series index.

    Returns dict with:
      sales   float32 [n_series, n_days]
      covs    float32 [n_series, n_days, TIME_FEATURES]
      cats    int64   [n_series, 3]  (item, dept, store codes)
      ids     the id order (categorical categories preserved)
    """
    ids = panel[id_col].cat.categories
    n_series, n_days = len(ids), panel["d"].max()

    sales = np.zeros((n_series, n_days), dtype="float32")
    covs = np.zeros((n_series, n_days, TIME_FEATURES), dtype="float32")

    sidx = panel[id_col].cat.codes.to_numpy()
    didx = panel["d"].to_numpy() - 1
    sales[sidx, didx] = panel["sales"].to_numpy(dtype="float32")

    covs[sidx, didx, 0] = panel["dow"].to_numpy(dtype="float32") / 7.0
    covs[sidx, didx, 1] = panel["dom"].to_numpy(dtype="float32") / 31.0
    covs[sidx, didx, 2] = panel["month"].to_numpy(dtype="float32") / 12.0
    covs[sidx, didx, 3] = panel["snap"].to_numpy(dtype="float32")
    covs[sidx, didx, 4] = panel["is_event"].to_numpy(dtype="float32")
    price = panel["sell_price"].fillna(0.0).to_numpy(dtype="float32")
    covs[sidx, didx, 5] = np.log1p(price)

    first = panel.groupby(id_col, observed=True).first()
    cats = np.stack(
        [
            first["item_id"].cat.codes.to_numpy(),
            first["dept_id"].cat.codes.to_numpy(),
            first["store_id"].cat.codes.to_numpy(),
        ],
        axis=1,
    ).astype("int64")
    return {"sales": sales, "covs": covs, "cats": cats, "ids": ids}


class WindowDataset(Dataset):
    """Randomly positioned training windows, resampled every epoch.

    `resample()` draws `windows_per_series` window starts per series, with
    series repetition probability proportional to nu (scale-weighted).
    """

    def __init__(self, arrays, context: int, horizon: int, max_day: int, windows_per_series: int, seed: int):
        self.a = arrays
        self.context, self.horizon = context, horizon
        self.max_day = max_day  # last usable target day (train_end)
        self.wps = windows_per_series
        self.rng = np.random.default_rng(seed)
        n = arrays["sales"].shape[0]
        # series weight ~ average volume in the training region
        vol = arrays["sales"][:, : self.max_day].mean(axis=1) + 0.01
        self.weights = vol / vol.sum()
        self.n_series = n
        self.resample()

    def resample(self):
        total = self.n_series * self.wps
        self.series_idx = self.rng.choice(self.n_series, size=total, p=self.weights)
        lo = self.context
        hi = self.max_day - self.horizon
        self.starts = self.rng.integers(lo, hi, size=total)  # index of first target day

    def __len__(self):
        return len(self.series_idx)

    def __getitem__(self, i):
        s, t0 = int(self.series_idx[i]), int(self.starts[i])
        past = slice(t0 - self.context, t0)
        fut = slice(t0, t0 + self.horizon)

        past_sales = self.a["sales"][s, past]
        nu = 1.0 + past_sales.mean()

        return {
            "past_scaled": torch.from_numpy(past_sales / nu),
            "past_covs": torch.from_numpy(self.a["covs"][s, past]),
            "fut_covs": torch.from_numpy(self.a["covs"][s, fut]),
            "target": torch.from_numpy(self.a["sales"][s, fut].copy()),
            "cats": torch.from_numpy(self.a["cats"][s]),
            "nu": torch.tensor(nu, dtype=torch.float32),
        }
