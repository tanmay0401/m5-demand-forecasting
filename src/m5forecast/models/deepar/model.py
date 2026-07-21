"""DeepARForecaster: the ForecastModel wrapper around dataset + network.

fit():   build dense arrays from the panel slice, train by likelihood
         maximization with per-epoch window resampling.
predict(): ancestral sampling (config n_samples paths x 28 days), point
         forecast = per-day sample MEDIAN (robust under skewed NegBin
         tails), quantiles stored on self.quantiles_ for Phase 13.

Compute honesty: trains on whatever device is available (RTX 4050 here);
series are processed in chunks at prediction to bound GPU memory
(chunk x samples x horizon tensors).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from m5forecast.models.base import ForecastModel
from m5forecast.models.deepar.dataset import TIME_FEATURES, WindowDataset, build_arrays
from m5forecast.models.deepar.network import DeepARNet
from m5forecast.utils.logging import get_logger

log = get_logger(__name__)


class DeepARForecaster(ForecastModel):
    name = "deepar"

    def __init__(self, params):
        self.p = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- training ----------------------------------------------------------
    def fit(self, history, features=None):
        if features is None:
            raise ValueError("deepar needs the feature table (for calendar covariates)")
        train_end = int(history["d"].max())
        cols = ["id", "item_id", "dept_id", "store_id", "d", "sales",
                "dow", "dom", "month", "snap", "is_event", "sell_price"]
        panel = features[features["d"] <= train_end + int(self.p.get("predict_horizon", 28))][cols]
        self.arrays = build_arrays(panel)
        # belt-and-braces leakage guard: the arrays contain test-day rows (for
        # their KNOWN covariates); blank their sales so no code path can ever
        # read a post-cutoff target, even by future bug
        self.arrays["sales"][:, train_end:] = 0.0
        self.train_end = train_end

        ds = WindowDataset(
            self.arrays,
            context=int(self.p.context_length),
            horizon=28,
            max_day=train_end,
            windows_per_series=int(self.p.windows_per_series_per_epoch),
            seed=42,
        )
        dl = DataLoader(ds, batch_size=int(self.p.batch_size), shuffle=True,
                        num_workers=0, drop_last=True)

        self.net = DeepARNet(
            n_items=int(self.arrays["cats"][:, 0].max()) + 1,
            n_depts=int(self.arrays["cats"][:, 1].max()) + 1,
            n_stores=int(self.arrays["cats"][:, 2].max()) + 1,
            cov_dim=TIME_FEATURES,
            hidden=int(self.p.hidden_size),
            layers=int(self.p.num_layers),
            dropout=float(self.p.dropout),
            embed_dim=int(self.p.embedding_dim),
        ).to(self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.p.lr))
        self.net.train()
        for epoch in range(int(self.p.epochs)):
            ds.resample()
            total, nb = 0.0, 0
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.net.loss(batch)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                opt.step()
                total, nb = total + float(loss), nb + 1
            log.info("deepar epoch %d/%d: nll=%.4f", epoch + 1, self.p.epochs, total / max(nb, 1))
        return self

    # -- prediction ----------------------------------------------------------
    def predict(self, future):
        quantile_levels = [float(q) for q in self.p.get("quantiles", [0.05, 0.165, 0.25, 0.5, 0.75, 0.835, 0.95])]
        n_samples = int(self.p.num_samples)
        context = int(self.p.context_length)
        horizon = int(future["d"].max() - self.train_end)

        a, T = self.arrays, self.train_end
        n_series = a["sales"].shape[0]
        self.net.eval()

        med = np.zeros((n_series, horizon), dtype="float32")
        qs = {q: np.zeros((n_series, horizon), dtype="float32") for q in quantile_levels}

        chunk = int(self.p.get("predict_chunk", 2048))
        for lo in range(0, n_series, chunk):
            hi = min(lo + chunk, n_series)
            past_sales = a["sales"][lo:hi, T - context : T]
            nu = 1.0 + past_sales.mean(axis=1)
            batch = {
                "past_scaled": torch.from_numpy(past_sales / nu[:, None]).to(self.device),
                "past_covs": torch.from_numpy(a["covs"][lo:hi, T - context : T]).to(self.device),
                "fut_covs": torch.from_numpy(a["covs"][lo:hi, T : T + horizon]).to(self.device),
                "cats": torch.from_numpy(a["cats"][lo:hi]).to(self.device),
                "nu": torch.from_numpy(nu.astype("float32")).to(self.device),
            }
            paths = self.net.sample_paths(
                batch["past_scaled"], batch["past_covs"], batch["fut_covs"],
                batch["cats"], batch["nu"], n_samples,
            )  # [B, S, H]
            med[lo:hi] = paths.quantile(0.5, dim=1).cpu().numpy()
            for q in quantile_levels:
                qs[q][lo:hi] = paths.quantile(q, dim=1).cpu().numpy()

        # map dense arrays back onto the requested (id, d) rows
        id_codes = pd.Categorical(future["id"], categories=a["ids"]).codes
        day_idx = future["d"].to_numpy() - (T + 1)
        yhat = med[id_codes, day_idx]

        self.quantiles_ = future[["id", "d"]].copy()
        for q in quantile_levels:
            self.quantiles_[f"q{q}"] = qs[q][id_codes, day_idx]

        return self._finalize(future, pd.Series(yhat, index=future.index))

    def predict_quantiles(self, future, quantiles):
        if not hasattr(self, "quantiles_"):
            self.predict(future)
        return self.quantiles_
