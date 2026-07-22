"""TFTForecaster: ForecastModel wrapper around the TFT-style network.

Same interface, dataset, and fold loop as DeepAR (Phase 10) — reuses
build_arrays / WindowDataset unchanged. Differences are internal:
  - trains on pinball loss over scaled targets (not likelihood NLL)
  - predicts DIRECTLY: one forward pass emits all 28 days x 7 quantiles;
    no sampling, so it is much faster at inference than DeepAR
  - point forecast = the q=0.5 output column
  - quantiles are sorted at predict time to guarantee monotonicity (the
    independent linear heads can otherwise cross)
  - exposes mean attention over context positions for the Phase 11 figure
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from m5forecast.models.base import ForecastModel
from m5forecast.models.deepar.dataset import TIME_FEATURES, WindowDataset, build_arrays
from m5forecast.models.tft.network import TFTNet
from m5forecast.utils.logging import get_logger

log = get_logger(__name__)


class TFTForecaster(ForecastModel):
    name = "tft"

    def __init__(self, params):
        self.p = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantile_levels = [float(q) for q in params.get("quantiles", [0.05, 0.165, 0.25, 0.5, 0.75, 0.835, 0.95])]

    def fit(self, history, features=None):
        if features is None:
            raise ValueError("tft needs the feature table (calendar covariates)")
        train_end = int(history["d"].max())
        cols = ["id", "item_id", "dept_id", "store_id", "d", "sales",
                "dow", "dom", "month", "snap", "is_event", "sell_price"]
        panel = features[features["d"] <= train_end + 28][cols]
        self.arrays = build_arrays(panel)
        self.arrays["sales"][:, train_end:] = 0.0  # leakage guard (see DeepAR)
        self.train_end = train_end

        ctx = int(self.p.context_length)
        ds = WindowDataset(self.arrays, context=ctx, horizon=28, max_day=train_end,
                           windows_per_series=int(self.p.get("windows_per_series_per_epoch", 2)), seed=42)
        dl = DataLoader(ds, batch_size=int(self.p.batch_size), shuffle=True, drop_last=True)

        self.net = TFTNet(
            n_items=int(self.arrays["cats"][:, 0].max()) + 1,
            n_depts=int(self.arrays["cats"][:, 1].max()) + 1,
            n_stores=int(self.arrays["cats"][:, 2].max()) + 1,
            cov_dim=TIME_FEATURES, context_len=ctx, horizon=28,
            quantiles=self.quantile_levels, hidden=int(self.p.hidden_size),
            heads=int(self.p.attention_heads), dropout=float(self.p.dropout),
            embed_dim=int(self.p.get("embedding_dim", 16)),
        ).to(self.device)

        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.p.lr))
        self.net.train()
        for epoch in range(int(self.p.epochs)):
            ds.resample()
            tot, nb = 0.0, 0
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.net(batch["past_scaled"], batch["past_covs"], batch["fut_covs"], batch["cats"])
                loss = self.net.quantile_loss(preds, batch["target"] / batch["nu"].unsqueeze(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                opt.step()
                tot, nb = tot + float(loss.detach()), nb + 1
            log.info("tft epoch %d/%d: pinball=%.4f", epoch + 1, self.p.epochs, tot / max(nb, 1))
        return self

    @torch.no_grad()
    def predict(self, future):
        ctx = int(self.p.context_length)
        horizon = int(future["d"].max() - self.train_end)
        a, T = self.arrays, self.train_end
        n_series = a["sales"].shape[0]
        Q = len(self.quantile_levels)
        self.net.eval()

        out_q = np.zeros((n_series, horizon, Q), dtype="float32")
        attn_accum = np.zeros(ctx, dtype="float64")
        attn_n = 0
        chunk = int(self.p.get("predict_chunk", 1024))
        for lo in range(0, n_series, chunk):
            hi = min(lo + chunk, n_series)
            past = a["sales"][lo:hi, T - ctx : T]
            nu = 1.0 + past.mean(axis=1)
            ps = torch.from_numpy(past / nu[:, None]).to(self.device)
            pc = torch.from_numpy(a["covs"][lo:hi, T - ctx : T]).to(self.device)
            fc = torch.from_numpy(a["covs"][lo:hi, T : T + horizon]).to(self.device)
            cats = torch.from_numpy(a["cats"][lo:hi]).to(self.device)

            want_attn = lo == 0
            res = self.net(ps, pc, fc, cats, return_attn=want_attn)
            preds = (res[0] if want_attn else res).clamp(min=0)
            preds = preds * torch.from_numpy(nu.astype("float32")).to(self.device)[:, None, None]
            preds, _ = preds.sort(dim=-1)  # enforce quantile monotonicity
            out_q[lo:hi] = preds.cpu().numpy()
            if want_attn:
                # attention from decoder queries over the encoder positions, mean over batch+queries
                aw = res[1][:, :, :ctx].mean(dim=(0, 1)).cpu().numpy()
                attn_accum += aw; attn_n += 1

        self.attention_ = attn_accum / max(attn_n, 1)  # [ctx] mean weight per past day

        id_codes = pd.Categorical(future["id"], categories=a["ids"]).codes
        day_idx = future["d"].to_numpy() - (T + 1)
        med_col = self.quantile_levels.index(0.5)
        yhat = out_q[id_codes, day_idx, med_col]

        self.quantiles_ = future[["id", "d"]].copy()
        for k, q in enumerate(self.quantile_levels):
            self.quantiles_[f"q{q}"] = out_q[id_codes, day_idx, k]
        return self._finalize(future, pd.Series(yhat, index=future.index))

    def predict_quantiles(self, future, quantiles):
        if not hasattr(self, "quantiles_"):
            self.predict(future)
        return self.quantiles_
