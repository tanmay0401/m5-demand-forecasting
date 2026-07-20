"""Feature-matrix orchestrator: panel.parquet -> features/ (partitioned by store).

Why per-store streaming: the full 59M-row x ~35-column float32 table is
~8GB plus intermediates — more than this machine's free RAM. Each store
is ~5.9M rows and builds comfortably; the output is a directory of ten
parquet files that pandas/pyarrow read back as one dataset. (This is the
Phase 4 risk-register mitigation, exercised.)

Build order matters: rolling means must exist before momentum; calendar
and price families are independent of the target-derived ones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from m5forecast.features.calendar import add_calendar_features
from m5forecast.features.lags import (
    add_ewm_features,
    add_expanding_mean,
    add_lag_features,
    add_momentum,
    add_rolling_features,
)
from m5forecast.features.price import add_price_features
from m5forecast.utils.logging import get_logger

log = get_logger(__name__)

STORES = [f"CA_{i}" for i in range(1, 5)] + [f"TX_{i}" for i in range(1, 4)] + [f"WI_{i}" for i in range(1, 4)]


def build_store_features(store_panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """Apply every feature family (config-driven) to one store's panel slice."""
    df = store_panel.sort_values(["id", "d"], ignore_index=True)
    horizon = int(cfg.data.horizon)

    df = add_lag_features(df, list(cfg.features.lags), horizon)
    df = add_rolling_features(
        df, list(cfg.features.rolling.windows), list(cfg.features.rolling.stats),
        int(cfg.features.rolling.shift), horizon,
    )
    df = add_ewm_features(df, list(cfg.features.ewm.alphas), int(cfg.features.rolling.shift), horizon)
    df = add_expanding_mean(df, int(cfg.features.rolling.shift), horizon)
    df = add_momentum(df)
    df = add_calendar_features(df)
    df = add_price_features(df)
    return df


def build_features(panel_path: str | Path, out_dir: str | Path, cfg) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for store in STORES:
        # pyarrow filter pushdown: only this store's rows ever enter RAM
        chunk = pd.read_parquet(panel_path, filters=[("store_id", "=", store)])
        feat = build_store_features(chunk, cfg)
        out = out_dir / f"store={store}.parquet"
        feat.to_parquet(out, index=False)
        log.info("%s: %d rows, %d cols -> %s", store, len(feat), feat.shape[1], out.name)
        del chunk, feat  # keep peak memory to one store


def read_features(out_dir: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read the partitioned feature set back as one frame."""
    parts = sorted(Path(out_dir).glob("store=*.parquet"))
    return pd.concat((pd.read_parquet(p, columns=columns) for p in parts), ignore_index=True)
