"""Parquet IO helpers: create parent dirs, log sizes, keep dtypes intact."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from m5forecast.utils.logging import get_logger

log = get_logger(__name__)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to Parquet, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("wrote %s (%d rows, %.1f MB)", path, len(df), path.stat().st_size / 1e6)
    return path


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a Parquet file, optionally selecting columns (much cheaper than loading all)."""
    df = pd.read_parquet(path, columns=columns)
    log.info("read %s (%d rows, %d cols)", path, len(df), df.shape[1])
    return df
