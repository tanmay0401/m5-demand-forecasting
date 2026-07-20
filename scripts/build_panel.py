"""Stage 1 entry point: raw M5 CSVs -> data/interim/panel.parquet.

Run from the repo root:
    python scripts/build_panel.py
    python scripts/build_panel.py data.files.sales=sales_train_validation.csv
"""

from __future__ import annotations

import sys

from m5forecast.data.preprocess import build_panel
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.seed import set_seed


def main(overrides: list[str]) -> None:
    cfg = load_config(overrides)
    set_seed(cfg.seed)
    build_panel(
        raw_dir=REPO_ROOT / cfg.paths.raw,
        out_path=REPO_ROOT / cfg.paths.interim / "panel.parquet",
        sales_file=cfg.data.files.sales,
        n_series=cfg.data.n_series,
        n_days=cfg.data.n_days,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
