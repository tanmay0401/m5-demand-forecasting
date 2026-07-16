"""Stage 1 entry point: raw M5 CSVs -> data/interim/panel.parquet.

Run from the repo root:
    python scripts/build_panel.py
    python scripts/build_panel.py data.files.sales=sales_train_validation.csv
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from m5forecast.data.preprocess import build_panel
from m5forecast.utils.seed import set_seed


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    repo_root = Path(__file__).resolve().parents[1]
    build_panel(
        raw_dir=repo_root / cfg.paths.raw,
        out_path=repo_root / cfg.paths.interim / "panel.parquet",
        sales_file=cfg.data.files.sales,
        n_series=cfg.data.n_series,
        n_days=cfg.data.n_days,
    )


if __name__ == "__main__":
    main()
