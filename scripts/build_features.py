"""Stage 2 entry point: panel.parquet -> data/processed/features/ (per-store parquet)."""

from __future__ import annotations

import sys

from m5forecast.features.build import build_features
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.seed import set_seed


def main(overrides: list[str]) -> None:
    cfg = load_config(overrides)
    set_seed(cfg.seed)
    build_features(
        panel_path=REPO_ROOT / cfg.paths.interim / "panel.parquet",
        out_dir=REPO_ROOT / cfg.paths.processed / "features",
        cfg=cfg,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
