"""Phase 6 entry point: panel.parquet -> reports/figures/*.png + eda_stats.json."""

from __future__ import annotations

import json
import sys

from m5forecast.analysis.eda import run_all
from m5forecast.utils.config import REPO_ROOT, load_config
from m5forecast.utils.io import read_parquet
from m5forecast.utils.seed import set_seed


def main(overrides: list[str]) -> None:
    cfg = load_config(overrides)
    set_seed(cfg.seed)
    panel = read_parquet(REPO_ROOT / cfg.paths.interim / "panel.parquet")
    stats = run_all(panel, REPO_ROOT / cfg.paths.figures)
    stats_path = REPO_ROOT / cfg.paths.figures / "eda_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
