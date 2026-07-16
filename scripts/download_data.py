"""Download the M5 competition files from Kaggle into data/raw.

Prerequisites (one-time):
  1. Kaggle account -> Settings -> API -> "Create New Token"
  2. Save the downloaded kaggle.json to  %USERPROFILE%/.kaggle/kaggle.json
  3. Accept the competition rules once at
     https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules

Plain argparse (not Hydra) on purpose: downloading has no experiment
surface — there is nothing to sweep or compose.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REQUIRED = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
COMPETITION = "m5-forecasting-accuracy"


def main(raw_dir: Path) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in REQUIRED if not (raw_dir / f).exists()]
    if not missing:
        print(f"all files already present in {raw_dir}, nothing to do")
        return 0

    print(f"downloading {COMPETITION} -> {raw_dir}")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(raw_dir)],
    )
    if result.returncode != 0:
        print(
            "\nKaggle download failed. Check that kaggle.json is in place and that\n"
            "you accepted the competition rules (see module docstring).",
            file=sys.stderr,
        )
        return result.returncode

    archive = raw_dir / f"{COMPETITION}.zip"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()

    still_missing = [f for f in REQUIRED if not (raw_dir / f).exists()]
    if still_missing:
        print(f"archive extracted but files missing: {still_missing}", file=sys.stderr)
        return 1
    print("done:", ", ".join(REQUIRED))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    raise SystemExit(main(parser.parse_args().raw_dir))
