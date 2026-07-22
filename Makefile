# M5 demand forecasting — pipeline & dev tasks.
# Windows: run targets via `make <target>` (Git Bash / mingw32-make) or copy the
# command. Each stage reads the previous stage's parquet artifact (Phase 4 design).

PY := python

.PHONY: help setup test lint data panel features baselines gbm deep reconcile evaluate analyze all clean

help:
	@echo "setup      install package + deps (editable)"
	@echo "test       run the pytest suite"
	@echo "lint       ruff check"
	@echo "data       download M5 from Kaggle -> data/raw"
	@echo "panel      raw CSVs -> data/interim/panel.parquet"
	@echo "features   panel -> data/processed/features/"
	@echo "baselines|gbm|deep   train a model family (backtest)"
	@echo "reconcile|evaluate|analyze   post-hoc studies"
	@echo "all        panel -> features -> baselines+gbm -> evaluate"

setup:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

test:
	$(PY) -m pytest

lint:
	ruff check src scripts tests

data:
	$(PY) scripts/download_data.py

panel:
	$(PY) scripts/build_panel.py

features:
	$(PY) scripts/build_features.py

baselines:
	$(PY) scripts/train.py model=baselines

gbm:
	$(PY) scripts/train.py model=lightgbm
	$(PY) scripts/train.py model=xgboost

deep:
	$(PY) scripts/train.py model=deepar training.backtest.n_folds=1
	$(PY) scripts/train.py model=tft training.backtest.n_folds=1

reconcile:
	$(PY) scripts/reconcile.py

evaluate:
	$(PY) scripts/evaluate.py

analyze:
	$(PY) scripts/analyze_promos.py
	$(PY) scripts/analyze_errors.py
	$(PY) scripts/analyze_research.py

all: panel features baselines gbm evaluate

clean:
	rm -rf data/interim/* data/processed/* outputs/forecasts outputs/*.json outputs/mlflow.db
