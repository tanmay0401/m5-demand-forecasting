# Phase 4 — Project Planning & Architecture

> Status: ✅ Complete · Output: architecture, folder structure, pipeline contracts, module dependency graph, tooling decisions — and the scaffolded skeleton in the repo.

---

## 1. Design philosophy

Three rules govern every choice below:

1. **Pipelines are stages connected by files, not function calls.** Each stage reads artifacts (Parquet/CSV) from disk and writes new ones. Why: any stage can be rerun, inspected, or debugged in isolation; a crashed 3-hour training run doesn't force re-doing preprocessing; and it mirrors how industrial systems (Airflow/Dagster DAGs) actually work.
2. **One direction of dependency.** `utils ← data ← features ← models ← evaluation ← analysis`. Lower layers never import higher ones. Why: no circular imports, testable layers, and you can explain the system as a DAG in an interview.
3. **Config over code.** Anything you might want to change between experiments (lags, model params, folds) lives in YAML, not in Python. Why: an experiment = a config + a git commit + an MLflow run — fully reproducible.

## 2. System architecture (dataflow)

```
                        ┌────────────────────────────────────────────────┐
                        │                  configs/ (Hydra)              │
                        └────────────────────────────────────────────────┘
 raw M5 CSVs                                                            
 (sales, calendar, prices)                                              
      │                                                                 
      ▼                                                                 
 [1] DATA PIPELINE ──────────► data/processed/panel.parquet             
      melt wide→long, join calendar+prices, dtypes, validation          
      │                                                                 
      ▼                                                                 
 [2] FEATURE PIPELINE ───────► data/processed/features.parquet          
      lags, rollings, calendar, price/promo features (shift→roll!)      
      │                                                                 
      ▼                                                                 
 [3] TRAINING PIPELINE ──────► outputs/forecasts/{model}/*.parquet      
      baselines · LightGBM · DeepAR-style · TFT-style                   
      backtest folds · MLflow logging · point + quantile forecasts      
      │                                                                 
      ▼                                                                 
 [4] HIERARCHY PIPELINE ─────► aggregated base forecasts (12 levels)    
      summing matrix S · aggregate actuals & forecasts                  
      │                                                                 
      ▼                                                                 
 [5] RECONCILIATION PIPELINE ► coherent forecasts (BU / TD / MinT)      
      │                                                                 
      ▼                                                                 
 [6] EVALUATION PIPELINE ────► reports/ (WRMSSE, quantile loss,         
      metrics tables, comparison plots, calibration)                    
      │                                                                 
      ▼                                                                 
 [7] ANALYSIS ───────────────► promo/event studies, error taxonomy,     
                               research-question answers (Phases 14–16) 
```

## 3. Folder structure (scaffolded in this phase)

```
m5-demand-forecasting/
├── configs/                    # Hydra config tree — the experiment surface
│   ├── config.yaml             #   defaults composition + global settings (seed, paths)
│   ├── data/m5.yaml            #   file names, date ranges, dtypes
│   ├── features/default.yaml   #   which lags, windows, encodings
│   ├── model/                  #   one file per model family
│   │   ├── baselines.yaml  lightgbm.yaml  deepar.yaml  tft.yaml
│   ├── training/default.yaml   #   folds, horizon, device, batch sizes
│   └── evaluation/default.yaml #   metrics, quantile levels
├── data/                       # gitignored — artifacts, never in git
│   ├── raw/                    #   M5 CSVs exactly as downloaded (read-only)
│   ├── interim/                #   long-format panel, pre-feature
│   └── processed/              #   model-ready feature tables
├── docs/phases/                # the teaching/design docs (Phases 1–3 live here)
├── notebooks/                  # EDA & scratch — import from src, never define logic
├── outputs/                    # gitignored — forecasts, checkpoints, mlruns
├── reports/figures/            # committed — final plots & tables for the README/report
├── scripts/                    # thin CLI entry points (one per pipeline stage)
│   ├── download_data.py  build_features.py  train.py
│   ├── reconcile.py  evaluate.py  analyze.py
├── src/m5forecast/             # the installable package — ALL logic lives here
│   ├── data/                   #   loading.py, preprocess.py, validate.py
│   ├── features/               #   calendar.py, lags.py, price.py, build.py
│   ├── models/                 #   base.py, baselines.py, lgbm.py, deepar/, tft/
│   ├── hierarchy/              #   aggregation.py, reconciliation.py
│   ├── evaluation/             #   wrmsse.py, quantile_loss.py, backtest.py, compare.py
│   ├── analysis/               #   promotions.py, errors.py, research.py
│   └── utils/                  #   seed.py, logging.py, io.py
├── tests/                      # pytest — mirrors src layout
├── requirements.txt
└── .gitignore
```

## 4. Why every module exists

| Module | Exists because… | Key contract |
|---|---|---|
| `data/loading.py` | Raw CSVs are wide (1 row per series, 1,941 day-columns) and 450MB — need memory-aware loading with explicit dtypes | returns typed DataFrames |
| `data/preprocess.py` | Models need long format (1 row per item-store-day) with calendar+price joined | writes `panel.parquet` |
| `data/validate.py` | Silent data corruption is the #1 pipeline killer — assert row counts, date continuity, no negative sales | raises on violation |
| `features/lags.py` | The shift→roll→aggregate discipline (Phase 2) must live in ONE audited place, not be re-implemented per notebook | all features leakage-safe by construction |
| `features/calendar.py` | Day-of-week/SNAP/events encode seasonality & events (Phase 1) | pure calendar → features |
| `features/price.py` | Promotions are *inferred* from price moves — the project's namesake feature set | price → promo flags, elasticity inputs |
| `features/build.py` | One orchestrator that assembles the feature matrix from config, so "which features?" is a YAML question | writes `features.parquet` |
| `models/base.py` | All models must expose identical `fit / predict / predict_quantiles` so evaluation can't tell them apart — comparisons stay fair | abstract interface |
| `models/baselines.py` | Phase 3 lesson: 92% of M5 teams lost to a baseline. Everything is measured against these | naive, seasonal-naive, MA, ES |
| `models/lgbm.py` | Primary model family (M5 winner's choice): Tweedie point model + per-quantile models | tabular in, forecasts out |
| `models/deepar/` | Probabilistic RNN: NegBin likelihood, ancestral sampling (own dataset/network/sampling submodules) | samples → quantiles |
| `models/tft/` | Attention model: static/known/observed routing, direct multi-quantile head | quantiles in one pass |
| `hierarchy/aggregation.py` | The 12 M5 levels as a **summing matrix S** — one canonical definition used by metrics AND reconciliation | S + level metadata |
| `hierarchy/reconciliation.py` | Coherence: BU / TD / MinT-shrinkage as matrix operations on base forecasts | incoherent → coherent |
| `evaluation/wrmsse.py` | The official M5 metric — scale + weights must exactly match the competition definition (subtle; tested against known values) | forecasts → score |
| `evaluation/quantile_loss.py` | Pinball loss + coverage + calibration for the probabilistic claims | quantiles → scores |
| `evaluation/backtest.py` | Expanding-window fold generator (Phase 2) — the ONLY place train/test boundaries are computed | fold indices |
| `analysis/*` | Phases 14–16: promo effects, error taxonomy, research answers — kept out of evaluation so metrics stay pure | reports |
| `utils/seed.py` | One `set_seed(seed)` for random/numpy/torch/lightgbm — reproducibility is a resume claim | determinism |
| `scripts/*` | Thin Hydra entry points; logic stays importable/testable in the package | CLI |

## 5. Module dependency graph

```
utils ──► data ──► features ──► models ──► evaluation ──► analysis
              │                    ▲            ▲
              └── hierarchy ───────┴────────────┘
   (aggregation feeds model targets at higher levels, reconciliation,
    and the WRMSSE weights — hierarchy depends only on data)
scripts/ orchestrate everything; notebooks/ only import, never define.
```

Arrows = "may import from". Nothing imports backwards. `hierarchy` is deliberately independent of `models` so reconciliation works on *any* model's forecasts — that's what makes the Phase 16 "how much does reconciliation help *each* model family" experiment a one-liner.

## 6. Tooling decisions (each is an interview question)

| Decision | Choice | Why (and the alternative) |
|---|---|---|
| Config system | **Hydra** | Composable YAML groups (`model=lightgbm` ↔ `model=tft` on the CLI), auto output dirs, standard in research code. Alt: plain YAML + argparse (simpler, but no composition/overrides). |
| Experiment tracking | **MLflow** | Fully local (`mlruns/`), no account, params+metrics+artifacts per run, UI via `mlflow ui`. Alt: W&B — nicer UI but external service; for a public repo, local-first is cleaner. |
| Data format | **Parquet** | Columnar, typed, compressed; 46M-row feature table is ~1GB in Parquet vs ~10GB CSV; preserves dtypes across stages. |
| DL framework | **PyTorch** | We implement DeepAR/TFT from scratch (the learning goal) rather than using GluonTS/pytorch-forecasting (the production shortcut — we'll say exactly this in the README). |
| GBM | **LightGBM primary, XGBoost comparison** | Phase 3 evidence; Tweedie objective; native categoricals; speed at 46M rows. |
| Testing | **pytest** | Unit tests for leakage (feature dates!), WRMSSE against hand-computed values, S-matrix coherence; smoke tests on a 100-series sample. |
| Reproducibility | `set_seed()` + pinned `requirements.txt` + config-in-git + MLflow | An experiment is re-runnable from (commit, config) alone. |
| Compute plan | LightGBM: all 30,490 series, CPU. Deep models: all series, **sampled training windows** per epoch (DeepAR paper's own trick); GPU if available, else reduced epochs on a documented subset. | Honesty over heroics — the config records exactly what ran. |

## 7. Risks & mitigations (planned now, so they don't surprise us later)

| Risk | Mitigation |
|---|---|
| Memory blow-up on 46M-row feature table | float32/int16 dtypes, categorical codes, per-store processing if needed |
| Leakage via features | leakage unit tests: assert every feature at day *t* is computable from data ≤ t−horizon |
| WRMSSE implementation drift from official definition | test against published benchmark scores + hand-computed toy hierarchy |
| Deep models too slow on laptop | window sampling, mixed precision on GPU, documented series subset fallback |
| Christmas zero-sales outlier distorting scales | explicit handling decided in Phase 5 EDA (flag, not silently drop) |

## 8. Interview questions — Phase 4

1. *Why files between pipeline stages instead of one big script?* (Isolation, resumability, inspectability — mirrors DAG orchestrators.)
2. *Why must all models share one interface?* (Fair comparison — evaluation code literally cannot special-case a model; swapping models is a config change.)
3. *Why is the hierarchy module independent of models?* (Reconciliation is a post-processing projection applicable to any base forecast — that separation IS the experiment design for "does reconciliation help each family equally?")
4. *Why Hydra over argparse?* (Config composition & sweeps: `python train.py model=tft training.folds=1` — every run's full config is serialized next to its outputs.)
5. *Where would this break at Walmart scale, and what changes?* (Single-machine Parquet → distributed store; per-run training → scheduled retraining service; MLflow local → model registry; but the *layering* survives — that's the point of it.)

---

*Next: Phase 5 — Dataset download, understanding, and the data pipeline (first real code).*
