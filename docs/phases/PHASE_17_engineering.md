# Phase 17 — Engineering Standards

> Status: ✅ Complete · The hardening pass: reproducibility, CI, an integration test, a pipeline runner, and a linter — turning a working research codebase into a production-grade one.

---

## 1. What "production-quality research code" means here

The resume claim is production-quality, reproducible, testable code. Concretely, this repo satisfies:

| standard | how |
|---|---|
| **Reproducible** | `set_seed(cfg.seed)` seeds random/numpy/torch in every training entry point; every experiment is `(git commit, config, seed)`; deterministic analysis scripts |
| **Config-driven** | Hydra-style YAML tree composed by `utils/config.py`; a new experiment is a CLI override, never a code edit |
| **Tested** | 60 tests: unit (per module), property (leakage, metric identities), and one **end-to-end integration** test through the real pipeline |
| **Typed & documented** | type hints throughout; every module has a docstring stating its contract and *why* it exists |
| **Layered** | strict `utils ← data ← features ← models ← evaluation ← analysis` dependency direction; `hierarchy` independent of `models` |
| **Tracked** | MLflow (sqlite backend, `outputs/mlflow.db`) logs params + metrics per (model, fold); browse with `mlflow ui` |
| **CI** | GitHub Actions runs the suite on every push/PR |
| **Runnable** | `Makefile` targets for every pipeline stage; `pip install -e . && pytest` clean from scratch |

## 2. The integration test earned its keep immediately

Unit tests check modules in isolation; they miss **contract breaks between stages**. `tests/test_integration.py` runs synthetic raw CSVs through the *real* modules end-to-end — melt → validate → features → baseline → backtest → hierarchy → WRMSSE — and it caught a genuine bug on first run:

> `ForecastModel._finalize` called `.clip()` on the forecast Series, but when the caller mapped a **categorical** `id` column the result was itself categorical, which cannot be clipped (`TypeError: Unordered Categoricals`). Unit tests missed it because their fixtures use *string* ids; the real runs missed it because the Parquet round-trip changes the id dtype. The integration test, using categorical ids like the real panel, exposed it.

Fix: `_finalize` now coerces to a float array (`pd.to_numeric → np.clip`) regardless of input dtype. This is precisely the class of defect integration tests exist to catch — a seam between two modules that each looked correct alone.

## 3. Reproducibility, verified

- `pip install -e .` succeeds from the `src`-layout `pyproject.toml`; all **20 package modules import cleanly** (checked in this phase).
- The full suite (`pytest`) is **green (60 passed)** and requires **no downloaded data** — every test uses synthetic fixtures, so CI runs without Kaggle credentials. The deep-model tests `importorskip("torch")`, so the suite degrades gracefully where torch is absent.
- `requirements.txt` audited against actual imports; `scipy` (used by the hierarchy) promoted from a transitive to an explicit dependency.

## 4. Tooling added

- **`Makefile`** — `setup`, `test`, `lint`, and one target per pipeline stage (`panel`, `features`, `gbm`, `deep`, `reconcile`, `evaluate`, `analyze`), plus `all`.
- **`.github/workflows/tests.yml`** — installs core deps + CPU torch, runs `pytest`, and runs `ruff` (advisory).
- **`ruff`** config in `pyproject.toml` (E/F/I/UP/B rules); 26 issues auto-fixed (import ordering), the rest are deliberate (deferred matplotlib imports inside figure functions to keep module import light; compact `;` statements).

## 5. Honest remaining gaps (documented, not hidden)

- **Deep models run a single fold** (compute honesty on a laptop); a cluster run would do all 3 for tighter variance estimates.
- **Exact MinT** is demonstrated on the upper hierarchy only (the 30,490-bottom solve is infeasible without sparse solvers — Phase 12).
- **Analysis scripts assume the stored fold artifacts exist** (run `train.py` first); they are deterministic so no seed is set in them.
- **No Dockerfile** — the venv + pinned requirements are the reproducibility boundary; a container would be the next step for true environment hermeticity.

## 6. Interview questions — Phase 17

1. What makes an experiment reproducible in this repo? *((commit, config, seed) fully determines it; seeds set in every training entry point; configs in git.)*
2. Why an integration test on top of unit tests? *(Unit tests miss inter-module contract breaks — dtypes, column names, id ordering across stages; the integration test found a real categorical-clip bug unit tests couldn't.)*
3. Why does CI run without the M5 data? *(All tests use synthetic fixtures; the pipeline is validated on tiny generated data so CI needs no credentials and stays fast.)*
4. Your tests skip if torch is missing — good or bad? *(Deliberate: keeps the core suite runnable in minimal environments; the deep-model tests are guarded with importorskip and run in full CI where torch is installed.)*

---

*Next: Phase 18 — the full documentation set (technical report, guides, limitations) and the single-touch held-out final evaluation.*
