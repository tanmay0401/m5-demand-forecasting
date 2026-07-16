# Phase 5 — The M5 Dataset & Data Pipeline

> Status: ✅ Code complete · Teaches the three raw files, how they relate, and the pipeline that turns them into `panel.parquet`.

---

## 1. What the M5 dataset contains

Three CSV files from Walmart, covering **2011-01-29 → 2016-06-19** (1,969 calendar days; sales observed for the first 1,941):

```
sales_train_evaluation.csv   30,490 rows × 1,947 cols   (the target)
calendar.csv                  1,969 rows × 14 cols      (time context)
sell_prices.csv           6,841,121 rows × 4 cols       (price context)
```

### 1.1 `sales_train_evaluation.csv` — the target, in WIDE format

One row per **item-store series**, one column per day:

| id | item_id | dept_id | cat_id | store_id | state_id | d_1 | d_2 | … | d_1941 |
|---|---|---|---|---|---|---|---|---|---|
| FOODS_3_090_CA_1_evaluation | FOODS_3_090 | FOODS_3 | FOODS | CA_1 | CA | 0 | 2 | … | 1 |

**The ids encode the entire hierarchy** — read `FOODS_3_090_CA_1` right-to-left:
- `state_id = CA` — 3 states: **CA** (4 stores), **TX** (3), **WI** (3)
- `store_id = CA_1` — 10 stores total
- `cat_id = FOODS` — 3 categories: FOODS, HOUSEHOLD, HOBBIES
- `dept_id = FOODS_3` — 7 departments (FOODS_1-3, HOUSEHOLD_1-2, HOBBIES_1-2)
- `item_id = FOODS_3_090` — 3,049 distinct products
- 3,049 items × 10 stores = **30,490 series**

Values are **unit sales** (integer counts). Two crucial properties we verify in EDA:
- **~68% of all values are 0** — intermittent demand is the norm, not the exception.
- Sales are *censored* demand (Phase 1): a 0 can mean "no buyers" or "no stock" — indistinguishable in this data.

Note: `sales_train_validation.csv` is the same file cut at d_1913 (the competition's public split). We use the evaluation file (all 1,941 days) and enforce splits ourselves in the backtester.

### 1.2 `calendar.csv` — one row per calendar day

| Column | Meaning | Why it matters |
|---|---|---|
| `date`, `d` | calendar date ↔ day index (`d_1` = 2011-01-29) | joins sales to real time |
| `wm_yr_wk` | Walmart week id (e.g. 11101) | **the join key to prices** |
| `weekday`, `wday`, `month`, `year` | derivable from date — we recompute in features | — |
| `event_name_1`, `event_type_1` | e.g. SuperBowl/Sporting, Christmas/National, Ramadan starts/Religious | holiday & event features |
| `event_name_2/type_2` | second event on the same day (rare — e.g. Easter ∩ OrthodoxEaster) | — |
| `snap_CA`, `snap_TX`, `snap_WI` | 1 if SNAP food-stamp purchases are disbursed that day *in that state* | grocery demand spikes on SNAP days; states have different SNAP schedules |

**SNAP explained** (interview favorite): the US Supplemental Nutrition Assistance Program loads benefits onto cards on fixed monthly schedules that differ per state (e.g., first 10 days in CA, staggered in TX/WI). Low-income households concentrate grocery shopping on those days → a strong, perfectly *predictable* demand driver. Our pipeline collapses the three columns into one `snap` flag per row, matched to the row's own state — one honest feature instead of three two-thirds-irrelevant ones.

### 1.3 `sell_prices.csv` — one row per (store, item, week)

| store_id | item_id | wm_yr_wk | sell_price |
|---|---|---|---|
| CA_1 | FOODS_3_090 | 11101 | 2.50 |

- **Weekly, not daily** — the finest price granularity Walmart released. Joining to the daily panel gives every day of a week that week's price.
- **A missing row means the item was not offered that store-week.** This is *signal*: it marks pre-launch periods (and the occasional delisting), not missing data to impute. After the join, `sell_price = NaN` exactly on those days — and validation asserts nothing was ever *sold* without a price.
- **This file is where promotions live.** M5 has no "on promo" flag; a week where price drops 25% below the item's norm *is* the promotion. Phase 7 builds that inference.

## 2. How the files relate (the join graph)

```
sales (30,490 × 1,941 days)          calendar (1,969 days)
        │  d  ────────────────────────────►  d, date, wm_yr_wk, events, snap
        │                                         │
        │                                         │ wm_yr_wk
        ▼                                         ▼
      melt → long panel  ◄──── (store_id, item_id, wm_yr_wk) ──── sell_prices
```

One fact table (sales) + two dimension tables (calendar keyed by day, prices keyed by store-item-week). The classic star-schema shape of retail data warehouses.

## 3. The pipeline (`src/m5forecast/data/`)

**Wide → long ("melt").** Models want one row per (series, day) — the *panel* format: 30,490 × 1,941 = **59,181,090 rows**. Implementation detail that matters at this scale: we rename day columns `d_1913 → 1913` (ints) *before* melting, because melting 59M string labels and slicing them afterwards allocates gigabytes of throwaway Python strings.

**Dtype discipline.** int16 sales (max observed daily sales is 763), int16 day index, float32 prices, category dtype for all ids (integer codes + one lookup table instead of 59M strings). Result: ~2GB in RAM instead of ~15GB naive. This is the difference between "runs on a laptop" and "needs a cluster" — and it's a deliberate, defensible engineering claim.

**Calendar join** on `d`, then **SNAP resolution** (`snap_CA/TX/WI` → single `snap` matched to the row's state). **Price join** on `(store_id, item_id, wm_yr_wk)` — with merge-key categories aligned first so pandas takes its fast categorical path.

**Validation (`validate.py`)** — every run asserts: exact row count (a bad join silently multiplies or drops rows), no null/negative sales, exactly 1,941 distinct days and dates, and **zero rows with sales > 0 but no price** (the invariant that proves the price join is correct). Violations raise `DataValidationError` and kill the run — corrupted artifacts must never flow downstream.

Output contract: `data/interim/panel.parquet`, sorted by (id, d) — the **only** file the rest of the project reads.

## 4. Getting the data

```bash
# one-time Kaggle setup: kaggle.com → Settings → API → Create New Token
#   → save kaggle.json to %USERPROFILE%\.kaggle\kaggle.json
# accept rules once: kaggle.com/competitions/m5-forecasting-accuracy/rules
python scripts/download_data.py
python scripts/build_panel.py
pytest                       # 11 pipeline tests on synthetic mini-M5 fixtures
```

## 5. Interview questions — Phase 5

**Easy**
1. Describe the three M5 files and their keys. *(Wide sales per item-store; calendar per day; prices per store-item-week; joined via `d` and `wm_yr_wk`.)*
2. How many series, and how do 3,049 items become 30,490 series? *(× 10 stores — a series is an item AT a store.)*

**Medium**
3. Why store the panel in long format? *(One row per observation is what feature engineering, groupby-lags, and every ML library expect; wide format can't hold per-day covariates like price.)*
4. What does a missing `sell_prices` row mean and how do you treat it? *(Item not offered that store-week — mostly pre-launch. Keep NaN as signal; assert never sold-without-price; never impute a price for a product that wasn't on sale.)*
5. What is SNAP and why does it predict grocery demand? *(US food-assistance disbursed on fixed per-state monthly schedules; recipients concentrate purchases on those days — a strong periodic demand driver, flagged per state in the calendar.)*

**Hard**
6. Your melted panel has 59,181,091 rows instead of 59,181,090. What happened and how would you catch it? *(A duplicate key in a dimension table fanned out the join; the row-count validation catches it — this is exactly why the check exists.)*
7. Why int16/float32/category dtypes — and where's the risk? *(4-8× memory reduction on 59M rows; risk is silent overflow — int16 caps at 32,767; we verified max daily sales = 763 and validation would catch negatives from wraparound.)*
8. The sales file gives units, prices are weekly averages. What can you NOT compute reliably? *(Exact daily revenue and within-week price moves — daily revenue = units × weekly price is an approximation; intra-week promotions are invisible.)*

---

*Next: Phase 6 — EDA: seeing the seasonality, SNAP effects, events, and intermittency we've been claiming.*
