"""Hierarchy + reconciliation tests on a tiny hand-checkable hierarchy.

Toy structure: 2 stores x 2 items = 4 bottom series, with a plausible
subset of the M5 levels so S is small enough to write out by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m5forecast.hierarchy.aggregation import build_hierarchy
from m5forecast.hierarchy.reconciliation import (
    bottom_up,
    coherence_error,
    historical_proportions,
    mint,
    top_down,
)


@pytest.fixture
def meta():
    # 2 items x 2 stores in 1 state, 1 category, 1 department
    rows = [
        ("I1_S1", "I1", "D1", "C1", "S1", "ST"),
        ("I1_S2", "I1", "D1", "C1", "S2", "ST"),
        ("I2_S1", "I2", "D1", "C1", "S1", "ST"),
        ("I2_S2", "I2", "D1", "C1", "S2", "ST"),
    ]
    return pd.DataFrame(rows, columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])


def test_summing_matrix_shape_and_total(meta):
    h = build_hierarchy(meta)
    assert h.S.shape[1] == 4  # 4 bottom series
    # total row sums all 4 bottoms
    total_row = h.S[h.level_slices["total"]].toarray()
    assert total_row.sum() == 4 and (total_row == 1).sum() == 4
    # bottom level is identity
    bottom = h.S[h.bottom_slice].toarray()
    assert np.allclose(bottom, np.eye(4))


def test_aggregation_sums_correctly(meta):
    h = build_hierarchy(meta)
    b = np.array([1.0, 2.0, 3.0, 4.0])  # I1_S1, I1_S2, I2_S1, I2_S2
    allv = h.aggregate(b)
    assert allv[h.level_slices["total"]][0] == 10.0
    # store level: S1 = I1_S1 + I2_S1 = 1+3=4 ; S2 = 2+4=6
    stores = h.nodes[h.nodes["level"] == "store"].sort_values("key")
    store_vals = {r.key: allv[r.node_idx] for r in stores.itertuples()}
    assert store_vals["S1"] == 4.0 and store_vals["S2"] == 6.0
    # item level: I1 = 1+2=3 ; I2 = 3+4=7
    items = h.nodes[h.nodes["level"] == "item"].sort_values("key")
    item_vals = {r.key: allv[r.node_idx] for r in items.itertuples()}
    assert item_vals["I1"] == 3.0 and item_vals["I2"] == 7.0


def test_base_incoherent_bu_coherent(meta):
    h = build_hierarchy(meta)
    rng = np.random.default_rng(0)
    # independent per-node base forecasts (incoherent on purpose)
    base = rng.uniform(1, 10, size=(h.n_nodes, 3))
    assert coherence_error(base, h) > 1e-6           # base doesn't add up
    rec = bottom_up(base, h)
    assert coherence_error(rec, h) < 1e-9            # BU is coherent


def test_top_down_coherent_and_matches_total(meta):
    h = build_hierarchy(meta)
    hist = np.array([[2, 2], [4, 4], [6, 6], [8, 8]], dtype=float)  # means 2,4,6,8
    p = historical_proportions(hist)
    assert np.allclose(p, [0.1, 0.2, 0.3, 0.4])
    base = np.zeros((h.n_nodes, 1))
    base[h.level_slices["total"]] = 100.0
    rec = top_down(base, h, p)
    assert coherence_error(rec, h) < 1e-9
    assert rec[h.level_slices["total"]][0, 0] == pytest.approx(100.0)  # total preserved
    assert np.allclose(rec[h.bottom_slice][:, 0], [10, 20, 30, 40])    # split by proportions


def test_mint_is_coherent_and_reduces_to_bu(meta):
    h = build_hierarchy(meta)
    rng = np.random.default_rng(1)
    base = rng.uniform(1, 10, size=(h.n_nodes, 2))
    # MinT with huge variance on aggregates and tiny on bottom -> trusts only
    # the bottom -> must equal bottom-up
    w = np.full(h.n_nodes, 1e6)
    w[h.bottom_slice] = 1e-6
    rec = mint(base, h, w)
    assert coherence_error(rec, h) < 1e-7
    assert np.allclose(rec, bottom_up(base, h), atol=1e-3)


def test_leaf_rows_align_with_columns_when_id_order_differs():
    # ids deliberately sort DIFFERENTLY from "item|store" keys: id "z1".."z4"
    # vs keys "B|Y","A|Y",... — catches leaf rows sorted by key not by column
    rows = [
        ("z1", "B", "D1", "C1", "Y", "ST"),
        ("z2", "A", "D1", "C1", "Y", "ST"),
        ("z3", "B", "D1", "C1", "X", "ST"),
        ("z4", "A", "D1", "C1", "X", "ST"),
    ]
    meta = pd.DataFrame(rows, columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
    h = build_hierarchy(meta)
    # leaf block is the identity in column (id) order
    assert np.allclose(h.S[h.bottom_slice].toarray(), np.eye(4))
    # a reconciled (coherent) vector must have zero coherence error
    b = np.array([[5.0], [1.0], [9.0], [3.0]])  # keyed to z1..z4 = bottom_ids order
    allv = h.aggregate(b)
    from m5forecast.hierarchy.reconciliation import coherence_error
    assert coherence_error(allv, h) < 1e-9
    # bottom-up recovers the exact bottom values in id order
    from m5forecast.hierarchy.reconciliation import bottom_up
    rec = bottom_up(allv, h)
    assert np.allclose(rec[h.bottom_slice], b)


def test_full_m5_hierarchy_counts():
    # synthesize the real M5 cardinalities: 3049 items x 10 stores (3 states)
    stores = [("CA_1", "CA"), ("CA_2", "CA"), ("CA_3", "CA"), ("CA_4", "CA"),
              ("TX_1", "TX"), ("TX_2", "TX"), ("TX_3", "TX"),
              ("WI_1", "WI"), ("WI_2", "WI"), ("WI_3", "WI")]
    cats = {"FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
            "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
            "HOBBIES": ["HOBBIES_1", "HOBBIES_2"]}
    # a couple items per department so counts are structurally right (not 3049)
    rows = []
    for cat, depts in cats.items():
        for dept in depts:
            for k in range(2):
                item = f"{dept}_{k:03d}"
                for store, state in stores:
                    rows.append((f"{item}_{store}", item, dept, cat, store, state))
    meta = pd.DataFrame(rows, columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"])
    h = build_hierarchy(meta)
    counts = {name: (sl.stop - sl.start) for name, sl in h.level_slices.items()}
    assert counts["total"] == 1
    assert counts["state"] == 3
    assert counts["store"] == 10
    assert counts["category"] == 3
    assert counts["department"] == 7
    assert counts["state_category"] == 9
    assert counts["state_department"] == 21
    assert counts["store_category"] == 30
    assert counts["store_department"] == 70
    # 12 levels present
    assert len(h.level_slices) == 12
