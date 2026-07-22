"""The 12 M5 aggregation levels as a summing matrix S.

A hierarchical forecasting system needs forecasts at every level a business
reads: total (finance), state (logistics), store/department (shelf space),
item-store (replenishment). M5 defines exactly 12 levels totalling 42,840
series over 30,490 bottom (item x store) series:

    L1  total                    1
    L2  state                    3
    L3  store                   10
    L4  category                 3
    L5  department               7
    L6  state x category         9
    L7  state x department      21
    L8  store x category        30
    L9  store x department      70
    L10 item                  3049
    L11 item x state          9147
    L12 item x store (bottom) 30490     <- what our models forecast
                             ------
                             42840

The **summing matrix** S is [42840 x 30490]: row a is the 0/1 indicator of
which bottom series sum into aggregate node a. Then for any bottom-level
vector b (forecast or actual), `S @ b` gives values at ALL levels at once.
The bottom level appears in S as identity rows, so S contains the whole
hierarchy including the leaves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

#: (level name, groupby columns). Order defines node numbering; L12 is the bottom.
LEVELS: list[tuple[str, list[str]]] = [
    ("total", []),
    ("state", ["state_id"]),
    ("store", ["store_id"]),
    ("category", ["cat_id"]),
    ("department", ["dept_id"]),
    ("state_category", ["state_id", "cat_id"]),
    ("state_department", ["state_id", "dept_id"]),
    ("store_category", ["store_id", "cat_id"]),
    ("store_department", ["store_id", "dept_id"]),
    ("item", ["item_id"]),
    ("item_state", ["item_id", "state_id"]),
    ("item_store", ["item_id", "store_id"]),  # bottom
]

#: The 9 aggregate levels with store x department (70 series) as the leaves.
#: Exact MinT needs an [n_leaf x n_leaf] solve, infeasible at the 30,490 bottom
#: but trivial here — used to demonstrate MinT where it is exactly computable.
UPPER_LEVELS: list[tuple[str, list[str]]] = LEVELS[:9]  # total ... store_department


@dataclass
class Hierarchy:
    S: sparse.csr_matrix          # [n_nodes, n_bottom]
    nodes: pd.DataFrame           # one row per node: level, key, global index
    level_slices: dict[str, slice]  # node-row range for each level
    bottom_ids: pd.Index          # ordered bottom-series ids (columns of S)
    leaf_level: str = "item_store"  # name of the level acting as leaves

    @property
    def n_nodes(self) -> int:
        return self.S.shape[0]

    @property
    def bottom_slice(self) -> slice:
        return self.level_slices[self.leaf_level]

    def aggregate(self, bottom: np.ndarray) -> np.ndarray:
        """Bottom-level values [n_bottom] or [n_bottom, H] -> all nodes."""
        return self.S @ bottom


def build_hierarchy(meta: pd.DataFrame, levels: list[tuple[str, list[str]]] = LEVELS) -> Hierarchy:
    """meta: one row per bottom (leaf) series with id + hierarchy columns.

    Rows are ordered by `id` so column j of S is leaf series `bottom_ids[j]`.
    The last entry of `levels` defines the leaves (its `bottom_slice`).
    """
    meta = meta.drop_duplicates("id").sort_values("id").reset_index(drop=True)
    n_bottom = len(meta)
    bottom_col = np.arange(n_bottom)
    leaf_level = levels[-1][0]

    rows, cols, node_records = [], [], []
    node_idx = 0
    level_slices: dict[str, slice] = {}
    ids = meta["id"].to_numpy()
    for name, keys in levels:
        start = node_idx
        if name == leaf_level:
            # leaves MUST be identity rows aligned to the columns (bottom_ids
            # order), or bottom-level indexing / coherence / bottom-up scramble
            groups = [(ids[c], np.array([c])) for c in bottom_col]
        elif not keys:  # total: one node summing every bottom series
            groups = [("__total__", bottom_col)]
        else:
            key = meta[keys].astype(str).agg("|".join, axis=1) if len(keys) > 1 else meta[keys[0]].astype(str)
            groups = [(k, bottom_col[idx.to_numpy()]) for k, idx in key.groupby(key).groups.items()]
            groups.sort(key=lambda kv: kv[0])
        for node_key, member_cols in groups:
            rows.extend([node_idx] * len(member_cols))
            cols.extend(member_cols.tolist())
            node_records.append({"level": name, "key": node_key, "node_idx": node_idx})
            node_idx += 1
        level_slices[name] = slice(start, node_idx)

    S = sparse.csr_matrix((np.ones(len(rows), dtype="float64"), (rows, cols)), shape=(node_idx, n_bottom))
    nodes = pd.DataFrame(node_records)
    return Hierarchy(S=S, nodes=nodes, level_slices=level_slices,
                     bottom_ids=pd.Index(meta["id"]), leaf_level=leaf_level)
