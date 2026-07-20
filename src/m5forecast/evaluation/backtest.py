"""Expanding-window backtest folds, aligned to the M5 split.

Layout (horizon 28, last_train_day 1913, n_folds 3):

    fold 1: train d1..1829, test d1830..1857
    fold 2: train d1..1857, test d1858..1885
    fold 3: train d1..1885, test d1886..1913

The final M5 evaluation block d1914..1941 is NOT produced here — it is
reserved for exactly one touch at the very end (Phase 13+). All model
selection and tuning happens on these folds only.

This module is the ONLY place train/test boundaries are computed
(Phase 4 rule): every model sees identical folds.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_end: int           # last day index included in training
    test_start: int
    test_end: int

    @property
    def test_days(self) -> range:
        return range(self.test_start, self.test_end + 1)


def expanding_folds(n_folds: int, horizon: int, last_train_day: int) -> list[Fold]:
    """Contiguous horizon-sized test blocks ending at last_train_day, oldest first."""
    folds = []
    for i in range(n_folds, 0, -1):
        test_end = last_train_day - (n_folds - i) * horizon
        test_start = test_end - horizon + 1
        folds.append(Fold(fold_id=i, train_end=test_start - 1, test_start=test_start, test_end=test_end))
    folds.sort(key=lambda f: f.fold_id)
    return folds
