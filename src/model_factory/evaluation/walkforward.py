from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl

DAY_NS = 24 * 3600 * 1_000_000_000


@dataclass(slots=True)
class AnchoredWalkForwardEvaluator:
    """
    Anchored walk-forward cross-validation.
    """

    ordering_col: str
    purge_days: int = 1
    embargo_days: int = 0
    min_train_days: int = 3

    def split(self, df: pl.DataFrame) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
        if self.ordering_col not in df.columns:
            raise ValueError(f"Missing ordering column: {self.ordering_col}")
        frame = df.sort(self.ordering_col)
        day_numbers = frame.select((pl.col(self.ordering_col) // DAY_NS).unique().sort()).to_series().to_list()
        folds: list[tuple[pl.DataFrame, pl.DataFrame]] = []
        gap = self.purge_days + self.embargo_days
        for val_index in range(self.min_train_days + gap, len(day_numbers)):
            train_days = day_numbers[: val_index - gap]
            val_day = day_numbers[val_index]
            train = frame.filter((pl.col(self.ordering_col) // DAY_NS).is_in(train_days))
            val = frame.filter((pl.col(self.ordering_col) // DAY_NS) == val_day)
            if train.is_empty() or val.is_empty():
                continue
            max_train = train[self.ordering_col].max()
            min_val = val[self.ordering_col].min()
            if max_train >= min_val:
                raise ValueError("no_future violation: max(train_ts) >= min(val_ts)")
            folds.append((train, val))
        return folds

    def evaluate_across_folds(
        self,
        model_fn: Callable,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str,
        metrics_fn: Callable,
    ) -> dict:
        fold_metrics = []
        for train, val in self.split(df):
            model = model_fn(train.select(feature_cols), train[target_col])
            predictions = model(val.select(feature_cols))
            fold_metrics.append(metrics_fn(val[target_col].to_numpy(), predictions))
        values = [float(item) for item in fold_metrics]
        return {
            "folds": fold_metrics,
            "mean": sum(values) / len(values) if values else None,
            "std": (sum((v - (sum(values) / len(values))) ** 2 for v in values) / len(values)) ** 0.5 if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

