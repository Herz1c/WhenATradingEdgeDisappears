from __future__ import annotations

from typing import Any

import polars as pl


"""
asof_join_lib_v1: causal as-of join library.

All joins in this codebase must go through this library.
Direct manual as-of joins are forbidden.

Rules enforced by every join function:
- recv_ts_ns is the only join clock
- No future data: right table rows with ts > left table ts are excluded
- Result is deterministic given same input
- Missing right-side rows produce null, never raise
"""


def _require_recv_clock(left_ts_col: str, right_ts_col: str) -> None:
    if left_ts_col != "recv_ts_ns" or right_ts_col != "recv_ts_ns":
        raise ValueError(
            "asof_join_lib_v1: recv_ts_ns is the only permitted shared join clock."
        )


def _empty_join_result(left: pl.DataFrame, right: pl.DataFrame, suffix: str) -> pl.DataFrame:
    result = left.clone()
    for column, dtype in right.schema.items():
        if column in {"recv_ts_ns"}:
            continue
        output_column = f"{column}{suffix}" if column in result.columns else column
        if output_column not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=dtype).alias(output_column))
    return result


def asof_join_left_causal(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_ts_col: str,
    right_ts_col: str,
    by: list[str] | None = None,
    suffix: str = "_right",
    strategy: str = "backward",
    tolerance_ns: int | None = None,
) -> pl.DataFrame:
    """
    Causal as-of join: for each row in left, find most recent row in right
    where right_ts_col <= left_ts_col.
    """
    if strategy != "backward":
        raise ValueError(
            "asof_join_lib_v1: only backward (causal) strategy is permitted. "
            "Forward joins would introduce future data leakage."
        )
    _require_recv_clock(left_ts_col, right_ts_col)
    if left.is_empty():
        return left.clone()
    if right.is_empty():
        return _empty_join_result(left, right, suffix)

    by = by or []
    sort_cols_left = [*by, left_ts_col]
    sort_cols_right = [*by, right_ts_col]
    left_sorted = left.sort(sort_cols_left)
    right_sorted = right.sort(sort_cols_right)
    kwargs: dict[str, Any] = {
        "left_on": left_ts_col,
        "right_on": right_ts_col,
        "strategy": "backward",
        "suffix": suffix,
    }
    if by:
        kwargs["by"] = by
    if tolerance_ns is not None:
        kwargs["tolerance"] = int(tolerance_ns)
    return left_sorted.join_asof(right_sorted, **kwargs)


def asof_join_live_reference(
    df: pl.DataFrame,
    live_ref: pl.DataFrame,
    df_ts_col: str = "recv_ts_ns",
    tolerance_ns: int | None = None,
) -> pl.DataFrame:
    """
    Standard join for live_reference_events_v1. Always causal backward.
    Joins on a recv_ts_ns materialization of the 1-second grid.
    """
    if df_ts_col != "recv_ts_ns":
        raise ValueError("asof_join_live_reference requires df_ts_col='recv_ts_ns'.")
    if "recv_ts_ns" in live_ref.columns:
        right = live_ref
    elif "ts_seconds" in live_ref.columns:
        right = live_ref.with_columns((pl.col("ts_seconds").cast(pl.Int64) * 1_000_000_000).alias("recv_ts_ns"))
    else:
        raise ValueError("live_reference_events_v1 must contain recv_ts_ns or ts_seconds.")
    return asof_join_left_causal(
        df,
        right,
        left_ts_col="recv_ts_ns",
        right_ts_col="recv_ts_ns",
        suffix="_live_ref",
        tolerance_ns=tolerance_ns,
    )


def verify_causal_integrity(
    joined_df: pl.DataFrame,
    left_ts_col: str,
    right_ts_col: str,
    strict: bool = False,
) -> dict[str, float | int]:
    """
    Verify joined DataFrame has no future right-side rows.
    """
    if left_ts_col not in joined_df.columns or right_ts_col not in joined_df.columns:
        return {"n_violations": 0, "violation_rate": 0.0, "max_violation_ns": 0}
    violations = joined_df.filter(
        pl.col(right_ts_col).is_not_null() & (pl.col(right_ts_col) > pl.col(left_ts_col))
    )
    n_violations = int(violations.height)
    total = max(int(joined_df.height), 1)
    max_violation = 0
    if n_violations:
        max_violation = int((violations[right_ts_col] - violations[left_ts_col]).max())
    if strict and n_violations:
        raise ValueError(f"asof_join_lib_v1: detected {n_violations} future-data violations.")
    return {
        "n_violations": n_violations,
        "violation_rate": float(n_violations / total),
        "max_violation_ns": max_violation,
    }

