from __future__ import annotations

import math
from typing import Any

SECOND_NS = 1_000_000_000


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def compute_markout_value(
    *,
    fill_price: float,
    fill_size: float,
    fill_side: str,
    future_mid: float,
) -> float:
    if str(fill_side).lower() == "buy":
        markout = (float(future_mid) - float(fill_price)) * float(fill_size)
    else:
        markout = (float(fill_price) - float(future_mid)) * float(fill_size)
    return round(markout, 6)


def compute_markout(
    *,
    fill_price: float,
    fill_size: float,
    fill_side: str,
    token_side: str,
    future_mids: dict[int, float | None],
    fill_ts_ns: int,
    market_close_ts_ns: int,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for horizon_s, future_mid in future_mids.items():
        target_ts_ns = int(fill_ts_ns) + int(horizon_s) * SECOND_NS
        key = f"markout_{int(horizon_s)}s"
        if target_ts_ns > int(market_close_ts_ns) or future_mid is None or _is_nan(future_mid):
            results[key] = float("nan")
            continue
        results[key] = compute_markout_value(
            fill_price=fill_price,
            fill_size=fill_size,
            fill_side=fill_side,
            future_mid=float(future_mid),
        )
    return results


def compute_post_fill_adverse_flag(
    *,
    fill_side: str,
    post_fill_mid_move_1s: float,
    threshold: float = 0.005,
) -> bool:
    if _is_nan(post_fill_mid_move_1s):
        return False
    if str(fill_side).lower() == "buy":
        return float(post_fill_mid_move_1s) < -float(threshold)
    return float(post_fill_mid_move_1s) > float(threshold)
