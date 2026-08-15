from __future__ import annotations

from polymarket_recorder.execution_markout import compute_markout, compute_post_fill_adverse_flag


def test_markout_computation_is_size_scaled_for_buys_and_sells() -> None:
    t0 = 1_000_000_000
    markouts = compute_markout(
        fill_price=0.48,
        fill_size=100,
        fill_side="buy",
        token_side="UP",
        future_mids={1: 0.485, 3: 0.490, 5: 0.475},
        fill_ts_ns=t0,
        market_close_ts_ns=t0 + 300_000_000_000,
    )

    assert abs(markouts["markout_1s"] - 0.50) < 0.001
    assert abs(markouts["markout_3s"] - 1.00) < 0.001
    assert abs(markouts["markout_5s"] - (-0.50)) < 0.001

    markouts_sell = compute_markout(
        fill_price=0.52,
        fill_size=100,
        fill_side="sell",
        token_side="UP",
        future_mids={1: 0.515},
        fill_ts_ns=t0,
        market_close_ts_ns=t0 + 300_000_000_000,
    )
    assert abs(markouts_sell["markout_1s"] - 0.50) < 0.001


def test_markout_horizon_after_close_is_nan() -> None:
    t0 = 1_000_000_000
    markouts = compute_markout(
        fill_price=0.48,
        fill_size=100,
        fill_side="buy",
        token_side="UP",
        future_mids={1: 0.485},
        fill_ts_ns=t0,
        market_close_ts_ns=t0 + 500_000_000,
    )
    assert markouts["markout_1s"] != markouts["markout_1s"]


def test_post_fill_adverse_flag_direction() -> None:
    assert compute_post_fill_adverse_flag(fill_side="buy", post_fill_mid_move_1s=-0.006)
    assert not compute_post_fill_adverse_flag(fill_side="buy", post_fill_mid_move_1s=0.006)
    assert compute_post_fill_adverse_flag(fill_side="sell", post_fill_mid_move_1s=0.006)
    assert not compute_post_fill_adverse_flag(fill_side="sell", post_fill_mid_move_1s=-0.006)
