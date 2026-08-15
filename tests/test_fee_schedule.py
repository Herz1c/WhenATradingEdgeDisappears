from __future__ import annotations

import pandas as pd

from polymarket_recorder.execution_fee_schedule import (
    compute_maker_fee,
    compute_maker_rebate,
    compute_taker_fee,
)


def test_fee_formula_post_march_30_crypto_schedule() -> None:
    deploy_ts = int(pd.Timestamp("2026-04-19", tz="UTC").timestamp() * 1e9)

    test_cases = [
        (0.10, 100, 0.06),
        (0.25, 100, 0.34),
        (0.50, 100, 0.90),
        (0.75, 100, 1.01),
        (0.90, 100, 0.58),
    ]
    for price, size, expected_fee in test_cases:
        computed = compute_taker_fee(price, size, deploy_ts)
        assert abs(computed - expected_fee) <= 0.01

    assert compute_maker_fee(0.50, 100) == 0.0

    taker_fee = compute_taker_fee(0.50, 100, deploy_ts)
    rebate = compute_maker_rebate(taker_fee, deploy_ts)
    assert abs(rebate - taker_fee * 0.20) < 0.001


def test_fee_formula_pre_march_30_schedule_peak() -> None:
    deploy_ts = int(pd.Timestamp("2026-03-29", tz="UTC").timestamp() * 1e9)
    computed = compute_taker_fee(0.50, 100, deploy_ts)
    assert abs(computed - 0.7812) < 0.0001
