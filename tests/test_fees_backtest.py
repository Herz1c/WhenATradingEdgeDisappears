"""Sanity tests for backtest fee calculator."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from backtest.fees import FeeCalculator  # noqa: E402


def test_taker_fee_at_p50_post_schedule():
    """Post-2026-03-30: taker_fee = size * 0.072 * (p*(1-p))^1; at p=0.5 = 0.018/contract."""
    fc = FeeCalculator.for_date("2026-05-09")
    fee = fc.taker_fee_usd(price=0.50, size=1.0)
    assert abs(fee - 0.018) < 1e-9, f"expected 0.018, got {fee}"


def test_maker_rebate_is_20pct_of_taker():
    fc = FeeCalculator.for_date("2026-05-09")
    rebate = fc.maker_rebate_usd(price=0.50, size=1.0)
    assert abs(rebate - 0.0036) < 1e-9, f"expected 0.0036, got {rebate}"


def test_fee_at_extreme_price_smaller():
    """At p=0.10, p*(1-p)=0.09, fee = 0.072*0.09 = 0.00648/contract."""
    fc = FeeCalculator.for_date("2026-05-09")
    fee_low = fc.taker_fee_usd(price=0.10, size=1.0)
    fee_high = fc.taker_fee_usd(price=0.90, size=1.0)
    assert abs(fee_low - 0.00648) < 1e-9
    assert abs(fee_high - 0.00648) < 1e-9  # symmetric


def test_fee_scaling_for_stress_test():
    fc = FeeCalculator.for_date("2026-05-09", fee_scale=1.2)
    fee = fc.taker_fee_usd(price=0.50, size=1.0)
    assert abs(fee - 0.018 * 1.2) < 1e-9


def test_maker_fee_is_zero():
    fc = FeeCalculator.for_date("2026-05-09")
    assert fc.maker_fee_usd(price=0.50, size=1.0) == 0.0


if __name__ == "__main__":
    test_taker_fee_at_p50_post_schedule()
    test_maker_rebate_is_20pct_of_taker()
    test_fee_at_extreme_price_smaller()
    test_fee_scaling_for_stress_test()
    test_maker_fee_is_zero()
    print("all fee tests passed")
