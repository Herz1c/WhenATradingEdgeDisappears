"""Sanity tests for grading function."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from backtest.grading import grade_strategy, DAILY_LOSS_CAP_USD


def _week(pnl, dd, worst_day):
    return {"net_pnl_usd": pnl, "max_drawdown_usd": dd, "worst_day_pnl_usd": worst_day}


def test_all_positive_no_disqualifications():
    weeks = [_week(10, 2, -1) for _ in range(20)]
    g = grade_strategy(weeks)
    assert not g["rejected"]
    assert g["n_weeks_disqualified"] == 0
    # mean calmar = 10/2 = 5; med_pnl = 10 > 0 so no penalty
    assert abs(g["mean_calmar_clean"] - 5.0) < 1e-9


def test_disqualification_floor():
    """If > 20% of weeks have worst-day < -$10, strategy is rejected."""
    weeks = [_week(50, 10, -20) for _ in range(15)] + [_week(50, 10, -5) for _ in range(5)]
    g = grade_strategy(weeks)
    assert g["rejected"], "15/20 = 75% disqualified should be rejected"
    assert g["disqualification_rate"] == 0.75


def test_negative_median_penalty():
    """Negative median weekly PnL adds -1.0 * |median| penalty."""
    # 10 weeks of +5, 10 weeks of -10 -> median = -2.5
    weeks = [_week(5, 1, 0) for _ in range(10)] + [_week(-10, 5, -3) for _ in range(10)]
    g = grade_strategy(weeks)
    assert not g["rejected"]
    assert g["median_net_pnl_usd"] == -2.5
    # mean calmar across non-disqualified: positive weeks have calmar 5; neg weeks -2
    # = (10*5 + 10*-2)/20 = 30/20 = 1.5
    # penalty = 1.0 * 2.5 = 2.5
    # disqual rate = 0
    # score = 1.5 - 0 - 2.5 = -1.0
    assert abs(g["score"] - (-1.0)) < 1e-9


def test_disqual_rate_reduces_score_by_half_per_unit():
    """One disqualifying week in 20 reduces score by exactly 0.5/20 = 0.025."""
    base = [_week(10, 2, -1) for _ in range(20)]  # all clean
    g_base = grade_strategy(base)
    bad = list(base)
    bad[0] = _week(10, 2, -50)  # disqualifying
    g_bad = grade_strategy(bad)
    # disqual_rate goes from 0 to 1/20, score drops by 0.5 * 1/20 = 0.025
    # AND that week is excluded from mean_calmar — but since all weeks have same
    # calmar (5), mean_calmar_clean doesn't change. So delta should be ~-0.025.
    assert abs((g_base["score"] - g_bad["score"]) - 0.025) < 1e-9


if __name__ == "__main__":
    test_all_positive_no_disqualifications()
    test_disqualification_floor()
    test_negative_median_penalty()
    test_disqual_rate_reduces_score_by_half_per_unit()
    print("all grading tests passed")
