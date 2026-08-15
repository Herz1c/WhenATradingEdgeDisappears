"""Calmar-style strategy grade with per-week disqualification.

Per-week metrics:
    net_pnl_w        = ending - starting
    max_dd_w         = max(peak - balance) over the week, USD
    worst_day_w      = min daily PnL
    calmar_w         = net_pnl_w / max(max_dd_w, 1)
    disqualified_w   = (worst_day_w < -DAILY_LOSS_CAP_USD)

Aggregate score:
    score = mean(calmar_w | not disqualified) - 0.5 * disqualification_rate
            - 1.0 * max(0, -median(net_pnl_w))

Hard floor: disqualification_rate > 0.20 -> rejected.
"""
from __future__ import annotations

from statistics import mean, median
from typing import Iterable


DAILY_LOSS_CAP_USD = 10.0   # > $10/day loss on a $100 bankroll = disqualifying
MAX_DISQUAL_RATE = 0.20     # > 10/50 weeks blown = strategy rejected


def grade_strategy(per_week: list[dict],
                   daily_loss_cap_usd: float = DAILY_LOSS_CAP_USD,
                   max_disqual_rate: float = MAX_DISQUAL_RATE) -> dict:
    """Return {score, rejected, ...diagnostics} from a list of week summaries.

    Each per-week dict is a PnLTracker.summary() (contains net_pnl_usd,
    max_drawdown_usd, daily_pnl, worst_day_pnl_usd).
    """
    if not per_week:
        return {"score": -1e9, "rejected": True, "reason": "no weeks"}

    weeks_pnl: list[float] = []
    weeks_dd: list[float] = []
    weeks_calmar: list[float] = []
    n_disqual = 0

    for w in per_week:
        pnl = float(w.get("net_pnl_usd", 0.0))
        dd = float(w.get("max_drawdown_usd", 0.0))
        worst = float(w.get("worst_day_pnl_usd", 0.0))
        weeks_pnl.append(pnl)
        weeks_dd.append(dd)
        calmar = pnl / max(dd, 1.0)
        weeks_calmar.append(calmar)
        if worst < -daily_loss_cap_usd:
            n_disqual += 1

    disqual_rate = n_disqual / len(per_week)
    rejected = disqual_rate > max_disqual_rate

    # Mean calmar excluding disqualified weeks
    keep = [
        c for c, w in zip(weeks_calmar, per_week)
        if float(w.get("worst_day_pnl_usd", 0.0)) >= -daily_loss_cap_usd
    ]
    mean_calmar_clean = mean(keep) if keep else -1e9
    med_pnl = median(weeks_pnl)
    mean_pnl = mean(weeks_pnl)
    p10_pnl = _percentile(weeks_pnl, 10)
    p90_pnl = _percentile(weeks_pnl, 90)
    p90_dd = _percentile(weeks_dd, 90)

    score = mean_calmar_clean - 0.5 * disqual_rate - 1.0 * max(0.0, -med_pnl)
    if rejected:
        score = -1e6 + score  # keep ordering but mark as rejected

    return {
        "score": float(score),
        "rejected": rejected,
        "n_weeks": len(per_week),
        "n_weeks_disqualified": n_disqual,
        "disqualification_rate": disqual_rate,
        "mean_calmar_clean": float(mean_calmar_clean),
        "median_net_pnl_usd": float(med_pnl),
        "mean_net_pnl_usd": float(mean_pnl),
        "p10_net_pnl_usd": float(p10_pnl),
        "p90_net_pnl_usd": float(p90_pnl),
        "p90_max_drawdown_usd": float(p90_dd),
        "n_weeks_with_positive_pnl": sum(1 for x in weeks_pnl if x > 0),
    }


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)
