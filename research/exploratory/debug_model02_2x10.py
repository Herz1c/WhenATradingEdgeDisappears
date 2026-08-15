#!/usr/bin/env python3
"""Compare: 1 entry/market vs 2 entries/market with min 10s gap, otherwise identical."""
import io, sys, json, random, statistics
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from backtest.model02_taker_engine import run_taker_backtest
from backtest.grading import grade_strategy

OOS_DATES = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10",
             "2026-05-11","2026-05-12","2026-05-13","2026-05-14",
             "2026-05-15","2026-05-16"]


def run_grade(label, cfg, n_weeks=20):
    weeks = []
    for i in range(n_weeks):
        r = random.Random(4000 + i)
        weeks.append(sorted(r.sample(OOS_DATES, 7)))
    per_week = []
    for w in weeks:
        s = run_taker_backtest(w, **cfg)
        per_week.append(s)
    g = grade_strategy(per_week)
    pnls = [w["net_pnl_usd"] for w in per_week]
    dds = [w["max_drawdown_usd"] for w in per_week]
    fills = [w["n_fills"] for w in per_week]
    worsts = [w["worst_day_pnl_usd"] for w in per_week]
    print(f"\n=== {label} ===")
    print(f"  fills/wk : mean={statistics.mean(fills):.0f}  min={min(fills)}  max={max(fills)}")
    print(f"  PnL/wk   : mean=${g['mean_net_pnl_usd']:+.2f}  med=${g['median_net_pnl_usd']:+.2f}  "
          f"min=${min(pnls):+.2f}  max=${max(pnls):+.2f}  std=${statistics.stdev(pnls):.2f}")
    print(f"  Max DD   : mean=${statistics.mean(dds):.2f}  max=${max(dds):.2f}")
    print(f"  Worst-day: mean=${statistics.mean(worsts):+.2f}  min=${min(worsts):+.2f}")
    print(f"  Positive : {g['n_weeks_with_positive_pnl']}/{g['n_weeks']}")
    print(f"  Calmar   : {g['mean_calmar_clean']:.2f}")
    return {"label": label, "config": cfg, "grade": g, "per_week": per_week, "weeks": weeks}


base = dict(
    edge_threshold_usd=0.10, allow_up_side=False, allow_dn_side=True,
    min_t_to_close_s=10, max_t_to_close_s=60,
    max_margin_usd=100.0, max_per_day=30_000,
)

r1 = run_grade("1 entry/market (current)", {**base, "max_positions_per_market": 1, "min_seconds_between_entries_per_market": 0})
r2 = run_grade("2 entries/market, 10s gap", {**base, "max_positions_per_market": 2, "min_seconds_between_entries_per_market": 10})
r3 = run_grade("2 entries/market, 30s gap", {**base, "max_positions_per_market": 2, "min_seconds_between_entries_per_market": 30})
r4 = run_grade("3 entries/market, 10s gap", {**base, "max_positions_per_market": 3, "min_seconds_between_entries_per_market": 10})

# Save
Path("docs").mkdir(exist_ok=True)
Path("docs/model02_taker_2x10_compare.json").write_text(json.dumps(
    [r1, r2, r3, r4], indent=2, default=str))
print(f"\nSaved to docs/model02_taker_2x10_compare.json")
