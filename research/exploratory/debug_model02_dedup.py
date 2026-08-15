#!/usr/bin/env python3
"""Re-grade Model 02 DOWN strategy with per-market position dedup.

Before fix: each snapshot was a fresh trade → trading same market 50-300x.
After fix: max 1 position per (market, side) → realistic ~150 trades/day max.
"""
import io, sys, json, statistics, random
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

print("=" * 78)
print("MODEL 02 DOWN STRATEGY — DEDUPED (max 1 position per (market, side))")
print("=" * 78)

print(f"\n--- TEST A: Full OOS window, vary edge threshold ---")
print(f"{'edge':>6} {'fills':>7} {'win%':>7} {'pnl':>10} {'dd':>8} {'per-fill':>11}")
for thr in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    s = run_taker_backtest(
        OOS_DATES, edge_threshold_usd=thr,
        allow_up_side=False, allow_dn_side=True,
        min_t_to_close_s=10, max_t_to_close_s=60,
        max_margin_usd=100.0, max_per_day=30_000,   # bigger sample, less subsampling artifact
        max_positions_per_market=1,
    )
    pf = s['net_pnl_usd'] / s['n_fills'] if s['n_fills'] else 0
    print(f"≥${thr:.2f} {s['n_fills']:>7} {s['win_rate']*100:>6.1f}% "
          f"${s['net_pnl_usd']:>+8.2f} ${s['max_drawdown_usd']:>+6.2f} ${pf:>+10.4f}")

print(f"\n--- TEST B: Per-day breakdown (edge≥$0.10) ---")
print(f"{'date':<14} {'fills':>7} {'win%':>7} {'pnl':>10} {'dd':>8}")
for d in OOS_DATES:
    s = run_taker_backtest(
        [d], edge_threshold_usd=0.10,
        allow_up_side=False, allow_dn_side=True,
        min_t_to_close_s=10, max_t_to_close_s=60,
        max_margin_usd=100.0, max_per_day=30_000,
        max_positions_per_market=1,
    )
    print(f"{d:<14} {s['n_fills']:>7} {s['win_rate']*100:>6.1f}% "
          f"${s['net_pnl_usd']:>+8.2f} ${s['max_drawdown_usd']:>+6.2f}")

print(f"\n--- TEST C: 20 MC weeks, official grade (edge≥$0.10) ---")
weeks = []
for i in range(20):
    r = random.Random(4000 + i)
    weeks.append(sorted(r.sample(OOS_DATES, 7)))

per_week = []
for i, w in enumerate(weeks):
    s = run_taker_backtest(
        w, edge_threshold_usd=0.10,
        allow_up_side=False, allow_dn_side=True,
        min_t_to_close_s=10, max_t_to_close_s=60,
        max_margin_usd=100.0, max_per_day=30_000,
        max_positions_per_market=1,
    )
    per_week.append(s)
    if i < 5 or i == 19:
        print(f"  w{i+1:>2}: fills={s['n_fills']:>5}  win%={s['win_rate']*100:>5.1f}  "
              f"pnl=${s['net_pnl_usd']:+8.2f}  dd=${s['max_drawdown_usd']:5.2f}  "
              f"worst=${s['worst_day_pnl_usd']:+6.2f}")
    elif i == 5:
        print("  ...")
g = grade_strategy(per_week)
pnls = [w['net_pnl_usd'] for w in per_week]
print(f"\n  GRADE: score={g['score']:+.2f}  rejected={g['rejected']}")
print(f"         mean=${g['mean_net_pnl_usd']:+.2f}  med=${g['median_net_pnl_usd']:+.2f}  "
      f"min=${min(pnls):+.2f}  max=${max(pnls):+.2f}")
print(f"         Calmar={g['mean_calmar_clean']:.2f}  pos={g['n_weeks_with_positive_pnl']}/{g['n_weeks']}  "
      f"disq={g['disqualification_rate']*100:.0f}%")

Path("docs").mkdir(exist_ok=True)
Path("docs/model02_taker_deduped_grade.json").write_text(json.dumps({
    "grade": g, "per_week": per_week, "weeks": weeks,
}, indent=2, default=str))
print(f"\nSaved to docs/model02_taker_deduped_grade.json")
