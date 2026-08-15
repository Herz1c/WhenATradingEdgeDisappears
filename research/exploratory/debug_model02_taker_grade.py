#!/usr/bin/env python3
"""Official 20-MC-week grade of the Model 02 taker strategy with proper lifecycle.

Two configurations:
  A) DOWN-side only (per calibration audit: most profitable)
  B) BOTH sides (UP + DOWN)
"""
import io, sys, json, random, statistics
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

from backtest.model02_taker_engine import run_taker_backtest
from backtest.grading import grade_strategy

OOS_POOL = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10",
            "2026-05-11","2026-05-12","2026-05-13","2026-05-14",
            "2026-05-15","2026-05-16"]

configs = [
    ("DOWN only, edge≥$0.05, ttc 10-60",
     dict(edge_threshold_usd=0.05, allow_up_side=False, allow_dn_side=True,
          min_t_to_close_s=10, max_t_to_close_s=60)),
    ("DOWN only, edge≥$0.10, ttc 10-60",
     dict(edge_threshold_usd=0.10, allow_up_side=False, allow_dn_side=True,
          min_t_to_close_s=10, max_t_to_close_s=60)),
    ("DOWN only, edge≥$0.20, ttc 10-60",
     dict(edge_threshold_usd=0.20, allow_up_side=False, allow_dn_side=True,
          min_t_to_close_s=10, max_t_to_close_s=60)),
    ("BOTH sides, edge≥$0.10, ttc 10-60",
     dict(edge_threshold_usd=0.10, allow_up_side=True, allow_dn_side=True,
          min_t_to_close_s=10, max_t_to_close_s=60)),
    ("BOTH sides, edge≥$0.20, ttc 10-60",
     dict(edge_threshold_usd=0.20, allow_up_side=True, allow_dn_side=True,
          min_t_to_close_s=10, max_t_to_close_s=60)),
    ("DOWN only, edge≥$0.10, ttc 30-90",
     dict(edge_threshold_usd=0.10, allow_up_side=False, allow_dn_side=True,
          min_t_to_close_s=30, max_t_to_close_s=90)),
]

# 20 MC weeks from OOS pool
weeks = []
for i in range(20):
    r = random.Random(3000 + i)
    weeks.append(sorted(r.sample(OOS_POOL, 7)))

print("=" * 95)
print(f"MODEL 02 TAKER STRATEGY — 20 MC WEEKS, OOS POOL (May 7-16)")
print("=" * 95)

all_results = {}
for label, cfg in configs:
    print(f"\n--- {label} ---")
    per_week = []
    for i, w in enumerate(weeks):
        s = run_taker_backtest(w, max_per_day=15_000, max_margin_usd=100.0, **cfg)
        per_week.append(s)
        if i < 5 or i == len(weeks) - 1:   # print first 5 and last
            print(f"  w{i+1:>2}: fills={s['n_fills']:>5}  win%={s['win_rate']*100:>5.1f}  "
                  f"pnl=${s['net_pnl_usd']:+8.2f}  dd=${s['max_drawdown_usd']:5.2f}  "
                  f"worst=${s['worst_day_pnl_usd']:+6.2f}")
        elif i == 5:
            print(f"  ...")
    g = grade_strategy(per_week)
    pnls = [w['net_pnl_usd'] for w in per_week]
    print(f"  GRADE: score={g['score']:+.2f}  rejected={g['rejected']}  "
          f"mean=${g['mean_net_pnl_usd']:+.2f}  med=${g['median_net_pnl_usd']:+.2f}  "
          f"calmar={g['mean_calmar_clean']:.1f}  pos={g['n_weeks_with_positive_pnl']}/{g['n_weeks']}  "
          f"disq={g['disqualification_rate']*100:.0f}%")
    print(f"         min=${min(pnls):+.2f}  max=${max(pnls):+.2f}  std=${statistics.stdev(pnls):.2f}")
    all_results[label] = {"grade": g, "per_week": per_week}

# Save
Path("docs").mkdir(exist_ok=True)
Path("docs/model02_taker_grade.json").write_text(json.dumps(all_results, indent=2, default=str))
print(f"\nSaved to docs/model02_taker_grade.json")
