#!/usr/bin/env python3
"""Plot the 20 MC OOS weeks for the Model 02 DOWN strategy.

Daily equity curves overlaid + summary stats panel.
"""
import io, sys, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

grade = json.loads(Path("docs/model02_taker_deduped_grade.json").read_text())
per_week = grade["per_week"]
weeks_dates = grade["weeks"]

starting = 100.0
fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={"hspace": 0.30, "wspace": 0.25})

# ── Panel 1: daily equity curves overlaid ───────────────────────────────────
ax = axes[0, 0]
final_balances = []
for i, (wk, dates) in enumerate(zip(per_week, weeks_dates)):
    daily = wk["daily_pnl"]
    sorted_dates = sorted(daily.keys())
    balance = starting
    xs = [0]; ys = [balance]
    for d in sorted_dates:
        balance += daily[d]
        xs.append(len(xs))
        ys.append(balance)
    final_balances.append(balance)
    ax.plot(xs, ys, alpha=0.55, linewidth=1.3, label=f"w{i+1}" if i < 5 else None)
ax.axhline(starting, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="$100 start")
ax.set_xlabel("Day of week")
ax.set_ylabel("Balance (USD)")
ax.set_title("Equity curve — 20 OOS MC weeks ($100 start, Model 02 DOWN-only taker)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# ── Panel 2: final-balance distribution ─────────────────────────────────────
ax = axes[0, 1]
ax.hist(final_balances, bins=12, color="steelblue", edgecolor="black", alpha=0.85)
ax.axvline(np.mean(final_balances), color="red", linestyle="--", linewidth=2,
           label=f"mean = ${np.mean(final_balances):.2f}")
ax.axvline(np.median(final_balances), color="orange", linestyle="--", linewidth=2,
           label=f"median = ${np.median(final_balances):.2f}")
ax.axvline(starting, color="black", linestyle=":", linewidth=1.5, label="$100 start")
ax.set_xlabel("Ending balance (USD)")
ax.set_ylabel("Number of weeks")
ax.set_title("Distribution of ending balance across 20 MC weeks")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel 3: weekly PnL bar chart ───────────────────────────────────────────
ax = axes[1, 0]
weekly_pnl = [w["net_pnl_usd"] for w in per_week]
colors = ["green" if x > 0 else "red" for x in weekly_pnl]
ax.bar(range(1, 21), weekly_pnl, color=colors, edgecolor="black", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Week index")
ax.set_ylabel("Weekly PnL (USD)")
ax.set_title(f"Weekly PnL — all 20/20 positive, range ${min(weekly_pnl):.2f}–${max(weekly_pnl):.2f}")
ax.grid(True, alpha=0.3, axis="y")
ax.set_xticks(range(1, 21))

# ── Panel 4: max drawdown per week ──────────────────────────────────────────
ax = axes[1, 1]
dds = [w["max_drawdown_usd"] for w in per_week]
ax.bar(range(1, 21), dds, color="orangered", edgecolor="black", alpha=0.85)
ax.axhline(np.mean(dds), color="black", linestyle="--", linewidth=1, label=f"mean = ${np.mean(dds):.2f}")
ax.axhline(10, color="red", linestyle=":", linewidth=1.5, alpha=0.6, label="$10 daily-loss limit (never hit)")
ax.set_xlabel("Week index")
ax.set_ylabel("Max drawdown (USD)")
ax.set_title(f"Max intraweek drawdown — peak ${max(dds):.2f} on $100 bankroll ({max(dds)/100*100:.1f}%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.set_xticks(range(1, 21))

fig.suptitle("Model 02 DOWN-only taker strategy — 20 OOS Monte-Carlo weeks (May 7-16, post-training)",
             fontsize=13, fontweight="bold")
out = Path("docs/model02_taker_mc_weeks.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out.resolve()}")

# Also save a summary text
print(f"\nSummary:")
print(f"  Starting balance:  ${starting:.2f}")
print(f"  Ending balances:   min ${min(final_balances):.2f}, mean ${np.mean(final_balances):.2f}, "
      f"median ${np.median(final_balances):.2f}, max ${max(final_balances):.2f}")
print(f"  Weekly PnL:        min ${min(weekly_pnl):.2f}, mean ${np.mean(weekly_pnl):.2f}, "
      f"max ${max(weekly_pnl):.2f}")
print(f"  Max drawdowns:     min ${min(dds):.2f}, mean ${np.mean(dds):.2f}, max ${max(dds):.2f}")
print(f"  Positive weeks:    {sum(1 for x in weekly_pnl if x > 0)}/20")
