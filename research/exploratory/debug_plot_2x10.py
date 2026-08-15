#!/usr/bin/env python3
"""Plot: 1 entry/market vs 2 entries/market (10s gap), side by side."""
import io, sys, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cmp = json.loads(Path("docs/model02_taker_2x10_compare.json").read_text())
labels = [r["label"] for r in cmp]
pw_by_label = {r["label"]: r["per_week"] for r in cmp}

# We'll only plot the first 2 configurations to compare directly
plot_labels = ["1 entry/market (current)", "2 entries/market, 10s gap"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={"hspace": 0.32, "wspace": 0.25})
starting = 100.0
colors = ["#1f77b4", "#d62728"]

# Equity curves overlaid
ax = axes[0, 0]
for color, label in zip(colors, plot_labels):
    pw = pw_by_label[label]
    for wk in pw:
        daily = wk["daily_pnl"]
        sorted_dates = sorted(daily.keys())
        balance = starting
        xs = [0]; ys = [balance]
        for d in sorted_dates:
            balance += daily[d]
            xs.append(len(xs))
            ys.append(balance)
        ax.plot(xs, ys, alpha=0.35, linewidth=1.2, color=color)
    # legend handle
    ax.plot([], [], color=color, linewidth=2, label=label)
ax.axhline(starting, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlabel("Day of week")
ax.set_ylabel("Balance (USD)")
ax.set_title("Equity curves — 1 vs 2 entries/market (overlaid)")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)

# Ending-balance distribution
ax = axes[0, 1]
for color, label in zip(colors, plot_labels):
    pw = pw_by_label[label]
    finals = [starting + w["net_pnl_usd"] for w in pw]
    ax.hist(finals, bins=12, color=color, edgecolor="black", alpha=0.55,
            label=f"{label}\nmean ${np.mean(finals):.0f}")
ax.set_xlabel("Ending balance (USD)")
ax.set_ylabel("Number of weeks")
ax.set_title("Ending balance distribution")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

# Weekly PnL bar chart
ax = axes[1, 0]
x = np.arange(1, 21)
width = 0.38
for offset, color, label in zip([-width/2, width/2], colors, plot_labels):
    pw = pw_by_label[label]
    pnls = [w["net_pnl_usd"] for w in pw]
    ax.bar(x + offset, pnls, width, color=color, edgecolor="black", alpha=0.85, label=label)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Week index")
ax.set_ylabel("Weekly PnL (USD)")
ax.set_title("Weekly PnL — 20 OOS MC weeks")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
ax.set_xticks(x)

# Max drawdown comparison
ax = axes[1, 1]
for offset, color, label in zip([-width/2, width/2], colors, plot_labels):
    pw = pw_by_label[label]
    dds = [w["max_drawdown_usd"] for w in pw]
    ax.bar(x + offset, dds, width, color=color, edgecolor="black", alpha=0.85,
           label=f"{label}\nmean ${np.mean(dds):.2f}, max ${max(dds):.2f}")
ax.axhline(10, color="red", linestyle=":", linewidth=1.5, alpha=0.6, label="$10 daily-loss floor")
ax.set_xlabel("Week index")
ax.set_ylabel("Max drawdown (USD)")
ax.set_title("Max intraweek drawdown")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.set_xticks(x)

fig.suptitle("Model 02 DOWN-only taker — 1 entry vs 2 entries/market (10s gap), 20 OOS MC weeks",
             fontsize=13, fontweight="bold")
out = Path("docs/model02_taker_1v2_entries.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out.resolve()}")
