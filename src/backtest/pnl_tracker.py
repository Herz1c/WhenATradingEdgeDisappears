"""$-denominated PnL accounting with daily and drawdown tracking.

Each trade is treated as a 5-second round-trip held to `future_mid`; PnL
realized immediately. Tracks daily PnL buckets and a running running-peak
drawdown so the grading function can compute Calmar and worst-day metrics
without re-walking the fill log.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trading.entities import Fill


class PnLTracker:
    def __init__(self, starting_balance_usd: float):
        self.starting = float(starting_balance_usd)
        self.balance = float(starting_balance_usd)
        self.peak_balance = float(starting_balance_usd)
        self.trough_balance = float(starting_balance_usd)
        self.max_drawdown_usd = 0.0          # running max(peak - balance)
        self.daily_pnl: dict[str, float] = defaultdict(float)
        self.n_wins = 0
        self.n_losses = 0
        self.n_fills = 0
        self.gross_fees = 0.0
        self.gross_rebates = 0.0

    def apply_fill(self, fill: Fill, date_str: str, future_mid: float) -> float:
        signed = fill.size if fill.order_side == "BID" else -fill.size
        markout = (future_mid - fill.price) * signed
        realized = markout + fill.rebate_earned - fill.fees_paid

        self.balance += realized
        self.daily_pnl[date_str] += realized
        self.gross_fees += fill.fees_paid
        self.gross_rebates += fill.rebate_earned
        self.n_fills += 1
        if realized > 0:
            self.n_wins += 1
        elif realized < 0:
            self.n_losses += 1

        self.peak_balance = max(self.peak_balance, self.balance)
        self.trough_balance = min(self.trough_balance, self.balance)
        self.max_drawdown_usd = max(self.max_drawdown_usd, self.peak_balance - self.balance)
        return realized

    def is_busted(self) -> bool:
        return self.balance <= 0.0

    def summary(self) -> dict:
        n_trades = self.n_wins + self.n_losses
        return {
            "starting_usd": self.starting,
            "ending_usd": self.balance,
            "net_pnl_usd": self.balance - self.starting,
            "max_drawdown_usd": self.max_drawdown_usd,
            "peak_balance_usd": self.peak_balance,
            "trough_balance_usd": self.trough_balance,
            "n_fills": self.n_fills,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "win_rate": (self.n_wins / n_trades) if n_trades else 0.0,
            "gross_fees_usd": self.gross_fees,
            "gross_rebates_usd": self.gross_rebates,
            "daily_pnl": dict(self.daily_pnl),
            "worst_day_pnl_usd": (min(self.daily_pnl.values()) if self.daily_pnl else 0.0),
            "best_day_pnl_usd": (max(self.daily_pnl.values()) if self.daily_pnl else 0.0),
        }
