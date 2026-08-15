"""Polymarket fee/rebate calculator.

Wraps src/execution_simulator/fee_schedule_registry.py so backtest code can
ask: "for a fill at this price and size on this date, what fee or rebate?"

Polymarket post-2026-03-30 schedule:
    taker_fee_usd = size * 0.072 * (price * (1 - price))^1
    maker_fee_usd = 0
    maker_rebate_usd = 0.20 * taker_fee_usd_at_same_price

All amounts in USD per contract (1 contract = $1 max payout).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from execution_simulator.fee_schedule_registry import FeeScheduleEntry, FeeScheduleRegistry


_DEFAULT_REGISTRY: Optional[FeeScheduleRegistry] = None


def get_registry() -> FeeScheduleRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = FeeScheduleRegistry(
            _REPO_ROOT / "config" / "fee_schedules" / "fee_schedule_registry.yaml"
        )
    return _DEFAULT_REGISTRY


class FeeCalculator:
    """Compute fees/rebates for fills on a given date.

    Pick the active schedule once per day via `for_date(date_str)`, then
    call `taker_fee_usd` / `maker_rebate_usd` per fill.
    """

    def __init__(self, entry: FeeScheduleEntry, fee_scale: float = 1.0):
        self.entry = entry
        self.fee_scale = float(fee_scale)

    @classmethod
    def for_date(cls, date_str: str, fee_scale: float = 1.0,
                 registry: Optional[FeeScheduleRegistry] = None) -> "FeeCalculator":
        reg = registry or get_registry()
        entry = reg.lookup(f"{date_str}T12:00:00Z", asset_class="crypto")
        return cls(entry, fee_scale=fee_scale)

    def taker_fee_usd(self, price: float, size: float) -> float:
        """Taker fee in USD. `size` is in contracts; `price` in [0, 1]."""
        price = max(1e-6, min(1.0 - 1e-6, float(price)))
        return self.entry.taker_fee(price=price, size=size) * self.fee_scale

    def maker_rebate_usd(self, price: float, size: float) -> float:
        """Maker rebate = 20% of what the taker fee would be at this price."""
        tf = self.taker_fee_usd(price=price, size=size)
        return self.entry.maker_rebate(taker_fee=tf)

    def maker_fee_usd(self, price: float, size: float) -> float:
        """Maker fee (Polymarket = 0)."""
        return self.entry.maker_fee(price=price, size=size)
