from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

if TYPE_CHECKING:
    # threshold_policy imports InventoryState from here, so this stays
    # type-checking-only to avoid a circular import at runtime.
    from policy.threshold_policy import TradingDecision


@dataclass
class InventoryState:
    """Current inventory state."""

    up_position: float = 0.0
    down_position: float = 0.0
    cash: float = 0.0
    daily_pnl: float = 0.0
    daily_fees: float = 0.0
    daily_rebates: float = 0.0
    open_orders: dict = field(default_factory=dict)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class InventoryController:
    """Manages position limits and inventory risk."""

    def __init__(self, config_path: str | Path = "config/policy/risk_caps_v1.yaml"):
        self.config_path = Path(config_path)
        raw = load_yaml(self.config_path)
        self.config = raw.get("risk_caps_v1", raw)
        self.state = InventoryState()

    def check_position_limits(
        self,
        proposed_action: "TradingDecision",
        current_state: InventoryState,
        market_slug: str,
    ) -> tuple[bool, str]:
        if proposed_action.action == "cancel_all" or not proposed_action.execute:
            return True, "non_position_increasing_action"

        max_open_orders = int(self.config.get("max_open_orders", 20))
        if len(current_state.open_orders) >= max_open_orders:
            return False, "max_open_orders"

        per_market_cap = float(self.config.get("per_market_size_cap_usdc", 100.0))
        order_size = float(proposed_action.size or 0.0)
        current_market = current_state.open_orders.get(market_slug, {})
        current_exposure = float(current_market.get("exposure_usdc", 0.0))
        if current_exposure + order_size > per_market_cap:
            return False, "per_market_size_cap"

        total_exposure = sum(float(order.get("exposure_usdc", 0.0)) for order in current_state.open_orders.values())
        if total_exposure + order_size > float(self.config.get("max_total_exposure_usdc", 500.0)):
            return False, "max_total_exposure"

        if current_state.daily_pnl < -float(self.config.get("per_day_loss_cap_usdc", 50.0)):
            return False, "per_day_loss_cap"

        return True, "allowed"

    def update_state(self, fill_event: dict, resolution_event: Optional[dict] = None) -> InventoryState:
        side = str(fill_event.get("token_side", fill_event.get("side", ""))).upper()
        size = float(fill_event.get("size", fill_event.get("filled_size", 0.0)) or 0.0)
        price = float(fill_event.get("price", fill_event.get("fill_price", 0.0)) or 0.0)
        fees = float(fill_event.get("fees_paid", 0.0) or 0.0)
        rebates = float(fill_event.get("rebate_earned", 0.0) or 0.0)

        if side == "UP":
            self.state.up_position += size
        elif side == "DOWN":
            self.state.down_position += size
        self.state.cash -= size * price + fees - rebates
        self.state.daily_fees += fees
        self.state.daily_rebates += rebates

        if resolution_event:
            realized = float(resolution_event.get("realized_pnl", 0.0) or 0.0)
            self.state.daily_pnl += realized
        return self.state

