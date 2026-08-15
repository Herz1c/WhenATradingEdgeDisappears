from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from policy.score_aggregator import ActionScore

if TYPE_CHECKING:
    # inventory_controller imports TradingDecision from here, so this stays
    # type-checking-only to avoid a circular import at runtime.
    from policy.inventory_controller import InventoryState


@dataclass
class TradingDecision:
    """Output of policy engine for one decision point."""

    action: str
    execute: bool
    price: Optional[float]
    size: Optional[float]
    reason: str
    override_reason: Optional[str] = None


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


class ThresholdPolicyEngine:
    """
    Converts action scores to executable trading decisions.

    Configured no-trade and emergency conditions are intentionally conservative
    until trained model outputs and production policy weights are available.
    """

    def __init__(self, config_path: str | Path = "config/policy/threshold_policy_v1.yaml"):
        self.config_path = Path(config_path)
        self.config = _load_yaml(self.config_path)
        self.thresholds = dict(self.config.get("thresholds", {}))

    def decide(
        self,
        action_scores: list[ActionScore],
        inventory: "InventoryState",
        market_context: dict,
    ) -> TradingDecision:
        if not action_scores:
            return TradingDecision("cancel_all", False, None, None, "no_action_scores", "no_scores")

        override = self._override_reason(inventory, market_context)
        if override:
            return TradingDecision("cancel_all", False, None, None, "override_cancel_all", override)

        ranked = sorted(action_scores, key=lambda score: score.score, reverse=True)
        for score in ranked:
            threshold = float(self.thresholds.get(score.action, 0.0))
            if score.score >= threshold:
                execute = score.action.startswith("quote_") or score.action == "taker_emergency_only"
                return TradingDecision(
                    action=score.action,
                    execute=execute,
                    price=market_context.get("limit_price"),
                    size=market_context.get("order_size"),
                    reason="score_above_threshold",
                    override_reason=None,
                )

        return TradingDecision("cancel_all", False, None, None, "no_action_above_threshold", None)

    def _override_reason(self, inventory: "InventoryState", market_context: dict) -> str | None:
        if bool(market_context.get("all_weights_placeholder", False)):
            return "all_weights_placeholder"
        if bool(market_context.get("live_reference_trusted") is False):
            return "live_reference_not_trusted"
        if float(market_context.get("t_to_close_s", 9999.0) or 9999.0) < 3.0:
            return "too_close_to_close"
        if bool(market_context.get("crossed_book", False)):
            return "crossed_book"
        if float(market_context.get("p_flip_before_close", 0.0) or 0.0) > 0.70:
            return "flip_risk_too_high"

        max_position = float(market_context.get("max_position", 0.0) or 0.0)
        if max_position > 0 and max(abs(inventory.up_position), abs(inventory.down_position)) > max_position:
            return "inventory_exceeds_max_position"

        daily_loss_cap = float(market_context.get("daily_loss_cap", 0.0) or 0.0)
        if daily_loss_cap > 0 and inventory.daily_pnl < -daily_loss_cap:
            return "daily_loss_cap"
        return None

