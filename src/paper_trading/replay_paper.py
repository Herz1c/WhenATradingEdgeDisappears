from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from policy.inventory_controller import InventoryController
from policy.score_aggregator import ModelOutputBundle, ScoreAggregator
from policy.threshold_policy import ThresholdPolicyEngine
from polymarket_recorder.execution_simulator import ExecutionSimulator as ExecutionSimulatorV1


class ReplayPaperTrader:
    """
    Runs the policy stack against historical data.

    This class is infrastructure only: without trained models and non-placeholder
    policy weights it records no executable trades.
    """

    def __init__(
        self,
        policy_config_path: str,
        risk_caps_path: str,
        score_aggregator_weights: dict[str, float] | None = None,
    ):
        self.simulator_class = ExecutionSimulatorV1
        self.simulator = None
        self.aggregator = ScoreAggregator(weights=score_aggregator_weights)
        self.policy = ThresholdPolicyEngine(config_path=policy_config_path)
        self.inventory = InventoryController(config_path=risk_caps_path)

    def run(self, date_from: str, date_to: str, decision_interval_s: float = 1.0) -> dict[str, Any]:
        bundle = ModelOutputBundle()
        scores = self.aggregator.compute_action_scores(bundle)
        decision = self.policy.decide(
            scores,
            self.inventory.state,
            {
                "all_weights_placeholder": self.aggregator.placeholder_mode,
                "live_reference_trusted": False,
                "t_to_close_s": 9999,
                "order_size": 0.0,
            },
        )
        return {
            "mode": "replay_paper",
            "date_from": date_from,
            "date_to": date_to,
            "decision_interval_s": decision_interval_s,
            "decisions": [
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "action": decision.action,
                    "execute": decision.execute,
                    "reason": decision.reason,
                    "override_reason": decision.override_reason,
                }
            ],
            "orders_submitted": 0,
            "fills": 0,
            "net_pnl": 0.0,
            "placeholder_mode": self.aggregator.placeholder_mode,
        }

    def generate_report(self, results: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Replay Paper Report",
            "",
            f"- Date range: {results.get('date_from')} to {results.get('date_to')}",
            f"- Placeholder mode: {results.get('placeholder_mode')}",
            f"- Orders submitted: {results.get('orders_submitted', 0)}",
            f"- Fills: {results.get('fills', 0)}",
            f"- Net PnL: {float(results.get('net_pnl', 0.0)):.6f}",
            "",
            "## Decision Breakdown",
            "| action | execute | reason | override |",
            "| --- | --- | --- | --- |",
        ]
        for decision in results.get("decisions", []):
            lines.append(
                "| {action} | {execute} | {reason} | {override} |".format(
                    action=decision.get("action", ""),
                    execute=decision.get("execute", False),
                    reason=decision.get("reason", ""),
                    override=decision.get("override_reason") or "",
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"output_path": str(path), "orders_submitted": results.get("orders_submitted", 0)}

