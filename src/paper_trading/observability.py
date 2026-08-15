from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from policy.score_aggregator import ModelOutputBundle
from policy.threshold_policy import TradingDecision


class TradingObservability:
    """Maintains rolling operational stats for paper and live trading."""

    def __init__(
        self,
        alert_config_path: str = "config/monitoring/alert_thresholds_v1.yaml",
        log_path: str = "logs/trading_observability.jsonl",
    ):
        with Path(alert_config_path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        self.thresholds = raw.get("alert_thresholds_v1", raw)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.latencies = deque(maxlen=500)
        self.freshness = deque(maxlen=500)
        self.fills = deque(maxlen=100)
        self.markouts = deque(maxlen=100)
        self.decisions = deque(maxlen=500)
        self.daily_pnl = 0.0

    def _log(self, event: dict[str, Any]) -> None:
        entry = {"ts_utc": datetime.now(timezone.utc).isoformat(), **event}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def record_latency(self, round_trip_ms: float):
        self.latencies.append(float(round_trip_ms))
        self._log({"event": "latency", "round_trip_ms": float(round_trip_ms)})

    def record_freshness(self, live_ref_age_s: float):
        self.freshness.append(float(live_ref_age_s))
        self._log({"event": "freshness", "live_ref_age_s": float(live_ref_age_s)})

    def record_fill(self, fill_event: dict):
        filled = bool(fill_event.get("filled", True))
        self.fills.append(filled)
        self.markouts.append(float(fill_event.get("markout_1s", 0.0) or 0.0))
        self.daily_pnl += float(fill_event.get("net_pnl", 0.0) or 0.0)
        self._log({"event": "fill", **fill_event})

    def record_decision(self, decision: TradingDecision, bundle: ModelOutputBundle):
        payload = {
            "event": "decision",
            "action": decision.action,
            "execute": decision.execute,
            "p_up_fair": bundle.p_up_fair,
            "p_fill_maker": bundle.p_fill_maker,
        }
        self.decisions.append(payload)
        self._log(payload)

    def check_alerts(self) -> list[dict]:
        alerts: list[dict] = []
        if self.latencies and max(self.latencies) > float(self.thresholds.get("latency_ms", 200)):
            alerts.append({"alert": "ALERT_HIGH_LATENCY", "value": max(self.latencies)})
        if self.freshness and max(self.freshness) > float(self.thresholds.get("freshness_s", 30)):
            alerts.append({"alert": "ALERT_STALE_REFERENCE", "value": max(self.freshness)})
        if len(self.fills) == self.fills.maxlen:
            fill_rate = sum(1 for item in self.fills if item) / len(self.fills)
            if fill_rate < float(self.thresholds.get("min_fill_rate_100", 0.05)):
                alerts.append({"alert": "ALERT_LOW_FILL_RATE", "value": fill_rate})
        if self.daily_pnl < -float(self.thresholds.get("daily_loss_cap_usdc", 50.0)):
            alerts.append({"alert": "ALERT_DAILY_LOSS_CAP", "value": self.daily_pnl})
        if self.markouts and mean(self.markouts) < float(self.thresholds.get("adverse_markout_threshold", -0.02)):
            alerts.append({"alert": "ALERT_ADVERSE_MARKOUT", "value": mean(self.markouts)})
        return alerts

    def generate_daily_audit_report(self, date: str, output_path: str) -> dict:
        alerts = self.check_alerts()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fill_rate = (sum(1 for item in self.fills if item) / len(self.fills)) if self.fills else 0.0
        lines = [
            f"# Trading Observability Audit {date}",
            "",
            f"- Decisions recorded: {len(self.decisions)}",
            f"- Fill rate window: {fill_rate:.6f}",
            f"- Daily PnL: {self.daily_pnl:.6f}",
            f"- Alerts: {len(alerts)}",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"date": date, "output_path": str(path), "alerts": alerts}

