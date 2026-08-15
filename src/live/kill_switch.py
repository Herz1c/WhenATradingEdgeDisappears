from __future__ import annotations

import json
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional


class KillReason(Enum):
    MANUAL = "manual"
    HIGH_LATENCY = "high_latency"
    DAILY_LOSS_CAP = "daily_loss_cap"
    STALE_REFERENCE = "stale_reference"
    MONITORING_ANOMALY = "monitoring_anomaly"
    SYSTEM_ERROR = "system_error"


class KillSwitch:
    """Thread-safe kill switch checked before every order submission."""

    def __init__(self, log_path: str = "logs/kill_switch.jsonl"):
        self._killed = threading.Event()
        self._lock = threading.Lock()
        self._kill_reason: Optional[KillReason] = None
        self._kill_ts: Optional[int] = None
        self._kill_log_path = Path(log_path)
        self._kill_log_path.parent.mkdir(parents=True, exist_ok=True)

    def kill(self, reason: KillReason, details: str = ""):
        """Trigger kill switch. Thread-safe and idempotent."""
        with self._lock:
            if not self._killed.is_set():
                self._killed.set()
                self._kill_reason = reason
                self._kill_ts = time.time_ns()
            self._log_kill(reason, details)

    def is_killed(self) -> bool:
        return self._killed.is_set()

    def reset(self, operator_auth: str):
        if not operator_auth:
            raise ValueError("operator_auth is required to reset KillSwitch")
        with self._lock:
            self._log_reset(operator_auth)
            self._kill_reason = None
            self._kill_ts = None
            self._killed.clear()

    @property
    def kill_reason(self) -> Optional[KillReason]:
        return self._kill_reason

    def _log_kill(self, reason: KillReason, details: str):
        entry = {
            "ts": time.time_ns(),
            "event": "KILL",
            "reason": reason.value,
            "details": details,
        }
        with self._kill_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def _log_reset(self, operator_auth: str):
        entry = {
            "ts": time.time_ns(),
            "event": "RESET",
            "operator_auth_hash": hash(operator_auth),
        }
        with self._kill_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

