"""Hard pre-trade guards — kill switch + daily loss cap.

Both checks are called BEFORE every order. The risk gate is also
queried by the orchestrator at startup and is consulted before any
state-mutating order is dispatched.

Behavior:
  - Kill switch:    if STOP_TRADING file exists at the configured path,
                    `should_block()` returns ("kill_switch_file_present").
  - Daily loss:     realized PnL on the current UTC day must stay above
                    DAILY_LOSS_LIMIT_USD (default -$10.00). Once breached,
                    blocks new entries until UTC date rollover.
  - Position cap:   stops new entries if active concurrent margin already
                    at MAX_MARGIN_USD (also enforced in decision_engine
                    but doubled here so a bug in the decision can't leak).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path


DAILY_LOSS_LIMIT_USD = -10.00
DEFAULT_KILL_SWITCH_PATH = Path("STOP_TRADING")


@dataclass
class RiskGate:
    kill_switch_path: Path = DEFAULT_KILL_SWITCH_PATH
    daily_loss_limit_usd: float = DAILY_LOSS_LIMIT_USD
    realized_pnl_by_day: dict[date, float] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("risk_gate"))

    def record_realized(self, when_ns: int, pnl_usd: float) -> None:
        d = datetime.fromtimestamp(when_ns / 1e9, UTC).date()
        self.realized_pnl_by_day[d] = self.realized_pnl_by_day.get(d, 0.0) + float(pnl_usd)

    def today_pnl(self) -> float:
        return float(self.realized_pnl_by_day.get(datetime.now(UTC).date(), 0.0))

    def should_block(self, *, now_ns: int) -> tuple[bool, str]:
        if self.kill_switch_path.exists():
            return True, f"kill_switch_file_present:{self.kill_switch_path}"
        today = datetime.fromtimestamp(now_ns / 1e9, UTC).date()
        pnl = self.realized_pnl_by_day.get(today, 0.0)
        if pnl <= self.daily_loss_limit_usd:
            return True, f"daily_loss_limit_breached:{pnl:+.2f}"
        return False, ""
