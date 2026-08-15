"""Tracks open positions to resolution and reconciles fills.

For each fill the bot opens, we record (market_slug, down_token_id,
shares, entry_price, notional_usd, market_close_ts_ns). When a market
resolves we look up `resolved_side` from the recorder's resolution
shard and compute realized PnL.

Persisted to bot_state/positions.jsonl so a crashed/restarted bot can
reconcile open positions on next launch.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

# imports for resolution lookup
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binance_recorder.compression import iter_zstd_jsonl_with_options


@dataclass
class OpenPosition:
    market_slug: str
    down_token_id: str
    shares: float
    entry_price: float
    notional_usd: float
    fee_usd_estimated: float
    snapshot_ts_ns: int
    market_close_ts_ns: int
    order_id: Optional[str]
    edge_dn: float
    side: str = "DOWN"          # which side we bought (persisted, not a dynamic attr)
    strike: float = 0.0         # price-to-beat (chainlink @ open) — for RTDS-based resolution fallback


@dataclass
class ClosedPosition:
    open: OpenPosition
    resolved_side: str
    resolved_ts_ns: int
    realized_pnl_usd: float
    won: bool


class PositionManager:
    def __init__(self, *, state_dir: Path = Path("bot_state"),
                 raw_root: Path = Path("data"),
                 logger: logging.Logger | None = None) -> None:
        self.state_dir = state_dir
        self.raw_root = raw_root
        self.logger = logger or logging.getLogger("position_manager")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.open_log = self.state_dir / "open_positions.jsonl"
        self.closed_log = self.state_dir / "closed_positions.jsonl"
        self.open_positions: list[OpenPosition] = []
        self._load_open()

    def _load_open(self) -> None:
        if not self.open_log.exists(): return
        for line in self.open_log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                self.open_positions.append(OpenPosition(**d))
            except Exception as exc:
                self.logger.warning("could not reload open position: %r", exc)
        self.logger.info("loaded %d open positions from %s", len(self.open_positions), self.open_log)

    def _persist_open(self) -> None:
        with self.open_log.open("w", encoding="utf-8") as fh:
            for p in self.open_positions:
                fh.write(json.dumps(asdict(p)) + "\n")

    def add_position(self, pos: OpenPosition) -> None:
        self.open_positions.append(pos)
        self._persist_open()

    def reconcile_resolved(self, now_ns: int) -> list[ClosedPosition]:
        """For each position whose market_close_ts_ns has passed, look
        up its resolved side from the recorder's resolution log and
        close it. Returns the list of closed positions."""
        closed: list[ClosedPosition] = []
        still_open: list[OpenPosition] = []
        for pos in self.open_positions:
            if pos.market_close_ts_ns > now_ns - 60 * 1_000_000_000:
                # Still inside the resolution-wait window
                still_open.append(pos); continue
            resolved = self._lookup_resolution(pos.market_slug)
            if resolved is None:
                # not resolved yet, keep waiting
                still_open.append(pos); continue
            resolved_side, resolved_ts_ns = resolved
            closed.append(self._finalize(pos, resolved_side, resolved_ts_ns, "recorder"))
        if len(closed) != len(self.open_positions) - len(still_open):
            self.logger.warning("position accounting mismatch — investigate")
        self.open_positions = still_open
        self._persist_open()
        return closed

    def _finalize(self, pos: OpenPosition, resolved_side: str, resolved_ts_ns: int,
                  source: str) -> ClosedPosition:
        """Compute PnL for a resolved position, log it, return ClosedPosition.
        Win iff our `side` matches resolved_side (UP positions win on 'up')."""
        position_side = str(getattr(pos, "side", "DOWN")).lower()
        won = (resolved_side == position_side)
        payoff = pos.shares * 1.0 if won else 0.0
        pnl = payoff - pos.notional_usd - pos.fee_usd_estimated
        c = ClosedPosition(open=pos, resolved_side=resolved_side,
                           resolved_ts_ns=resolved_ts_ns,
                           realized_pnl_usd=float(pnl), won=won)
        with self.closed_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "open": asdict(pos), "resolved_side": resolved_side,
                "resolved_ts_ns": resolved_ts_ns, "realized_pnl_usd": pnl,
                "won": won, "resolution_source": source,
            }) + "\n")
        return c

    def reconcile_resolved_rtds(self, now_ns: int, rtds_price_at, grace_s: float = 120.0):
        """FALLBACK: resolve positions the resolution RECORDER missed, using the
        RTDS/chainlink CLOSE price vs the position's strike (price-to-beat):
        close >= strike -> 'up' wins. Without this, a recorder gap leaves a
        position un-reconciled forever -> its loss never reaches the daily
        loss-limit and the risk gate goes blind. `rtds_price_at(ns)->float|None`."""
        closed: list[ClosedPosition] = []
        still_open: list[OpenPosition] = []
        grace_ns = int(grace_s * 1_000_000_000)
        for pos in self.open_positions:
            strike = float(getattr(pos, "strike", 0.0) or 0.0)
            # give the recorder `grace_s` after close before falling back to RTDS
            if pos.market_close_ts_ns > now_ns - grace_ns or strike <= 0:
                still_open.append(pos); continue
            cp = rtds_price_at(pos.market_close_ts_ns)
            if cp is None:
                still_open.append(pos); continue   # no RTDS at close -> keep waiting
            resolved_side = "up" if cp >= strike else "down"
            closed.append(self._finalize(pos, resolved_side, pos.market_close_ts_ns, "rtds_fallback"))
            self.logger.info("RTDS-reconciled %s: close=%.1f strike=%.1f -> %s (recorder missed)",
                             pos.market_slug, cp, strike, resolved_side)
        self.open_positions = still_open
        self._persist_open()
        return closed

    def _lookup_resolution(self, market_slug: str) -> tuple[str, int] | None:
        """Walk today's + yesterday's resolution shards and return the
        first matching market_resolution event."""
        today = datetime.now(UTC).date()
        from datetime import timedelta
        for d in (today, today - timedelta(days=1), today + timedelta(days=1)):
            folder = self.raw_root / "raw" / "polymarket" / "resolution" / "btc_updown_5m" / d.isoformat()
            if not folder.exists(): continue
            for p in sorted(folder.glob("*.resolution.jsonl.zst")):
                try:
                    for rec in iter_zstd_jsonl_with_options(p, allow_truncated_stream=True,
                                                             allow_partial_final_line=True):
                        if rec.get("record_type") != "market_resolution": continue
                        if str(rec.get("market_slug") or "") != market_slug: continue
                        side = str(rec.get("resolved_side") or rec.get("winning_outcome") or "").lower()
                        if side in ("up", "down"):
                            return side, int(rec.get("recv_ts_ns") or 0)
                except Exception:
                    continue
        return None
