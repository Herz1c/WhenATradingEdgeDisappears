"""Maker strategy: parallel POST_ONLY-at-midpoint execution path.

Runs alongside the existing FAK taker (which is untouched). When a model
ENTER signal fires, the maker submits a GTC POST_ONLY BUY for the DOWN
token at midpoint, then waits for either:
  (a) someone to hit our bid -> we collect a maker fill at our limit
  (b) our cancel deadline -> we cancel; if there was a partial fill before
      cancel, the cancel response gives us the filled amount.

Why a separate module: the FAK path has been hardened over many sessions.
We do NOT touch it. The maker path has different lifecycle semantics
(orders rest, fill async, must be cancelled) so it owns its own state,
its own per-market position cap, its own decision log, and its own
reconcile loop. Risk gates (daily PnL) are SHARED via the global RiskGate
so a maker loss counts the same as a taker loss for cutoff purposes.

Selection is env-driven in main.py:
  LIVE_BOT_STRATEGY=taker  (default)  -> only the existing FAK path
  LIVE_BOT_STRATEGY=maker             -> only this module
  LIVE_BOT_STRATEGY=both              -> both fire on the same signal,
                                          each with its own per-market cap
                                          so they don't block each other.

Backtest evidence on 4 OOS days (170k snapshots, 605 markets) that
motivated this code: midpoint-maker @ edge_threshold=0.20 gave 72.6%
fill rate, 86.6% win rate, +$0.36/fill. Same model signals as the FAK
taker that's been losing on live.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from backtest.fees import FeeCalculator
from live_bot.decision_engine import MIN_DOWN_PRICE, RiskState, SECOND_NS
from live_bot.order_router import FillReceipt, PolymarketOrderRouter
from live_bot.position_manager import OpenPosition, PositionManager


def _now_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1e9)


@dataclass
class _RestingOrder:
    """One in-flight POST_ONLY order we placed and are tracking.
    Will be cancelled at `cancel_deadline_ns`; before then we just leave
    it on the book."""
    order_id: str
    market_slug: str
    down_token_id: str
    limit_price: float
    size_shares: float
    notional_usd: float
    snapshot_ts_ns: int
    submitted_at_ns: int
    cancel_deadline_ns: int
    market_close_ts_ns: int
    edge_dn: float
    p_up: float
    # Updated after cancel/reconcile
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    final_status: str = "live"
    cancelled: bool = False


class MakerStrategy:
    """Parallel POST_ONLY-at-midpoint execution path."""

    def __init__(self, *,
                 router: PolymarketOrderRouter,
                 positions: PositionManager,
                 risk_state: RiskState,
                 decisions_dir: Path,
                 logger: logging.Logger | None = None) -> None:
        self.router = router
        self.positions = positions
        self.risk_state = risk_state             # SEPARATE state from taker
        self.decisions_dir = decisions_dir
        self.logger = logger or logging.getLogger("maker_strategy")

        self._open_orders: dict[str, _RestingOrder] = {}   # order_id -> resting
        # cache of per-market open count for the maker side (separate from
        # taker's positions_per_market so the two don't block each other)
        self._open_count_per_market: dict[str, int] = defaultdict(int)
        # Anti-spam: minimum gap between maker entries on the same market.
        self._last_entry_ns_per_market: dict[str, int] = {}
        self._shutdown = asyncio.Event()
        self._stats = {
            "submits_attempted": 0, "submits_ok": 0, "submits_rejected": 0,
            "fills_reconciled": 0, "cancels_called": 0,
            "filled_shares_total": 0.0, "filled_notional_total": 0.0,
        }
        # Lifetime of a posted order (seconds) before we cancel it.
        # Default 30 s = leaves it in book for ~half of the 60 s trading
        # window so a counter-flow has time to come, while still letting us
        # exit before the close in case it's about to resolve against us.
        self._order_lifetime_s = float(os.getenv("LIVE_BOT_MAKER_ORDER_LIFETIME_S", "30"))
        # Per-market cap and gap (independent of the taker side's caps).
        # Default: up to 2 maker bids per market, 10 s apart -- gives the
        # strategy two staggered shots at picking up counter-flow as the
        # book moves through the 60 s trading window.
        self._max_positions_per_market = int(os.getenv("LIVE_BOT_MAKER_MAX_PER_MARKET", "2"))
        self._min_gap_s = float(os.getenv("LIVE_BOT_MAKER_MIN_GAP_S", "10.0"))
        # How often the reconcile loop wakes up (seconds)
        self._reconcile_interval_s = float(os.getenv("LIVE_BOT_MAKER_RECONCILE_S", "2.0"))
        # Polymarket's per-market `min_order_size` is 5 shares on the
        # current BTC UpDown markets (discovered via the sub-dollar probe
        # tool — 2-share orders are rejected with "Size (2) lower than the
        # minimum: 5"). So the maker sends a fixed 5-share order every
        # time, and notional scales with the entry price:
        #   limit 0.30 -> notional $1.50
        #   limit 0.50 -> notional $2.50
        #   limit 0.80 -> notional $4.00
        # Env-overridable in case Polymarket relaxes/tightens this.
        self._fixed_shares = float(os.getenv("LIVE_BOT_MAKER_SHARES", "5.0"))
        # Notional CAP: skip the trade if 5 shares * limit_price would
        # blow past this. Default $5 -- comfortably above the worst-case
        # 5 * 0.97 = $4.85 but stops a runaway bid in degenerate cases.
        self._max_notional = float(os.getenv("LIVE_BOT_MAKER_MAX_NOTIONAL", "5.0"))
        self._fee_calc_cache: dict[str, FeeCalculator] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def stop(self) -> None:
        self._shutdown.set()

    @property
    def stats(self) -> dict:
        return self._stats

    # ── submit ──────────────────────────────────────────────────────────────
    def submit_for_decision(self, decision: dict) -> Optional[_RestingOrder]:
        """Called from LiveBot._handle_decision when ENTER fires AND the
        maker strategy is enabled. The taker path may also fire on the
        same decision; the two are independent. Returns the resting
        order on successful submit, None otherwise."""
        snap = decision.get("snap") or {}
        slug = snap.get("market_slug")
        if not slug:
            return None
        # Per-market open cap (independent of taker)
        if self._open_count_per_market[slug] >= self._max_positions_per_market:
            self._write_log({
                "logged_at_ns": _now_ns(), "strategy": "maker",
                "snapshot_ts_ns": int(snap.get("snapshot_ts_ns") or 0),
                "market_slug": slug, "decision": "SKIP",
                "reason": "maker_per_market_cap",
            })
            return None
        # Per-market entry gap (independent of taker)
        now_ns = _now_ns()
        if (last := self._last_entry_ns_per_market.get(slug)) is not None:
            if (now_ns - last) < int(self._min_gap_s * SECOND_NS):
                self._write_log({
                    "logged_at_ns": now_ns, "strategy": "maker",
                    "snapshot_ts_ns": int(snap.get("snapshot_ts_ns") or 0),
                    "market_slug": slug, "decision": "SKIP",
                    "reason": "maker_min_gap",
                })
                return None

        # Pricing: midpoint of the real DOWN book if both sides are fresh,
        # else fall back to derived (1 - up_bid) - small offset so we're
        # never crossing.
        up_bid = snap.get("up_token_best_bid")
        up_ask = snap.get("up_token_best_ask")
        down_ask = snap.get("down_token_best_ask")
        # We don't have down_bid in the snap dict by default — see comment
        # in feature_runtime; for now derive it as 1 - up_ask (the implied
        # down bid). That's conservative.
        try:
            up_bid_f = float(up_bid) if up_bid is not None else None
            up_ask_f = float(up_ask) if up_ask is not None else None
            down_ask_f = float(down_ask) if down_ask is not None else None
        except (TypeError, ValueError):
            return None

        if down_ask_f is not None and 0 < down_ask_f < 1 and up_ask_f is not None:
            implied_down_bid = 1.0 - up_ask_f
            if 0 < implied_down_bid < down_ask_f:
                limit = (implied_down_bid + down_ask_f) / 2.0
            else:
                # Spread inverted / weird; fall back to one tick inside ask
                limit = down_ask_f - 0.01
        elif up_bid_f is not None and 0 < up_bid_f < 1:
            # No fresh down book; derive a conservative midpoint from up
            implied_down_ask = 1.0 - up_bid_f
            implied_down_bid = 1.0 - (up_ask_f if up_ask_f is not None else up_bid_f)
            limit = max(0.02, (implied_down_bid + implied_down_ask) / 2.0)
        else:
            return None
        # Round to the tick
        limit = round(float(limit), 2)
        if not (0.02 <= limit <= 0.98):
            return None
        # HARD CAP at (down_ask - 1 tick). On thin spreads the midpoint we
        # just computed can equal or exceed down_ask, which Polymarket
        # rejects as "invalid post-only order: order crosses book". We MUST
        # be strictly below the current ask to qualify as a maker.
        if down_ask_f is not None and 0 < down_ask_f < 1:
            max_safe_limit = round(down_ask_f - 0.01, 2)
            if limit > max_safe_limit:
                limit = max_safe_limit
            if limit < 0.02:
                # Tiny spread, can't post safely
                return None
        # Floor on the actual ORDER price (not just the current ask). The
        # decision-engine MIN_DOWN_PRICE gates on snap['down_token_best_ask'],
        # but midpoint can land BELOW that ask. So if down_ask=0.31 and
        # down_bid=0.27 the decision engine lets ENTER through but the
        # midpoint would be 0.29 -- which we want to block as cheap-longshot
        # territory. Re-using MIN_DOWN_PRICE keeps the two paths consistent
        # (raise once, both honour it).
        if limit < MIN_DOWN_PRICE:
            self._write_log({
                "logged_at_ns": now_ns, "strategy": "maker",
                "snapshot_ts_ns": int(snap.get("snapshot_ts_ns") or 0),
                "market_slug": slug, "decision": "SKIP",
                "reason": f"maker_limit_below_floor_{MIN_DOWN_PRICE:.2f}_{limit:.3f}",
            })
            return None

        # Always 5 shares — Polymarket's per-market min_order_size on
        # current BTC UpDown markets. Notional scales linearly with the
        # entry price.
        size_shares = self._fixed_shares
        target_notional = size_shares * limit
        if target_notional > self._max_notional:
            self._write_log({
                "logged_at_ns": now_ns, "strategy": "maker",
                "snapshot_ts_ns": int(snap.get("snapshot_ts_ns") or 0),
                "market_slug": slug, "decision": "SKIP",
                "reason": f"maker_notional_cap_{self._max_notional:.2f}_"
                          f"would_be_{target_notional:.2f}",
            })
            return None

        # Submit
        down_token_id = decision.get("down_token_id") or snap.get("down_token_id")
        if not down_token_id:
            # main.py knows the asset_id_by_slug map; we expect it to be
            # patched into `decision` before calling us. See main.py's
            # _handle_decision wiring.
            self.logger.warning("maker submit skipped: no down_token_id for %s", slug)
            return None
        self._stats["submits_attempted"] += 1
        receipt: FillReceipt = self.router.submit_buy_down_post_only(
            down_token_id=str(down_token_id),
            limit_price=limit,
            size_shares=size_shares,
        )
        # Build a maker log record up front so even rejections show up
        market_close_ns = int(decision.get("market_close_ts_ns")
                              or snap.get("market_close_ts_ns") or 0)
        cancel_deadline = min(
            now_ns + int(self._order_lifetime_s * SECOND_NS),
            max(now_ns, market_close_ns - 3 * SECOND_NS),    # cancel ~3 s before close
        )
        rec = {
            "logged_at_ns": now_ns,
            "strategy": "maker",
            "snapshot_ts_ns": int(snap.get("snapshot_ts_ns") or 0),
            "market_slug": slug,
            "limit_price": limit,
            "size_shares": float(receipt.filled_size or size_shares),
            "target_notional": target_notional,
            "edge_dn": float(decision.get("edge_dn") or 0.0),
            "p_up": float(decision.get("p_up") or 0.0),
            "cancel_deadline_ns": cancel_deadline,
            "router_mode": "live" if self.router.live else "dry_run",
            "decision": "SUBMIT",
            "receipt": {
                "success": receipt.success, "dry_run": receipt.dry_run,
                "filled_size": receipt.filled_size, "avg_price": receipt.avg_price,
                "status": receipt.status, "order_id": receipt.order_id,
                "err": receipt.err,
            },
        }
        if not receipt.success:
            self._stats["submits_rejected"] += 1
            rec["decision"] = "REJECTED"
            self._write_log(rec)
            # Update the per-market gap timer EVEN on rejection. Otherwise
            # every subsequent snapshot (and there are dozens per second)
            # tries the exact same submit and hits the exact same wall.
            # The 10 s cooldown means a transient book condition (e.g. the
            # spread we computed midpoint from went tight or moved) has
            # time to clear before we retry.
            self._last_entry_ns_per_market[slug] = now_ns
            return None
        self._stats["submits_ok"] += 1
        self._last_entry_ns_per_market[slug] = now_ns
        self._open_count_per_market[slug] += 1

        order_id = receipt.order_id or f"DRY-{slug}-{now_ns}"
        resting = _RestingOrder(
            order_id=order_id,
            market_slug=slug,
            down_token_id=str(down_token_id),
            limit_price=limit,
            size_shares=float(size_shares),
            notional_usd=float(target_notional),
            snapshot_ts_ns=int(snap.get("snapshot_ts_ns") or 0),
            submitted_at_ns=now_ns,
            cancel_deadline_ns=cancel_deadline,
            market_close_ts_ns=market_close_ns,
            edge_dn=float(decision.get("edge_dn") or 0.0),
            p_up=float(decision.get("p_up") or 0.0),
            filled_size=float(receipt.filled_size or 0.0),
            avg_fill_price=float(receipt.avg_price or limit),
        )
        self._open_orders[order_id] = resting

        # Some POST_ONLY orders fill partially on submit (rare). If filled
        # size > 0, record it immediately so we own the position from the
        # earliest moment.
        if resting.filled_size > 0:
            self._record_fill(resting, source="submit_partial")

        self.logger.info("MAKER POSTED %s limit=%.3f size=%.4f order_id=%s",
                         slug, limit, size_shares, order_id)
        self._write_log(rec)
        return resting

    # ── reconcile loop ──────────────────────────────────────────────────────
    async def run_reconcile_loop(self) -> None:
        """Background task: periodically cancel expired orders and check
        for fills on the live ones. Stays light — only hits the network
        when there's something to do."""
        self.logger.info("maker_strategy reconcile loop started "
                         "(lifetime=%.0fs, reconcile_every=%.1fs)",
                         self._order_lifetime_s, self._reconcile_interval_s)
        while not self._shutdown.is_set():
            try:
                self._reconcile_once(_now_ns())
            except Exception as exc:
                self.logger.warning("maker_strategy reconcile error: %r", exc)
            try:
                await asyncio.wait_for(self._shutdown.wait(),
                                       timeout=self._reconcile_interval_s)
            except asyncio.TimeoutError:
                pass

    def _reconcile_once(self, now_ns: int) -> None:
        if not self._open_orders:
            return
        # Snapshot keys -- _open_orders may mutate inside the loop
        for order_id in list(self._open_orders.keys()):
            resting = self._open_orders.get(order_id)
            if resting is None or resting.cancelled:
                continue
            if now_ns < resting.cancel_deadline_ns:
                continue
            # Time to pull this one. Cancel returns the final state.
            self._stats["cancels_called"] += 1
            resp = self.router.cancel_order(order_id)
            resting.cancelled = True
            # Try to extract filled_size from the cancel response. If the
            # SDK didn't put it there, follow up with get_order.
            filled = _extract_filled(resp)
            if filled is None or filled == 0.0:
                # In dry-run or if cancel doesn't return fill info, ask
                # get_order. It's a single HTTP — cheap enough at cancel time.
                info = self.router.get_order(order_id)
                got = _extract_filled(info)
                if got is not None:
                    filled = got
            if filled and filled > 0:
                # Update resting + record as position
                resting.filled_size = float(filled)
                resting.avg_fill_price = float(resting.limit_price)
                self._record_fill(resting, source="cancel_with_partial")
            else:
                resting.final_status = "cancelled_unfilled"
                self._write_log({
                    "logged_at_ns": now_ns, "strategy": "maker",
                    "snapshot_ts_ns": resting.snapshot_ts_ns,
                    "market_slug": resting.market_slug,
                    "order_id": order_id,
                    "decision": "CANCELLED_UNFILLED",
                    "limit_price": resting.limit_price,
                    "submitted_at_ns": resting.submitted_at_ns,
                    "cancel_deadline_ns": resting.cancel_deadline_ns,
                })
            # Free the per-market slot regardless of fill outcome.
            self._open_count_per_market[resting.market_slug] = max(
                0, self._open_count_per_market[resting.market_slug] - 1
            )
            del self._open_orders[order_id]

    def _record_fill(self, resting: _RestingOrder, *, source: str) -> None:
        """Translate a maker fill into a tracked position so the existing
        reconcile_resolved -> realised-PnL path fires when the market
        closes."""
        actual_notional = float(resting.filled_size) * float(resting.avg_fill_price)
        date_str = datetime.fromtimestamp(_now_ns() / 1e9, UTC).date().isoformat()
        fc = self._fee_calc_cache.setdefault(date_str, FeeCalculator.for_date(date_str))
        # Polymarket maker fee == 0; rebate available but we don't claim it
        # in the PnL to stay pessimistic on fees.
        fee_estimate = fc.maker_fee_usd(price=resting.avg_fill_price,
                                        size=resting.filled_size)
        self.risk_state.add_lot(close_ns=resting.market_close_ts_ns,
                                notional=actual_notional)
        self.risk_state.positions_per_market[resting.market_slug] += 1
        self.risk_state.last_entry_ns_per_market[resting.market_slug] = _now_ns()
        position = OpenPosition(
            market_slug=resting.market_slug,
            down_token_id=resting.down_token_id,
            shares=float(resting.filled_size),
            entry_price=float(resting.avg_fill_price),
            notional_usd=actual_notional,
            fee_usd_estimated=fee_estimate,
            snapshot_ts_ns=_now_ns(),
            market_close_ts_ns=resting.market_close_ts_ns,
            order_id=resting.order_id,
            edge_dn=resting.edge_dn,
        )
        self.positions.add_position(position)
        self._stats["fills_reconciled"] += 1
        self._stats["filled_shares_total"] += float(resting.filled_size)
        self._stats["filled_notional_total"] += float(actual_notional)
        self.logger.info("MAKER FILL %s @ %.3f x%.4f (notional=$%.2f, src=%s)",
                         resting.market_slug, resting.avg_fill_price,
                         resting.filled_size, actual_notional, source)
        self._write_log({
            "logged_at_ns": _now_ns(),
            "strategy": "maker",
            "snapshot_ts_ns": resting.snapshot_ts_ns,
            "market_slug": resting.market_slug,
            "order_id": resting.order_id,
            "decision": "FILLED",
            "fill_source": source,
            "limit_price": resting.limit_price,
            "fill_price": resting.avg_fill_price,
            "shares": resting.filled_size,
            "notional_usd": actual_notional,
            "fee_estimate_usd": fee_estimate,
            "edge_dn": resting.edge_dn,
            "p_up": resting.p_up,
        })

    # ── logging ─────────────────────────────────────────────────────────────
    def _write_log(self, record: dict) -> None:
        try:
            date_iso = datetime.fromtimestamp(record.get("logged_at_ns", _now_ns()) / 1e9, UTC)
            path = self.decisions_dir / f"decisions_maker_{date_iso.date().isoformat()}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception as exc:
            self.logger.warning("maker log write failed: %r", exc)


def _extract_filled(resp: dict | None) -> float | None:
    """Best-effort fish for a filled-size number out of a Polymarket
    response dict (various shapes between cancel/get/post endpoints)."""
    if not isinstance(resp, dict) or resp.get("error"):
        return None
    for key in ("filledSize", "filled_size", "size_matched", "filledAmount",
                "filled_amount", "matchedSize"):
        v = resp.get(key)
        if v is not None:
            try: return float(v)
            except (TypeError, ValueError): pass
    # Maker amount (USDC) / price -> shares
    maker = resp.get("makingAmount") or resp.get("makerAmount")
    price = resp.get("price") or resp.get("limit_price")
    if maker is not None and price is not None:
        try:
            m, p = float(maker), float(price)
            if m > 0 and p > 0:
                return m / p
        except (TypeError, ValueError): pass
    return None
