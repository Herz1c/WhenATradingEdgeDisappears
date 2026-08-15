"""Direct Polymarket market-WebSocket client for the live bot.

Purpose: eliminate the recorder->disk->tail latency hop for L2/book
events on markets we are actively trading. The recorder also subscribes
to the public market WS, writes events to zstd-jsonl on disk, then the
live_bot tails those files. End-to-end that adds ~50 ms (recorder flush)
+ ~50 ms (tail poll) of latency on top of the network roundtrip itself.
On a 5-min UpDown market where prices move on every binance tick, that
~100 ms penalty is enough to flip the bot from "fills the profitable
side" to "only fills the side that has already moved against us"
(textbook latency-driven adverse selection).

This module connects DIRECTLY to wss://ws-subscriptions-clob.polymarket.com/ws/market
from the bot process and feeds L2 records to the runtime as soon as they
arrive on the wire. The record shape is byte-for-byte the same as what
the recorder writes to the L2 zst files, so feature_runtime.feed_l2 has
no idea where they came from.

What we copy from the recorder (proven, do not reinvent):
  * The websocket URL / subscribe message envelope.
  * PolymarketL2StateTracker -- reconstructs book state from price_change
    deltas, exact same code path the recorder uses, so the resulting
    "best_bid"/"best_ask" sequence is identical.
  * The reconnect-with-exponential-backoff loop pattern.
  * PING/PONG heartbeat semantics.

The file-tail ingest stays running as a fallback: if this direct WS hits
trouble (transient network glitch, Polymarket rate limit), the bot still
sees L2 events via the recorder a few hundred ms later.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import orjson
from websockets import connect

from market_recorders.time_utils import recv_timestamp_source, utc_now_ns
from polymarket_recorder.l2 import PolymarketL2StateTracker
from polymarket_recorder.utils import build_polymarket_source_timestamps


_DEFAULT_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass(slots=True)
class _MarketSub:
    """The minimum info the direct WS needs to mint records the
    streaming engine recognises by slug + outcome."""
    market_slug: str
    market_id: str
    condition_id: str
    up_asset_id: str
    down_asset_id: str
    # asset_id -> outcome label as the streaming engine matches ("Up"/"Down")
    outcome_map: dict[str, str] = field(default_factory=dict)


class PolyDirectWS:
    """Owns one persistent WS connection to Polymarket's market channel
    and forwards every L2/book event to an on_l2(record) callback in the
    same shape the recorder writes to disk."""

    def __init__(
        self,
        *,
        on_l2: Callable[[dict[str, Any]], None],
        logger: logging.Logger | None = None,
        ws_url: str = _DEFAULT_WS_URL,
        reconnect_min_seconds: float = 1.0,
        reconnect_max_seconds: float = 30.0,
        heartbeat_interval_seconds: float = 10.0,
        heartbeat_timeout_seconds: float = 45.0,
        recv_timeout_seconds: float = 30.0,
        resync_interval_seconds: float | None = None,
    ) -> None:
        self._on_l2 = on_l2
        self.logger = logger or logging.getLogger("poly_ws_direct")
        self._ws_url = ws_url
        self._reconnect_min = reconnect_min_seconds
        self._reconnect_max = reconnect_max_seconds
        self._heartbeat_interval = heartbeat_interval_seconds
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._recv_timeout = recv_timeout_seconds
        # PERIODIC BOOK RESYNC: the book is reconstructed incrementally from
        # price_change deltas; a single dropped delta (queue overflow / reconnect
        # gap) makes it drift PERMANENTLY -> a desynced top-of-book -> bad
        # implied_p_up -> bullshit trades. Every `resync_interval_seconds` we
        # re-request a full `initial_dump` snapshot, which the tracker's
        # `_apply_book` overwrites the (possibly drifted) state with -> drift is
        # bounded to one interval, self-healing, no disconnect. 0/None disables.
        # NOTE: the time-based SOFT re-subscribe resync is now OFF by default
        # (FV_BOOK_RESYNC_S=0) — it did NOT make the server re-dump for already-
        # subscribed assets, so the book drifted (corr~0 vs recorder). The bot
        # now drives a HARD reconnect (request_resync) ~15s before each window.
        if resync_interval_seconds is None:
            resync_interval_seconds = float(os.getenv("FV_BOOK_RESYNC_S", "0"))
        self._resync_interval = resync_interval_seconds

        # market_slug -> _MarketSub (used to look up outcome / slug when an
        # event for one of its asset_ids arrives)
        self._markets: dict[str, _MarketSub] = {}
        # asset_id -> _MarketSub (fast lookup path on each event)
        self._asset_to_market: dict[str, _MarketSub] = {}

        # State tracker -- same code the recorder uses, so reconstructed
        # books are bit-identical.
        self._tracker = PolymarketL2StateTracker()
        self._seeded_assets: set[str] = set()

        # Set whenever the subscription needs to be (re-)sent.
        self._pending_resubscribe = asyncio.Event()
        # HARD-RESYNC trigger: set by the bot ~15s before a market's trading
        # window. Makes _consume return cleanly -> run() immediately reconnects
        # (no backoff) -> the server re-dumps a FRESH authoritative book for all
        # assets. This is the reliable resync (the soft re-subscribe did NOT
        # re-dump, so the book drifted with corr~0 vs the recorder).
        self._force_resync = asyncio.Event()
        # Most-recent activity timestamp for heartbeat dead-connection detection.
        self._last_activity_ns = 0
        # Local message counter -- mirrors the recorder's local_msg_seq.
        self._local_seq = 0
        # The single in-flight ws_url we logged for each emitted record.
        self._connection_id: str = ""
        self._shutdown = asyncio.Event()

        # Stats for the bot's stats_loop to surface.
        self.stats = {
            "ws_connects": 0,
            "ws_reconnects": 0,
            "ws_messages_received": 0,
            "ws_records_emitted": 0,
            "ws_unseeded_records_skipped": 0,
            "ws_resyncs": 0,
            "ws_last_error": None,
        }

    # ── public API used by LiveBot ────────────────────────────────────────────
    def add_market(self, *, market_slug: str, market_id: str, condition_id: str,
                   up_asset_id: str, down_asset_id: str) -> None:
        """Register a market's tokens for live subscription. Safe to call
        before the WS is connected (we'll subscribe on first connect) and
        after (we set the resubscribe flag and the loop fires a subscribe
        message)."""
        if market_slug in self._markets:
            return
        sub = _MarketSub(
            market_slug=market_slug,
            market_id=market_id,
            condition_id=condition_id,
            up_asset_id=up_asset_id,
            down_asset_id=down_asset_id,
            outcome_map={up_asset_id: "Up", down_asset_id: "Down"},
        )
        self._markets[market_slug] = sub
        if up_asset_id:   self._asset_to_market[up_asset_id]   = sub
        if down_asset_id: self._asset_to_market[down_asset_id] = sub
        self._pending_resubscribe.set()
        self.logger.info("poly_ws_direct: market added %s (assets=%d)",
                         market_slug, len(self._asset_to_market))

    def remove_market(self, market_slug: str) -> None:
        """Unsubscribe a market (e.g. once it has closed). Drops its tokens and
        flags a resubscribe; the consume loop then reconnects with the reduced
        authoritative asset list so Polymarket stops sending its events."""
        sub = self._markets.pop(market_slug, None)
        if sub is None:
            return
        self._asset_to_market.pop(sub.up_asset_id, None)
        self._asset_to_market.pop(sub.down_asset_id, None)
        # Free the closed market's book state — without this the tracker's
        # per-asset state dict grows unbounded over a long run (memory leak ->
        # slowdown -> dropped deltas -> drift). This is the time-cumulative
        # degradation source.
        self._tracker.drop_asset(sub.up_asset_id)
        self._tracker.drop_asset(sub.down_asset_id)
        self._pending_resubscribe.set()
        self.logger.info("poly_ws_direct: market removed %s (assets=%d)",
                         market_slug, len(self._asset_to_market))

    def request_resync(self) -> None:
        """Ask for a HARD reconnect ASAP (fresh authoritative book dump). Called
        by the bot shortly before a market's trading window opens so decisions
        run on a freshly-seeded, un-drifted book."""
        self._force_resync.set()

    def stop(self) -> None:
        self._shutdown.set()

    def _all_asset_ids(self) -> list[str]:
        return sorted({a for a in self._asset_to_market.keys() if a})

    # ── connection lifecycle ──────────────────────────────────────────────────
    async def run(self) -> None:
        """Top-level loop: connect, consume, on error backoff + retry."""
        delay = self._reconnect_min
        while not self._shutdown.is_set():
            assets = self._all_asset_ids()
            if not assets:
                # No markets to subscribe to yet -- the bot is still
                # discovering them via REST/strike. Wait a moment.
                try:
                    await asyncio.wait_for(self._pending_resubscribe.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                self._pending_resubscribe.clear()
                continue
            try:
                await self._consume()
                delay = self._reconnect_min
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats["ws_last_error"] = repr(exc)
                self.stats["ws_reconnects"] += 1
                self.logger.warning("poly_ws_direct: connection error %r — backoff %.1fs", exc, delay)
                await asyncio.sleep(delay + random.random() * 1.5)
                delay = min(self._reconnect_max, max(self._reconnect_min, delay * 2))

    async def _consume(self) -> None:
        assets = self._all_asset_ids()
        if not assets:
            return
        # Treat each connection as a hard book-snapshot boundary. Starting the
        # tracker clean prevents stale reconstructed state from surviving a
        # reconnect if Polymarket sends deltas before the fresh dump.
        self._tracker = PolymarketL2StateTracker()
        self._seeded_assets = set()
        self._connection_id = f"bot-poly-{utc_now_ns():x}"
        self.stats["ws_connects"] += 1
        self.logger.info("poly_ws_direct: connecting to %s for %d asset(s)",
                         self._ws_url, len(assets))
        async with connect(
            self._ws_url,
            open_timeout=15.0,
            close_timeout=5.0,
            max_queue=1024,
            ping_interval=None,
        ) as ws:
            await ws.send(orjson.dumps({
                "assets_ids": assets,
                "type": "market",
                "custom_feature_enabled": True,
                "initial_dump": True,
            }).decode("utf-8"))
            self._pending_resubscribe.clear()
            self._last_activity_ns = utc_now_ns()
            self.logger.info("poly_ws_direct: connected, subscribed to %d asset(s)", len(assets))
            heartbeat = asyncio.create_task(self._heartbeat(ws), name="poly_ws_direct:heartbeat")
            resync = asyncio.create_task(self._resync_loop(ws), name="poly_ws_direct:resync")
            try:
                while not self._shutdown.is_set():
                    # Re-subscribe whenever a new market is added.
                    if self._pending_resubscribe.is_set():
                        new_assets = self._all_asset_ids()
                        if new_assets != assets:
                            # Keep one clean connection per authoritative asset
                            # set. Soft subscribe can duplicate existing assets
                            # or interleave fresh dumps with old deltas.
                            self.logger.info(
                                "poly_ws_direct: asset set changed %d -> %d; reconnecting cleanly",
                                len(assets), len(new_assets))
                            self._pending_resubscribe.clear()
                            return
                            removed = set(assets) - set(new_assets)
                            if removed:
                                # Removals (closed markets): Polymarket's market
                                # WS has no clean per-asset unsubscribe, so do a
                                # fast reconnect with the authoritative reduced
                                # list. run() reconnects immediately. This fires
                                # right after a close boundary — the safest time
                                # (no market is near its trading window then).
                                self.logger.info(
                                    "poly_ws_direct: %d asset(s) removed — reconnecting with %d",
                                    len(removed), len(new_assets))
                                self._pending_resubscribe.clear()
                                return
                            await ws.send(orjson.dumps({
                                "operation": "subscribe",
                                "assets_ids": new_assets,
                                "custom_feature_enabled": True,
                                "initial_dump": True,
                            }).decode("utf-8"))
                            assets = new_assets
                            self.logger.info("poly_ws_direct: resubscribed to %d asset(s)", len(assets))
                        self._pending_resubscribe.clear()
                    # HARD resync requested (pre-window): return cleanly so run()
                    # reconnects immediately (no backoff) and the server re-dumps
                    # a fresh book for every asset.
                    if self._force_resync.is_set():
                        self._force_resync.clear()
                        self.stats["ws_resyncs"] += 1
                        self.logger.info("poly_ws_direct: HARD resync — reconnecting for fresh book dump")
                        return
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=self._recv_timeout)
                    except asyncio.TimeoutError:
                        # No traffic -- heartbeat task will decide whether to close.
                        continue
                    self._last_activity_ns = utc_now_ns()
                    if message in {"PONG", "{}", b"{}"}:
                        continue
                    self.stats["ws_messages_received"] += 1
                    try:
                        payload = orjson.loads(message)
                    except Exception as exc:
                        self.logger.warning("poly_ws_direct: bad json %r", exc)
                        continue
                    for item in self._iter_items(payload):
                        self._handle_payload(item)
            finally:
                heartbeat.cancel()
                resync.cancel()
                for t in (heartbeat, resync):
                    try: await t
                    except asyncio.CancelledError: pass

    async def _resync_loop(self, ws: Any) -> None:
        """Periodically re-request a full `initial_dump` book snapshot so the
        incrementally-reconstructed book can't drift indefinitely. The server
        re-sends `book` snapshots for the current assets, and the tracker's
        `_apply_book` overwrites any drifted state -> self-healing top-of-book.
        Keeps the connection alive (no disconnect)."""
        if not self._resync_interval or self._resync_interval <= 0:
            return
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(self._resync_interval)
                assets = self._all_asset_ids()
                if not assets:
                    continue
                try:
                    # Same payload shape as the initial connect subscribe — the
                    # form proven to make the server emit a full `book` dump.
                    await ws.send(orjson.dumps({
                        "assets_ids": assets,
                        "type": "market",
                        "custom_feature_enabled": True,
                        "initial_dump": True,
                    }).decode("utf-8"))
                    self.stats["ws_resyncs"] += 1
                    self.logger.debug("poly_ws_direct: book resync requested (%d asset(s))", len(assets))
                except Exception:
                    return
        except asyncio.CancelledError:
            return

    async def _heartbeat(self, ws: Any) -> None:
        """Send PING periodically and close the socket if no traffic comes
        back -- proxies/load-balancers sometimes silently drop idle conns."""
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(self._heartbeat_interval)
                idle_ns = utc_now_ns() - self._last_activity_ns
                if idle_ns > int(self._heartbeat_timeout * 1_000_000_000):
                    self.logger.warning("poly_ws_direct: %.1fs idle — closing for reconnect",
                                        idle_ns / 1e9)
                    await ws.close()
                    return
                try:
                    await ws.send("PING")
                except Exception:
                    return
        except asyncio.CancelledError:
            return

    # ── payload handling ──────────────────────────────────────────────────────
    @staticmethod
    def _iter_items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        return []

    def _resolve_market(self, payload: dict[str, Any]) -> _MarketSub | None:
        """Find the market this event belongs to via its asset_id(s).
        For 'price_changes' events we need the per-change asset_id; for
        the others the event has a top-level asset_id."""
        aid = payload.get("asset_id")
        if aid and (sub := self._asset_to_market.get(str(aid))) is not None:
            return sub
        for change in payload.get("price_changes", []) or []:
            cid = change.get("asset_id") if isinstance(change, dict) else None
            if cid and (sub := self._asset_to_market.get(str(cid))) is not None:
                return sub
        return None

    def _handle_payload(self, payload: dict[str, Any]) -> None:
        sub = self._resolve_market(payload)
        if sub is None:
            return
        # Stamp recv_ts_ns ONCE at the bot, mirroring how the recorder
        # stamps it once at receive. We're the bot here, so this is the
        # earliest possible wall-clock for this event in this process.
        recv_ts_ns = utc_now_ns()
        # Compute once per payload; identical value was previously rebuilt per
        # emitted record (and once more inside the tracker).
        source_timestamps = build_polymarket_source_timestamps(payload)
        # Run the L2 state machine -- this is the recorder's proven
        # PolymarketL2StateTracker. It emits zero, one, or many records
        # depending on the event type.
        try:
            l2_records = self._tracker.process_event(payload, source_timestamps=source_timestamps)
        except Exception as exc:
            self.logger.warning("poly_ws_direct: tracker error %r", exc)
            return
        if not l2_records:
            return
        self._local_seq += 1
        for l2_record in l2_records:
            token_asset_id = str(l2_record.get("token_asset_id") or "")
            outcome = sub.outcome_map.get(token_asset_id)
            if outcome is None:
                continue
            if l2_record.get("event_type") == "book":
                self._seeded_assets.add(token_asset_id)
            elif token_asset_id not in self._seeded_assets:
                self.stats["ws_unseeded_records_skipped"] += 1
                continue
            out = {
                "source":               "polymarket",
                "channel":              "market",
                "connection_id":        self._connection_id,
                "local_msg_seq":        self._local_seq,
                "ws_url":               self._ws_url,
                "recv_ts_ns":           recv_ts_ns,
                "recv_ts_source":       recv_timestamp_source(),
                "selected_market_slug": sub.market_slug,
                "selected_market_id":   sub.market_id,
                "selected_condition_id": sub.condition_id,
                **l2_record,
                "token_outcome":        outcome,
                "token_modeling_safe":  True,
                "timestamp_semantics":  "source_metadata",
                "source_timestamps":    source_timestamps,
                "direct_ws":            True,   # flag so we can spot it in logs / audit
            }
            try:
                self._on_l2(out)
            except Exception as exc:
                self.logger.warning("poly_ws_direct: on_l2 callback failed %r", exc)
                continue
            self.stats["ws_records_emitted"] += 1
