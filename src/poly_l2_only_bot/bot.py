"""poly_l2_only_bot orchestrator.

Wires together:
  - BTCUpDownDiscovery  -- Gamma polling for upcoming markets
  - PolyDirectWS        -- direct WS feed of L2 frames (recorder's tested path)
  - V2Predictor         -- LightGBM v2 model with init_score residual head
  - decide()            -- pure decision logic from strategy.py
  - PolymarketOrderRouter -- existing order submission infrastructure
  - AsyncJsonlLogger    -- non-blocking decision/order/error logging

HOT PATH (per WS frame):
  1. asset_id -> MarketCtx lookup (dict, O(1))
  2. MarketCtx.done early-out
  3. update_state(state, frame)  (~10-100 us)
  4. TTC range check; return if outside [25, 60)s
  5. predictor.fill_features + predict_one  (~1-3 ms; dominates the path)
  6. decide(...)  (pure, ~5 us)
  7. If decision returned: mark strat_state, fire order in background task,
     log decision to async logger
  8. Return

Everything that can block (HTTP order submit, disk I/O, JSON serialization)
is off-thread or off-coroutine. The receive loop never waits on anything but
the next WS frame.

Run:
  py -3 -m poly_l2_only_bot                                 # DRY-RUN
  $env:ENABLE_REAL_ORDERS="1"; py -3 -m poly_l2_only_bot   # live, gated by key
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_bot.order_router import PolymarketCreds, PolymarketOrderRouter
from live_bot.poly_ws_direct import PolyDirectWS
from poly_l2_only.extractor import (
    EMIT_EVENT_TYPES, MarketState, NS_PER_S,
    state_to_features, update_state,
)
from poly_l2_only_bot.async_logger import AsyncJsonlLogger
from poly_l2_only_bot.discovery import BTCUpDownDiscovery, DiscoveredMarket
from poly_l2_only_bot.predictor import V2Predictor
from poly_l2_only_bot.strategy import (
    COOLDOWN_NS, MAX_ENTRIES_PER_MARKET, NOTIONAL_USD,
    ORDER_MODE_GTC,
    STRATEGY_FROZEN_DATE, STRATEGY_VERSION, TTC_MAX_S, TTC_MIN_S,
    Decision, MarketStrategyState, decide,
)

ARTIFACT_DIR_DEFAULT = REPO_ROOT / "artifacts_cleaned" / "poly_l2_only_v2"
LOG_DIR_DEFAULT = REPO_ROOT / "logs" / "poly_l2_only_bot"

# Disconnect a market this many seconds AFTER its close (ttc=0). The reserve
# avoids cutting off any straggler in-window events right at the boundary.
CLEANUP_GRACE_S = 30


def _utc_now_ns() -> int:
    return time.time_ns()


# ── Colored console output (green when the bot places an order) ──────────────
_GREEN = "\033[92m"
_RED = "\033[91m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _enable_windows_ansi() -> None:
    """Turn on ANSI/VT processing in the Windows console so green text shows."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def _cprint(color: str, msg: str) -> None:
    print(f"{color}{msg}{_RESET}", flush=True)


@dataclass(slots=True)
class MarketCtx:
    """Per-market context held in the bot. Pre-allocated; reused on every frame."""
    slug: str
    market_id: str
    condition_id: str
    up_asset_id: str
    down_asset_id: str
    market_open_s: int
    market_close_s: int
    state: MarketState = field(default_factory=MarketState)
    strat: MarketStrategyState = field(default_factory=MarketStrategyState)
    # Cached so we don't traverse asset_to_side per frame.
    close_ns_cached: int = 0


class PolyL2OnlyBot:
    def __init__(
        self,
        *,
        artifact_dir: Path,
        log_dir: Path,
        live_orders: bool,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("poly_l2_only_bot")
        self.predictor = V2Predictor(artifact_dir)
        self.logger.info("loaded model from %s (features=%d, best_iter=%d)",
                         artifact_dir, len(self.predictor.feature_order),
                         self.predictor.best_iter)

        # Markets we're trading. Both slug-keyed (for discovery) and
        # asset-id-keyed (for hot-path lookup) views point at the same MarketCtx.
        self._markets_by_slug: Dict[str, MarketCtx] = {}
        self._markets_by_asset: Dict[str, MarketCtx] = {}

        # Async loggers: separate streams so heavy decision logging doesn't
        # interleave with low-volume error / fill logs.
        self.log_decisions = AsyncJsonlLogger(log_dir=log_dir, basename="decisions")
        self.log_orders = AsyncJsonlLogger(log_dir=log_dir, basename="orders")
        self.log_events = AsyncJsonlLogger(log_dir=log_dir, basename="events")

        # Order infrastructure (reuses existing battle-tested router).
        self.creds = PolymarketCreds.from_keyfile_and_env()
        self.router = PolymarketOrderRouter(creds=self.creds, live=live_orders,
                                            logger=self.logger)

        # Direct WS pipe — feeds frames straight into self._on_l2_frame.
        self.direct_ws = PolyDirectWS(on_l2=self._on_l2_frame, logger=self.logger)

        # Market discovery — runs in background, calls _register_market.
        self.discovery = BTCUpDownDiscovery(
            on_new_market=self._register_market, logger=self.logger,
        )

        # Stats — printed periodically by the stats task.
        self._stats = {
            "frames_seen": 0,
            "frames_filtered_out": 0,    # wrong event_type
            "frames_no_ctx": 0,          # frame's market not registered
            "frames_market_done": 0,     # market already traded (done)
            "frames_update_fail": 0,     # update_state rejected the frame
            "frames_ttc_too_early": 0,   # ttc >= 50  (before window)
            "frames_ttc_too_late": 0,    # ttc < 10   (after window)
            "predictions": 0,            # frames that reached model scoring
            "decisions_logged": 0,
            "orders_attempted": 0,
            "orders_filled": 0,
            "markets_registered": 0,
        }
        self._shutdown = asyncio.Event()
        self._loop_ref: asyncio.AbstractEventLoop | None = None
        self._feat_dumps = 0   # one-shot live feature-vector diagnostics
        self.log_events.log({
            "event": "bot_init",
            "strategy_version": STRATEGY_VERSION,
            "strategy_frozen": STRATEGY_FROZEN_DATE,
            "artifact_dir": str(artifact_dir),
            "live_orders": bool(self.router.live),
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        })

    # ── market registration ────────────────────────────────────────────────
    async def _register_market(self, dm: DiscoveredMarket) -> None:
        """Called once per discovered market. Allocates the per-market ctx
        and wires the direct WS subscription."""
        if dm.market_slug in self._markets_by_slug:
            return
        ctx = MarketCtx(
            slug=dm.market_slug,
            market_id=dm.market_id,
            condition_id=dm.condition_id,
            up_asset_id=dm.up_asset_id,
            down_asset_id=dm.down_asset_id,
            market_open_s=dm.market_open_s,
            market_close_s=dm.market_close_s,
            close_ns_cached=dm.market_close_s * NS_PER_S,
        )
        # Pre-seed MarketState fields we already know from discovery so the
        # first frame doesn't have to populate them lazily.
        ctx.state.event_slug = dm.market_slug
        ctx.state.market_id = dm.market_id
        ctx.state.market_open_s = dm.market_open_s
        ctx.state.market_close_s = dm.market_close_s
        ctx.state.asset_to_side[dm.up_asset_id] = "Up"
        ctx.state.asset_to_side[dm.down_asset_id] = "Down"

        self._markets_by_slug[dm.market_slug] = ctx
        self._markets_by_asset[dm.up_asset_id] = ctx
        self._markets_by_asset[dm.down_asset_id] = ctx
        self._stats["markets_registered"] += 1

        # Subscribe in the direct WS so we start receiving L2 frames.
        self.direct_ws.add_market(
            market_slug=dm.market_slug,
            market_id=dm.market_id,
            condition_id=dm.condition_id,
            up_asset_id=dm.up_asset_id,
            down_asset_id=dm.down_asset_id,
        )
        self.log_events.log({
            "event": "market_registered",
            "slug": dm.market_slug,
            "close_s": dm.market_close_s,
            "ts_iso": datetime.now(timezone.utc).isoformat(),
        })

    # ── HOT PATH: WS frame -> decision -> order ───────────────────────────
    def _on_l2_frame(self, frame: Dict[str, Any]) -> None:
        """Called from the PolyDirectWS receive loop. Synchronous; do as
        little as possible here. Anything blocking (order HTTP) gets
        scheduled as a task off the loop."""
        self._stats["frames_seen"] += 1
        et = frame.get("event_type")
        if et not in EMIT_EVENT_TYPES:
            self._stats["frames_filtered_out"] += 1
            return
        slug = frame.get("selected_market_slug")
        ctx = self._markets_by_slug.get(slug) if slug else None
        if ctx is None:
            # Fall back to asset_id (in case slug isn't in this payload type).
            aid = frame.get("token_asset_id")
            if aid:
                ctx = self._markets_by_asset.get(str(aid))
        if ctx is None:
            self._stats["frames_no_ctx"] += 1
            return
        if ctx.strat.done:
            self._stats["frames_market_done"] += 1
            return

        # Update state. ALWAYS — so the book/trade buffers are correct
        # the moment we enter the trading window.
        if not update_state(ctx.state, frame):
            self._stats["frames_update_fail"] += 1
            return
        # update_state pulls market_open_s/close_s from the frame, but the
        # direct-WS frames don't carry them — so re-assert the authoritative
        # values from discovery (open=slug, close=open+300). Without this the
        # model's ttc_s feature collapses to 0.
        ctx.state.market_open_s = ctx.market_open_s
        ctx.state.market_close_s = ctx.market_close_s

        ts_ns = int(frame.get("recv_ts_ns") or _utc_now_ns())
        ttc_s = (ctx.close_ns_cached - ts_ns) / NS_PER_S
        if ttc_s < TTC_MIN_S:
            self._stats["frames_ttc_too_late"] += 1   # past the window (ttc<10)
            return
        if ttc_s >= TTC_MAX_S:
            self._stats["frames_ttc_too_early"] += 1  # before the window (ttc>=50)
            return

        # Compute features and predict.
        feats = state_to_features(ctx.state, ts_ns)
        self.predictor.fill_features(feats)
        p_up = self.predictor.predict_one()
        self._stats["predictions"] += 1

        # One-shot diagnostic: dump the first few in-window 55-feature vectors so
        # we can confirm live features (incl. trade features + ttc_s) are populated.
        if self._feat_dumps < 3:
            self._feat_dumps += 1
            order = self.predictor.feature_order
            nz = sum(1 for k in order if feats.get(k, 0.0))
            self.log_events.log({
                "event": "feature_dump", "slug": ctx.slug, "ttc_s": round(ttc_s, 2),
                "p_up": round(p_up, 4), "n_features": len(order), "n_nonzero": nz,
                "features": {k: round(float(feats.get(k, 0.0)), 5) for k in order},
            })

        # Decide. Pure function, fast.
        decision = decide(
            p_up=p_up,
            best_ask_up=feats["up_best_ask"],
            best_ask_down=feats["down_best_ask"],
            ttc_s=ttc_s,
            ts_ns=ts_ns,
            strat_state=ctx.strat,
        )
        if decision is None:
            return

        # Mark strat state BEFORE firing the order — prevents a second
        # decision from sneaking in while this order is in flight.
        ctx.strat.n_entries += 1
        ctx.strat.last_entry_ts_ns = ts_ns
        if ctx.strat.n_entries >= MAX_ENTRIES_PER_MARKET:
            ctx.strat.done = True

        # Log decision (non-blocking).
        self._stats["decisions_logged"] += 1
        self.log_decisions.log({
            "event": "decision",
            "slug": ctx.slug,
            "ts_ns": ts_ns,
            "ts_iso": datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat(),
            "ttc_s": round(decision.ttc_s, 3),
            "zone": decision.zone,
            "side": decision.side,
            "fill_price": decision.fill_price,
            "p_model": decision.p_model,
            "ev_per_share": decision.ev_per_share,
            "order_mode": decision.order_mode,
            "n_entries_after": ctx.strat.n_entries,
            "notional_usd": decision.notional_usd,
        })

        # Fire order asynchronously — never block the receive loop.
        if self._loop_ref is not None:
            self._loop_ref.call_soon_threadsafe(
                self._schedule_order, ctx, decision
            )

    def _schedule_order(self, ctx: MarketCtx, decision: Decision) -> None:
        """Schedules the order submission as an asyncio task. Called via
        call_soon_threadsafe so it's safe from the WS receive callback."""
        asyncio.create_task(self._submit_order_and_log(ctx, decision),
                            name=f"order:{ctx.slug}:{decision.side}")

    async def _submit_order_and_log(self, ctx: MarketCtx, decision: Decision) -> None:
        self._stats["orders_attempted"] += 1
        submit_start_ns = _utc_now_ns()
        token_id = ctx.up_asset_id if decision.side == "UP" else ctx.down_asset_id
        size_shares = decision.notional_usd / max(decision.fill_price, 1e-6)
        # GREEN console line the moment we fire an order — so it's obvious.
        tag = "DRY-RUN" if not self.router.live else "LIVE"
        _cprint(_GREEN,
                f">>> ORDER [{tag}] BUY {decision.side} @ {decision.fill_price:.2f} "
                f"${decision.notional_usd:.0f} | {ctx.slug} | ttc={decision.ttc_s:.0f}s "
                f"p={decision.p_model:.3f} ev={decision.ev_per_share:+.3f} (GTC)")
        # GTC crossing limit (taker): buys what it can, rests the remainder.
        # NEVER FAK/FOK.
        try:
            receipt = await asyncio.to_thread(
                self.router.submit_buy_gtc,
                token_id=token_id,
                limit_price=decision.fill_price,
                size_shares=size_shares,
                side_label=decision.side,
            )
        except Exception as exc:
            self.log_orders.log({
                "event": "order_exception",
                "slug": ctx.slug,
                "side": decision.side,
                "err": repr(exc),
                "ts_ns": _utc_now_ns(),
            })
            return
        submit_end_ns = _utc_now_ns()
        if receipt.success:
            self._stats["orders_filled"] += 1
        _cprint(
            _GREEN if receipt.success else _RED,
            f"    {'OK ' if receipt.success else 'ERR'} {decision.side} {ctx.slug} | "
            f"status={receipt.status} filled={receipt.filled_size:.4f} "
            f"@ {receipt.avg_price:.3f} "
            f"({(submit_end_ns - decision.ts_ns) / 1e6:.0f} ms)"
            + (f" err={receipt.err}" if receipt.err else ""))
        self.log_orders.log({
            "event": "order_result",
            "slug": ctx.slug,
            "side": decision.side,
            "zone": decision.zone,
            "order_mode": decision.order_mode,
            "fill_price_intended": decision.fill_price,
            "size_shares_intended": round(size_shares, 4),
            "success": receipt.success,
            "dry_run": receipt.dry_run,
            "order_id": receipt.order_id,
            "filled_size": receipt.filled_size,
            "avg_price": receipt.avg_price,
            "status": receipt.status,
            "err": receipt.err,
            "decision_ts_ns": decision.ts_ns,
            "submit_start_ns": submit_start_ns,
            "submit_end_ns": submit_end_ns,
            "submit_latency_ms": (submit_end_ns - submit_start_ns) / 1e6,
            "decision_to_fill_ms": (submit_end_ns - decision.ts_ns) / 1e6,
        })

    # ── housekeeping tasks ────────────────────────────────────────────────
    async def _stats_loop(self) -> None:
        """Periodically log stats. Cheap; runs every 30 s."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                pass
            self.log_events.log({
                "event": "bot_stats",
                "ts_iso": datetime.now(timezone.utc).isoformat(),
                "bot": dict(self._stats),
                "ws": dict(self.direct_ws.stats),
                "log_decisions": self.log_decisions.stats,
                "log_orders": self.log_orders.stats,
                "log_events": self.log_events.stats,
                "markets_active": len(self._markets_by_slug),
            })

    async def _market_cleanup_loop(self) -> None:
        """Disconnect markets shortly after they close (ttc reaches 0). We keep
        a CLEANUP_GRACE_S reserve past close so any late in-window events finish,
        then drop the market AND unsubscribe its tokens from the WS so we stop
        receiving (and processing) the post-close book frenzy. Runs every 15 s."""
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass
            now_s = int(time.time())
            stale = [s for s, ctx in self._markets_by_slug.items()
                     if ctx.market_close_s + CLEANUP_GRACE_S < now_s]
            for slug in stale:
                ctx = self._markets_by_slug.pop(slug, None)
                if ctx is None:
                    continue
                self._markets_by_asset.pop(ctx.up_asset_id, None)
                self._markets_by_asset.pop(ctx.down_asset_id, None)
                # Unsubscribe from the WS — stop the post-close frame stream.
                self.direct_ws.remove_market(ctx.slug)
            if stale:
                self.log_events.log({
                    "event": "markets_pruned",
                    "n_pruned": len(stale),
                    "n_active": len(self._markets_by_slug),
                    "ts_iso": datetime.now(timezone.utc).isoformat(),
                })

    # ── lifecycle ─────────────────────────────────────────────────────────
    def request_stop(self) -> None:
        self._shutdown.set()
        self.discovery.stop()
        self.direct_ws.stop()

    async def run(self) -> None:
        self._loop_ref = asyncio.get_running_loop()
        # Set up signal handlers.
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._loop_ref.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                pass  # Windows / non-Unix
        if self.router.live:
            try:
                report = await asyncio.to_thread(self.router.preflight)
                self.log_events.log({"event": "preflight", **report,
                                     "ts_iso": datetime.now(timezone.utc).isoformat()})
            except Exception as exc:
                self.logger.warning("preflight failed: %r", exc)
        tasks = [
            asyncio.create_task(self.discovery.run(), name="discovery"),
            asyncio.create_task(self.direct_ws.run(), name="direct_ws"),
            asyncio.create_task(self._stats_loop(), name="stats"),
            asyncio.create_task(self._market_cleanup_loop(), name="cleanup"),
        ]
        try:
            await self._shutdown.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.log_events.log({
                "event": "bot_shutdown",
                "ts_iso": datetime.now(timezone.utc).isoformat(),
                "final_stats": dict(self._stats),
            })
            self.log_decisions.stop()
            self.log_orders.stop()
            self.log_events.stop()


def setup_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s :: %(message)s",
    )
    return logging.getLogger("poly_l2_only_bot")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR_DEFAULT)
    ap.add_argument("--log-dir", type=Path, default=LOG_DIR_DEFAULT)
    ap.add_argument("--live-orders", action="store_true",
                    help="Allow real orders (also requires ENABLE_REAL_ORDERS=1 + key)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logger = setup_logging(args.log_level)
    _enable_windows_ansi()
    from poly_l2_only_bot.strategy import (
        FILL_MAX, FILL_MIN, MIN_EDGE, TTC_MAX_S as _TMAX, TTC_MIN_S as _TMIN,
    )
    mode = "LIVE ORDERS" if (args.live_orders and os.environ.get("ENABLE_REAL_ORDERS") == "1") else "DRY-RUN"
    _cprint(_GREEN if mode == "DRY-RUN" else _RED,
            f"poly_l2_only_bot  [{mode}]  strategy={STRATEGY_VERSION} ({STRATEGY_FROZEN_DATE})")
    _cprint(_DIM,
            f"  1 entry/market | ttc ({_TMIN:.0f},{_TMAX:.0f}]s | edge>={MIN_EDGE} | "
            f"fill[{FILL_MIN},{FILL_MAX}] | ${NOTIONAL_USD:.0f} | GTC (no FAK/FOK) | "
            f"source: Polymarket L2 only")
    bot = PolyL2OnlyBot(
        artifact_dir=args.artifact_dir,
        log_dir=args.log_dir,
        live_orders=args.live_orders,
        logger=logger,
    )
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
