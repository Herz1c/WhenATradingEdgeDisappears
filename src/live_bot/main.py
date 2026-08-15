"""Live trading bot — orchestrator.

The bot:

  1. Tails the raw recorder files (Polymarket L2 + REST,
     Binance spot/usdm bookTicker, Hyperliquid bbo).
  2. Builds an in-memory LiveBtcState that produces a per-second
     synthetic_corrected price from the live sources.
  3. Feeds Polymarket L2/REST events into a StreamingFeatureEngine,
     and also registers each new active market as the recorder
     discovers it.
  4. At each dense_close grid time (T-60s ... T, 250ms cadence then
     100ms for the last 10s), fires `_build_snapshot_row` over the
     current indices, runs Model 02, runs the decision engine.
  5. If decision == ENTER and the risk gate is clear, hands the order
     to PolymarketOrderRouter (which is DRY-RUN unless
     ENABLE_REAL_ORDERS=1 + a private key are present).
  6. Logs every decision to logs/live_bot/decisions_<date>.jsonl.
  7. Periodically reconciles open positions against the resolution
     shards and writes realized PnL to closed_positions.jsonl.

Hard guarantees:
  * NEVER fires snapshot for ttc > 60s (feature_runtime gate).
  * NEVER submits real orders unless ENABLE_REAL_ORDERS=1 AND
    POLYMARKET_PRIVATE_KEY is set (order_router gate).
  * Stops new entries the moment STOP_TRADING file appears.

Run:
  py -3 -m live_bot.main                          # DRY-RUN, default
  $env:ENABLE_REAL_ORDERS="1"; py -3 -m live_bot.main   # live, gated by key

Stop:
  Ctrl-C  (graceful, lets open positions resolve)
  touch STOP_TRADING  (halts new entries; in-flight positions resolve naturally)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Quiet a sklearn warning that fires once per model inference. The
# LightGBM model was trained on a DataFrame with column names; we
# pass it a NumPy array (faster for single-row live inference). The
# numeric output is identical — sklearn is just verbose about it.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from backtest.fees import FeeCalculator
from live_bot.btc_state import LiveBtcState
from live_bot.decision_engine import RiskState, SECOND_NS, TTC_MIN_S, TTC_MAX_S
from live_bot.feature_runtime import FeatureRuntime
from live_bot.file_tail import MultiSourceTail
from live_bot.maker_strategy import MakerStrategy
from live_bot.order_router import PolymarketCreds, PolymarketOrderRouter
from live_bot.poly_ws_direct import PolyDirectWS
# When the active model doesn't depend on the chainlink-residual bias
# (i.e. the no-bias calibrated v1), all bias compute is wasted work --
# CPU + disk + chainlink HTTP polls -- so we short-circuit every bias
# path on this flag.
from live_bot.feature_runtime import _MODEL_REQUIRES_CALIBRATOR
_BIAS_NEEDED = not _MODEL_REQUIRES_CALIBRATOR
from live_bot.position_manager import OpenPosition, PositionManager
from live_bot.risk_gate import RiskGate
from polymarket_recorder.dataset_factory import MarketLabel


LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s :: %(message)s"


def _ts_iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat(timespec="milliseconds")


def _now_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1e9)


class LiveBot:
    def __init__(self, *, decisions_dir: Path = Path("logs/live_bot"),
                 raw_root: Path = Path("data"),
                 live_orders: bool = False,
                 logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("live_bot")
        self.raw_root = raw_root
        self.decisions_dir = decisions_dir
        decisions_dir.mkdir(parents=True, exist_ok=True)

        self.btc = LiveBtcState()
        # Point bot at the canonical parquet that tools/canonical_daemon.py rebuilds.
        # Path is RE-COMPUTED in the tick loop based on current UTC date, so it
        # rolls over at midnight without restart.
        self._canonical_dir = (
            raw_root.parent / "data" / "canonical" / "live_reference_events_v1"
            if raw_root.name != "data" else
            raw_root / "canonical" / "live_reference_events_v1"
        )
        self._refresh_canonical_path()
        self.risk_state = RiskState()
        self.risk_gate = RiskGate()
        self.positions = PositionManager(raw_root=raw_root)
        self.creds = PolymarketCreds.from_keyfile_and_env()
        self.router = PolymarketOrderRouter(creds=self.creds, live=live_orders)
        self.runtime = FeatureRuntime(btc=self.btc, risk=self.risk_state,
                                      on_decision=self._handle_decision, logger=self.logger)

        # Direct Polymarket WS for L2 events on markets we're trading.
        # Eliminates the recorder->disk->tail (~50-150 ms) hop. File tail
        # stays active as a fallback. Disable with LIVE_BOT_DIRECT_WS=0.
        self._direct_ws_enabled = os.getenv("LIVE_BOT_DIRECT_WS", "1") == "1"
        self.direct_ws: PolyDirectWS | None = None
        if self._direct_ws_enabled:
            self.direct_ws = PolyDirectWS(
                on_l2=self.runtime.feed_l2, logger=self.logger,
            )

        # Strategy dispatch. Default unchanged: only the existing FAK taker
        # path runs. Set LIVE_BOT_STRATEGY=maker to swap to POST_ONLY-at-
        # midpoint, or =both to run both in parallel (each with its OWN
        # per-market position cap so they don't block each other; the
        # global RiskGate sees BOTH strategies' realised PnL).
        self._strategy_mode = os.getenv("LIVE_BOT_STRATEGY", "taker").strip().lower()
        if self._strategy_mode not in {"taker", "maker", "both"}:
            self.logger.warning("LIVE_BOT_STRATEGY=%s not recognised, defaulting to 'taker'",
                                self._strategy_mode)
            self._strategy_mode = "taker"
        self._taker_enabled = self._strategy_mode in {"taker", "both"}
        self._maker_enabled = self._strategy_mode in {"maker", "both"}
        # Maker gets its OWN RiskState so the per-market cap / margin pool
        # is independent of the taker. (Realised PnL still flows through
        # the shared RiskGate.)
        self.maker: MakerStrategy | None = None
        if self._maker_enabled:
            self.maker_risk_state = RiskState()
            self.maker = MakerStrategy(
                router=self.router,
                positions=self.positions,
                risk_state=self.maker_risk_state,
                decisions_dir=self.decisions_dir,
                logger=self.logger,
            )

        self._registered_slugs: set[str] = set()
        self._asset_id_by_slug: dict[str, tuple[str | None, str | None]] = {}
        # accumulator: slug -> partial dict {open_s, close_s, asset_ids, outcomes, price_to_beat, ...}
        # registration happens only when all required fields are present.
        self._pending_meta: dict[str, dict] = {}
        self._fee_calc_cache: dict[str, FeeCalculator] = {}
        self._shutdown = asyncio.Event()
        # decisions log is keyed off the SNAPSHOT date, not the boot date,
        # so bot rolling past UTC midnight starts writing decisions_<new-day>.jsonl
        # without restart.  See _decisions_log_for(ts_ns).
        self._decisions_log_cache: dict[str, Path] = {}
        # Stats counters (printed every N seconds)
        self._stats: dict[str, int] = {
            "events_poly_l2": 0, "events_poly_rest": 0,
            "events_binance_spot": 0, "events_binance_usdm": 0, "events_hyperliquid": 0,
            "snapshots_fired": 0, "decisions_logged": 0, "enters_attempted": 0,
        }

    # ── event handlers ───────────────────────────────────────────────────────
    def _accumulate_from_rest(self, record: dict) -> None:
        """REST events bring slug + open_s + close_s + asset_ids + outcomes
        (but NOT price_to_beat — that's the strike)."""
        slug = str(record.get("selected_market_slug") or "").strip()
        if not slug or slug in self._registered_slugs:
            return
        bucket = self._pending_meta.setdefault(slug, {})
        try:
            open_s  = int(record.get("market_open_s") or 0)
            close_s = int(record.get("market_close_s") or 0)
        except (TypeError, ValueError):
            return
        if open_s:  bucket["open_s"]  = open_s
        if close_s: bucket["close_s"] = close_s
        bucket.setdefault("market_id",    str(record.get("selected_market_id") or ""))
        bucket.setdefault("condition_id", str(record.get("selected_condition_id") or ""))
        asset_ids = list(record.get("selected_asset_ids") or [])
        outcomes  = [str(o).lower() for o in (record.get("selected_outcomes") or [])]
        if len(asset_ids) == 2 and len(outcomes) == 2:
            up_id = down_id = None
            for o, a in zip(outcomes, asset_ids):
                if o == "up":   up_id   = str(a)
                if o == "down": down_id = str(a)
            if up_id:   bucket["up_asset_id"]   = up_id
            if down_id: bucket["down_asset_id"] = down_id
        self._try_register(slug)

    def _accumulate_from_strike(self, record: dict) -> None:
        """STRIKE events bring price_to_beat."""
        slug = str(record.get("selected_market_slug") or "").strip()
        if not slug or slug in self._registered_slugs:
            return
        try:
            strike = record.get("price_to_beat")
            if strike is None:
                return
            price = float(strike)
        except (TypeError, ValueError):
            return
        if price > 0:
            bucket = self._pending_meta.setdefault(slug, {})
            bucket["price_to_beat"] = price
            bucket.setdefault("strike_recv_ts_ns", int(record.get("recv_ts_ns") or 0))
            self._try_register(slug)

    def _try_register(self, slug: str) -> None:
        bucket = self._pending_meta.get(slug, {})
        required = ("open_s", "close_s", "price_to_beat", "up_asset_id", "down_asset_id")
        if not all(k in bucket for k in required):
            return
        label = MarketLabel(
            market_slug      = slug,
            market_id        = bucket.get("market_id", ""),
            condition_id     = bucket.get("condition_id", ""),
            strike_recv_ts_ns= bucket.get("strike_recv_ts_ns", 0),
            price_to_beat    = float(bucket["price_to_beat"]),
            market_open_ts_ns= int(bucket["open_s"]) * SECOND_NS,
            market_close_ts_ns=int(bucket["close_s"]) * SECOND_NS,
            market_open_s    = int(bucket["open_s"]),
            market_close_s   = int(bucket["close_s"]),
            up_asset_id      = bucket["up_asset_id"],
            down_asset_id    = bucket["down_asset_id"],
            resolved_side    = "unknown",
            resolution_recv_ts_ns    = 0,
            resolution_resolved_ts_ns= 0,
        )
        self.runtime.register_market(label)
        self._registered_slugs.add(slug)
        self._asset_id_by_slug[slug] = (label.up_asset_id, label.down_asset_id)
        del self._pending_meta[slug]
        self.logger.info("registered market %s (strike=%.2f close %s)",
                         slug, label.price_to_beat, _ts_iso(label.market_close_ts_ns))

        # Fix #4: subscribe the direct-WS to this market's asset_ids so
        # L2 events skip the recorder->disk->tail hop.
        if self.direct_ws is not None:
            try:
                self.direct_ws.add_market(
                    market_slug=slug,
                    market_id=label.market_id,
                    condition_id=label.condition_id,
                    up_asset_id=label.up_asset_id,
                    down_asset_id=label.down_asset_id,
                )
            except Exception as exc:
                self.logger.warning("direct_ws add_market failed for %s: %r", slug, exc)

        # Fix #2: pre-warm the CLOB SDK's per-token caches (tick_size,
        # neg_risk) so the FIRST order on this market doesn't pay
        # ~170 ms of pre-flight GETs. After this, posting a FAK is one
        # HTTP roundtrip instead of four.
        self._prewarm_router_caches(label.down_asset_id)

    def _handle_decision(self, decision: dict) -> None:
        snap = decision["snap"]
        now_ns = int(snap["snapshot_ts_ns"])
        block, reason = self.risk_gate.should_block(now_ns=now_ns)
        record = {
            "logged_at_ns":      _now_ns(),
            "snapshot_ts_ns":    now_ns,
            "snapshot_iso":      _ts_iso(now_ns),
            "market_slug":       snap["market_slug"],
            "market_close_ts_ns": decision.get("market_close_ts_ns"),
            "t_to_close_s":      snap.get("t_to_close_s"),
            "up_token_best_bid": snap.get("up_token_best_bid"),
            "up_token_best_ask": snap.get("up_token_best_ask"),
            "p_up":              decision.get("p_up"),
            "edge_dn":           decision.get("edge_dn"),
            "decision":          decision["decision"],
            "reason":            decision["reason"],
            "notional":          decision["notional"],
            "risk_blocked":      block,
            "risk_reason":       reason,
            "router_mode":       "live" if self.router.live else "dry_run",
        }
        self._stats["snapshots_fired"] += 1
        self._stats["decisions_logged"] += 1
        if decision["decision"] != "ENTER" or block:
            self._append_jsonl(self._decisions_log_for(now_ns), record)
            return
        self._stats["enters_attempted"] += 1

        slug = snap["market_slug"]
        ids = self._asset_id_by_slug.get(slug, (None, None))
        up_asset_id, down_asset_id = ids
        # decide() now returns "side" = "UP" or "DOWN" -- routing here picks
        # the right asset_id and the right submit method based on it.
        side = decision.get("side", "DOWN")            # back-compat: default DOWN
        target_asset_id = up_asset_id if side == "UP" else down_asset_id
        if not target_asset_id:
            record["error"] = f"no_{side.lower()}_asset_id"
            self._append_jsonl(self._decisions_log_for(now_ns), record); return

        # Fire the MAKER path first (if enabled). It posts a passive bid and
        # walks away — it never blocks or slows the taker path. Wrapped in
        # try/except so any maker bug can't kill a real taker order below.
        # Note: the maker is currently DOWN-only -- if side==UP we skip the
        # maker leg entirely.
        if self._maker_enabled and self.maker is not None and side == "DOWN":
            try:
                decision_for_maker = dict(decision)
                decision_for_maker["down_token_id"] = down_asset_id
                self.maker.submit_for_decision(decision_for_maker)
            except Exception as exc:
                self.logger.warning("maker submit_for_decision failed: %r", exc)

        # ── FAK taker path ─────────────────────────────────────────────────
        # If taker is disabled (LIVE_BOT_STRATEGY=maker), log the decision
        # without placing an aggressive order and return.
        if not self._taker_enabled:
            record["taker_disabled"] = True
            record["side"] = side
            self._append_jsonl(self._decisions_log_for(now_ns), record)
            return

        # Pricing — cross the chosen side's ask AGGRESSIVELY. Default buffer
        # raised 0.02 -> 0.10 because the previous 2-cent buffer kept getting
        # rejected with "no orders found to match" -- by the time the order
        # arrived, the book had moved >2c. With +0.10 we essentially become a
        # market order: pay up to 10c above snapshot ask if needed. Slippage
        # is bounded by the buffer; rejection rate drops to ~0.
        LIMIT_BUFFER = float(os.environ.get("LIVE_BOT_LIMIT_BUFFER", "0.10"))
        up_bid = float(snap["up_token_best_bid"])
        up_ask = float(snap["up_token_best_ask"])
        if side == "DOWN":
            # Cross the REAL DOWN book. The implied (1-up_bid) sometimes sits
            # below the real DOWN ask whenever the two books decouple; we use
            # max(real_ask + buffer, implied) so we cross in either case.
            implied_limit = 1.0 - up_bid
            real_down_ask = snap.get("down_token_best_ask")
            try:
                ra = float(real_down_ask) if real_down_ask is not None else None
            except (TypeError, ValueError):
                ra = None
            if ra is not None and 0.0 < ra < 1.0:
                limit_price = min(0.99, max(implied_limit, ra + LIMIT_BUFFER))
            else:
                limit_price = implied_limit
        else:   # UP
            # Mirror the DOWN logic: cross the REAL UP ask with buffer, AND
            # also respect the implied UP ask (1 - down_bid if we had it, but
            # we only have down_ask in the snap right now). As a safety
            # backstop we floor the limit at (1 - down_token_best_ask) -- if
            # the DOWN book is at 0.40/0.50, the implied UP must be > 1-0.50
            # = 0.50 for an arb-free book. Always cross by at least BUFFER
            # so a stale snapshot doesn't strand us.
            real_down_ask = snap.get("down_token_best_ask")
            try:
                da = float(real_down_ask) if real_down_ask is not None else None
            except (TypeError, ValueError):
                da = None
            crossing_a = up_ask + LIMIT_BUFFER
            crossing_b = (1.0 - da) + LIMIT_BUFFER if (da is not None and 0.0 < da < 1.0) else 0.0
            limit_price = min(0.99, max(crossing_a, crossing_b))
        size_shares = decision["notional"] / max(limit_price, 1e-6)
        date_str = datetime.fromtimestamp(now_ns / 1e9, UTC).date().isoformat()
        fc = self._fee_calc_cache.setdefault(date_str, FeeCalculator.for_date(date_str))
        fee_estimate = fc.taker_fee_usd(price=limit_price, size=size_shares)
        if side == "UP":
            receipt = self.router.submit_buy_up_fak(
                up_token_id=target_asset_id,
                limit_price=limit_price,
                size_shares=size_shares,
            )
        else:
            receipt = self.router.submit_buy_down_fak(
                down_token_id=target_asset_id,
                limit_price=limit_price,
                size_shares=size_shares,
            )
        record["side"]             = side
        record["limit_price"]      = limit_price
        record["size_shares"]      = size_shares
        record["fee_estimate_usd"] = fee_estimate
        record["receipt"] = {"success": receipt.success, "dry_run": receipt.dry_run,
                             "filled_size": receipt.filled_size, "avg_price": receipt.avg_price,
                             "status": receipt.status, "order_id": receipt.order_id,
                             "err": receipt.err}
        self._append_jsonl(self._decisions_log_for(now_ns), record)

        if receipt.success and receipt.filled_size > 0:
            # Update risk state for the actual filled amount
            actual_notional = float(receipt.filled_size * receipt.avg_price)
            self.risk_state.add_lot(close_ns=int(decision["market_close_ts_ns"]),
                                    notional=actual_notional)
            self.risk_state.positions_per_market[slug] += 1
            # Per-side counter (1 UP + 1 DOWN per market max by default)
            self.risk_state.positions_per_market_side[(slug, side)] += 1
            self.risk_state.last_entry_ns_per_market[slug] = now_ns
            # OpenPosition's `down_token_id` field name pre-dates UP support
            # -- we store whichever side's token_id we just bought there so
            # the existing reconcile pipeline can resolve PnL at close.
            pos = OpenPosition(
                market_slug=slug,
                down_token_id=target_asset_id,
                shares=float(receipt.filled_size),
                entry_price=float(receipt.avg_price),
                notional_usd=actual_notional,
                fee_usd_estimated=fee_estimate,
                snapshot_ts_ns=now_ns,
                market_close_ts_ns=int(decision["market_close_ts_ns"]),
                order_id=receipt.order_id,
                edge_dn=float(decision.get("edge") or decision.get("edge_dn") or 0.0),
            )
            # Annotate side on the position for clarity in logs / state file.
            setattr(pos, "side", side)
            self.positions.add_position(pos)
            self.logger.info("ENTER %s [%s] @ %.3f x%.2f (notional=$%.2f, mode=%s)",
                             slug, side, limit_price, receipt.filled_size, actual_notional,
                             "LIVE" if self.router.live else "DRY")
        elif receipt.success and receipt.filled_size == 0:
            self.logger.info("FAK partial-fill: 0 shares (quote evaporated) - slug=%s", slug)
            # Back off this market briefly so we don't spam retries
            self.risk_state.last_entry_ns_per_market[slug] = now_ns
        elif not receipt.success:
            self.logger.warning("order error slug=%s: %s", slug, receipt.err)
            # Failed order — back off the same market for the per-market gap (10s)
            # so we don't burn 4-5 retries in one second on a server-side error.
            self.risk_state.last_entry_ns_per_market[slug] = now_ns

    def _prewarm_router_caches(self, down_token_id: str | None) -> None:
        """Pre-fetch tick_size + neg_risk + server version so the SDK's
        internal caches are warm before the first order on this token.
        The CLOB SDK caches all three per process, so this is a one-time
        cost per token paid at registration time (off the hot path),
        turning a ~520 ms 4-roundtrip first order into a ~350 ms 1-rt one.
        Silent no-op in dry-run / when token_id is missing."""
        if not down_token_id or self.router is None or not self.router.live:
            return
        client = getattr(self.router, "_client", None)
        if client is None:
            return
        try:
            client.get_tick_size(down_token_id)
            client.get_neg_risk(down_token_id)
            # __resolve_version is name-mangled (private); call get_version
            # directly to populate the same cache the order path reads.
            if hasattr(client, "get_version"):
                client.get_version()
            self.logger.debug("router cache prewarmed for token %s", down_token_id[:16] + "...")
        except Exception as exc:
            # Non-fatal — the order path will just pay the cost itself later.
            self.logger.warning("router cache prewarm failed for token=%s: %r",
                                (down_token_id or "")[:16], exc)

    def _append_jsonl(self, path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _refresh_canonical_path(self) -> None:
        """Point btc.canonical_parquet_path at today's UTC date file.
        Called at startup and from the tick loop so the bot transparently
        rolls over the canonical at midnight."""
        today_iso = datetime.now(UTC).date().isoformat()
        self.btc.canonical_parquet_path = self._canonical_dir / f"{today_iso}.parquet"

    def _decisions_log_for(self, snap_ts_ns: int) -> Path:
        """Returns the decisions_<date>.jsonl path for a given snapshot.
        Cached so we don't hit datetime conversion overhead per write."""
        date_iso = datetime.fromtimestamp(snap_ts_ns / 1e9, UTC).date().isoformat()
        path = self._decisions_log_cache.get(date_iso)
        if path is None:
            path = self.decisions_dir / f"decisions_{date_iso}.jsonl"
            self._decisions_log_cache[date_iso] = path
        return path

    # ── main loops ───────────────────────────────────────────────────────────
    async def _ingest_loop(self, tail: MultiSourceTail) -> None:
        async for source, _path, rec in tail.stream():
            try:
                if source == MultiSourceTail.SOURCE_POLY_L2:
                    self.runtime.feed_l2(rec); self._stats["events_poly_l2"] += 1
                elif source == MultiSourceTail.SOURCE_POLY_REST:
                    self.runtime.feed_rest(rec); self._stats["events_poly_rest"] += 1
                    self._accumulate_from_rest(rec)
                elif source == MultiSourceTail.SOURCE_POLY_STRIKE:
                    self._accumulate_from_strike(rec); self._stats["events_poly_strike"] = (
                        self._stats.get("events_poly_strike", 0) + 1)
                elif source == MultiSourceTail.SOURCE_BINANCE:
                    self.btc.feed_binance_spot(rec); self._stats["events_binance_spot"] += 1
                elif source == MultiSourceTail.SOURCE_BINUSDM:
                    self.btc.feed_binance_usdm(rec); self._stats["events_binance_usdm"] += 1
                elif source == MultiSourceTail.SOURCE_HYPERLIQ:
                    self.btc.feed_hyperliquid(rec); self._stats["events_hyperliquid"] += 1
                elif source == MultiSourceTail.SOURCE_CHAINLINK:
                    self.btc.feed_chainlink_onchain(rec)
                    self._stats["events_chainlink"] = self._stats.get("events_chainlink", 0) + 1
            except Exception as exc:
                self.logger.warning("ingest error src=%s: %r", source, exc)

    async def _stats_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(15.0)
            # Refresh canonical_parquet_path so it tracks today's UTC date
            # (cheap — just constructs a Path).  Without this, after midnight
            # the bot would keep looking at yesterday's canonical forever.
            try:
                self._refresh_canonical_path()
            except Exception:
                pass
            engine = self.runtime._engine
            n_markets = len(engine.markets) if engine is not None else 0
            mid = self.btc.current_mid()
            cl_price = self.btc.chainlink_onchain.last_mid
            bias = self.btc._last_bias
            dws_stats = self.direct_ws.stats if self.direct_ws is not None else None
            dws_str = (f" | dws_msg={dws_stats['ws_messages_received']} "
                       f"dws_emit={dws_stats['ws_records_emitted']} "
                       f"dws_reconn={dws_stats['ws_reconnects']}") if dws_stats else ""
            mk_stats = self.maker.stats if self.maker is not None else None
            mk_str = (f" | mk_sub={mk_stats['submits_ok']}"
                      f" mk_fill={mk_stats['fills_reconciled']}"
                      f" mk_open={len(self.maker._open_orders)}") if mk_stats else ""
            dws_str = dws_str + mk_str
            self.logger.info(
                "stats: markets=%d (pending=%d) btc_mid=%s chainlink=%s bias=%+.2f | "
                "poly_l2=%d poly_rest=%d strike=%d bin_spot=%d bin_usdm=%d hl=%d cl=%d | "
                "snaps=%d decisions=%d enters=%d%s",
                n_markets, len(self._pending_meta),
                f"{mid:.2f}" if mid else "?",
                f"{cl_price:.2f}" if cl_price else "?",
                bias,
                self._stats["events_poly_l2"], self._stats["events_poly_rest"],
                self._stats.get("events_poly_strike", 0),
                self._stats["events_binance_spot"], self._stats["events_binance_usdm"],
                self._stats["events_hyperliquid"],
                self._stats.get("events_chainlink", 0),
                self._stats["snapshots_fired"], self._stats["decisions_logged"],
                self._stats["enters_attempted"],
                dws_str)

    async def _tick_loop(self) -> None:
        # Fire engine.tick() aligned to a 100ms wall-clock grid so live
        # snapshots land on the same sub-second phase as the dense_close
        # backtest grid (:000/:100/.../:900), keeping live<->backtest paired.
        while not self._shutdown.is_set():
            self.runtime.tick(_now_ns())
            now = _now_ns()
            next_boundary = ((now // 100_000_000) + 1) * 100_000_000
            await asyncio.sleep(max(0.0, (next_boundary - now) / 1e9))

    async def _live_bias_loop(self) -> None:
        # Self-contained in-process bias refresh. The bot computes its own
        # fresh bias from recent recorder data on this loop, so the operator
        # never needs to run compute_bias.ps1 or canonical_daemon.py. Just
        # the two windows: recorders and bot. The compute (~3-20s) runs in a
        # thread executor so it doesn't block the 100ms tick loop.
        interval_s = float(os.getenv("LIVE_BOT_BIAS_REFRESH_S", "90"))
        window_hours = float(os.getenv("LIVE_BOT_BIAS_WINDOW_HOURS", "6.0"))
        max_workers = int(os.getenv("LIVE_BOT_BIAS_MAX_WORKERS", "4"))
        loop = asyncio.get_running_loop()
        while not self._shutdown.is_set():
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self.btc.reload_live_bias(
                        raw_root=self.raw_root,
                        window_hours=window_hours,
                        max_workers=max_workers,
                        logger=self.logger,
                    ),
                )
                self._append_bias_audit(result, window_hours=window_hours,
                                        trigger="refresh")
            except Exception as exc:
                self.logger.warning("live_bias refresh exception: %r", exc)
            await asyncio.sleep(interval_s)

    def _append_bias_audit(self, result: dict | None, *,
                           window_hours: float, trigger: str) -> None:
        """Append one row per bias compute to logs/live_bot/bias_audit_<UTC-date>.jsonl
        so we can later rebuild canonical for the same UTC day and check
        live_bias[ts] == canonical_rolling_bias[ts] bit-for-bit."""
        try:
            now_ns = int(datetime.now(UTC).timestamp() * 1e9)
            date_iso = datetime.fromtimestamp(now_ns / 1e9, UTC).date().isoformat()
            path = self.decisions_dir / f"bias_audit_{date_iso}.jsonl"
            record = {
                "wrote_at_ns": now_ns,
                "trigger": trigger,                    # "refresh" or "initial"
                "window_hours": float(window_hours),
                "applied_last_bias": float(self.btc._last_bias),
                "applied_last_bias_recv_ts_s": int(self.btc._last_bias_recv_ns // 1_000_000_000),
                "ok": result is not None,
            }
            if result is not None:
                # Persist every numeric field from compute_live_bias_value
                # so a canonical rebuild can be compared to it field-by-field.
                record.update({
                    "ts": int(result["ts"]),
                    "rolling_bias": float(result["rolling_bias"]),
                    "synthetic_corrected": float(result["synthetic_corrected"]),
                    "synthetic_raw": float(result["synthetic_raw"]),
                    "mids": {k: float(v) for k, v in (result.get("mids") or {}).items()},
                    "bot_extracted_bias": (None if result.get("bot_extracted_bias") is None
                                           else float(result["bot_extracted_bias"])),
                    "bias_active": bool(result.get("bias_active", False)),
                    "bias_mode": result.get("bias_mode"),
                    "value_age_seconds": float(result.get("value_age_seconds") or 0.0),
                    "elapsed_seconds": float(result.get("elapsed_seconds") or 0.0),
                })
            self._append_jsonl(path, record)
        except Exception as exc:
            self.logger.warning("bias_audit append failed: %r", exc)

    async def _reconcile_loop(self) -> None:
        # Every 60s look for newly-resolved positions
        while not self._shutdown.is_set():
            try:
                closed = self.positions.reconcile_resolved(_now_ns())
                for c in closed:
                    self.risk_gate.record_realized(c.resolved_ts_ns, c.realized_pnl_usd)
                    self.logger.info("RESOLVED %s: %s pnl=$%+.4f day_pnl=$%+.2f",
                                     c.open.market_slug, "WIN" if c.won else "LOSE",
                                     c.realized_pnl_usd, self.risk_gate.today_pnl())
            except Exception as exc:
                self.logger.exception("reconcile error: %r", exc)
            await asyncio.sleep(60.0)

    async def run(self) -> None:
        self.logger.info("=== LIVE BOT STARTING ===")
        self.logger.info("router mode:  %s", "LIVE" if self.router.live else "DRY-RUN")
        self.logger.info("API creds:    %s", "present" if self.creds.has_api_creds() else "MISSING")
        self.logger.info("wallet key:   %s", "present" if self.creds.has_signing_key() else "MISSING")
        self.logger.info("decision log dir: %s (path depends on snapshot date)", self.decisions_dir)
        self.logger.info("ENABLE_REAL_ORDERS=%s", os.environ.get("ENABLE_REAL_ORDERS", "0"))
        self.logger.info("canonical parquet: %s (exists=%s)",
                         self.btc.canonical_parquet_path,
                         self.btc.canonical_parquet_path.exists() if self.btc.canonical_parquet_path else False)
        # Eagerly load whatever bias we have RIGHT NOW (today's parquet if
        # built, else the most recent past day up to 7 days back). This means
        # the bot doesn't sit idle for the first canonical_daemon cycle.
        if not _BIAS_NEEDED:
            self.logger.info("bias compute DISABLED: active model is no-bias "
                             "(set LIVE_BOT_MODEL=legacy to re-enable)")
        elif self.btc.reload_canonical_bias(force=True):
            age_s = (int(datetime.now(UTC).timestamp()*1e9) - self.btc._last_bias_recv_ns) / 1e9
            self.logger.info("startup bias loaded: %+.2f from canonical (age=%s)",
                             self.btc._last_bias,
                             f"{age_s/3600:.1f}h" if age_s > 3600 else f"{age_s:.0f}s")
        else:
            self.logger.warning(
                "no canonical parquet found at startup — bot will idle until "
                "tools/canonical_daemon.py writes one (decisions will skip with "
                "reason=bias_not_loaded)")
        from live_bot.decision_engine import MAX_BIAS_AGE_S
        self.logger.info("bias-freshness gate: max age = %ds (= %.1fh). "
                         "Make sure tools/canonical_daemon.py is also running.",
                         int(MAX_BIAS_AGE_S), MAX_BIAS_AGE_S/3600)
        if not self.creds.has_api_creds():
            self.logger.error("API creds missing — bot cannot subscribe or place orders")
            return

        # Loud preflight: confirms wallet/allowance state before we ever
        # try to place an order. Read-only — safe to run in any mode.
        if self.router.live:
            preflight = self.router.preflight()
            self.logger.info("preflight: %s", preflight)
            bal = preflight.get("usdc_balance", 0)
            if bal < 1.0:
                self.logger.error(
                    "USDC balance is %.4f in funder %s — bot will run but EVERY order "
                    "will reject. Deposit USDC.e (bridged) on Polygon to fix.",
                    bal, self.creds.funder)
            elif any(v < 1.0 for v in (preflight.get("usdc_allowances") or {}).values()):
                self.logger.warning(
                    "USDC allowance is 0 for at least one exchange contract. "
                    "Run `py -3 -c \"from live_bot.order_router import *; "
                    "r = PolymarketOrderRouter(creds=PolymarketCreds.from_keyfile_and_env(), live=True); "
                    "r.set_max_usdc_allowance()\"` once to fix.")

        today_iso = datetime.now(UTC).date().isoformat()
        # Poll the recorder files aggressively (env-overridable). Combined with
        # a low recorder flush interval this collapses the file-hop latency
        # that was making the model decide on ~2.5s-stale BTC (see Plan A).
        tail_poll_s = float(os.getenv("LIVE_BOT_TAIL_POLL_S", "0.05"))
        tail = MultiSourceTail(raw_root=self.raw_root, today_iso=today_iso,
                               poll_interval_s=tail_poll_s, logger=self.logger)

        # Initial bias compute (synchronous, off the event loop). Done BEFORE
        # the tick loop starts so the very first decision uses a fresh
        # in-process bias rather than whatever stale value the canonical
        # parquet had on disk. This is what makes the bot self-contained:
        # only the recorders need to be running, no compute_bias.ps1.
        # SKIPPED when the active model doesn't depend on bias (no-bias v1).
        if _BIAS_NEEDED:
            self.logger.info("computing initial in-process bias from recorder data ...")
            loop = asyncio.get_running_loop()
            bias_window_h = float(os.getenv("LIVE_BOT_BIAS_WINDOW_HOURS", "6.0"))
            bias_workers = int(os.getenv("LIVE_BOT_BIAS_MAX_WORKERS", "4"))
            try:
                initial_result = await loop.run_in_executor(
                    None,
                    lambda: self.btc.reload_live_bias(
                        raw_root=self.raw_root,
                        window_hours=bias_window_h,
                        max_workers=bias_workers,
                        logger=self.logger,
                    ),
                )
                self._append_bias_audit(initial_result, window_hours=bias_window_h,
                                        trigger="initial")
                if not initial_result:
                    self.logger.warning(
                        "initial in-process bias compute returned no usable value; "
                        "bot will keep retrying via _live_bias_loop (every "
                        "LIVE_BOT_BIAS_REFRESH_S seconds, default 90)")
            except Exception as exc:
                self.logger.warning("initial bias compute failed: %r — _live_bias_loop will retry", exc)
        else:
            self.logger.info("(skipping initial bias compute -- model is no-bias)")

        tasks = [
            asyncio.create_task(self._ingest_loop(tail),    name="ingest"),
            asyncio.create_task(self._tick_loop(),          name="tick"),
            asyncio.create_task(self._reconcile_loop(),     name="reconcile"),
            asyncio.create_task(self._stats_loop(),         name="stats"),
        ]
        if _BIAS_NEEDED:
            tasks.append(asyncio.create_task(self._live_bias_loop(), name="live_bias"))
        if self.direct_ws is not None:
            tasks.append(asyncio.create_task(self.direct_ws.run(), name="poly_ws_direct"))
            self.logger.info("direct Polymarket WS task started (LIVE_BOT_DIRECT_WS=1)")
        else:
            self.logger.info("direct Polymarket WS disabled (LIVE_BOT_DIRECT_WS=0) — using file tail only")
        if self.maker is not None:
            tasks.append(asyncio.create_task(self.maker.run_reconcile_loop(),
                                             name="maker_reconcile"))
        self.logger.info("strategy mode: LIVE_BOT_STRATEGY=%s "
                         "(taker=%s, maker=%s)",
                         self._strategy_mode, self._taker_enabled, self._maker_enabled)
        await self._shutdown.wait()
        self.logger.info("shutdown requested — cancelling tasks")
        if self.direct_ws is not None:
            self.direct_ws.stop()
        if self.maker is not None:
            self.maker.stop()
        for t in tasks: t.cancel()
        for t in tasks:
            try: await t
            except asyncio.CancelledError: pass
        tail.close()
        self.logger.info("=== LIVE BOT STOPPED ===")

    def trigger_shutdown(self) -> None:
        self._shutdown.set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default="data", type=Path)
    ap.add_argument("--decisions-dir", default="logs/live_bot", type=Path)
    ap.add_argument("--live", action="store_true",
                    help="Allow live order placement (also requires ENABLE_REAL_ORDERS=1 and a private key).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))
    bot = LiveBot(raw_root=args.raw_root, decisions_dir=args.decisions_dir, live_orders=args.live)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig_name):
            try:
                loop.add_signal_handler(getattr(signal, sig_name), bot.trigger_shutdown)
            except (NotImplementedError, RuntimeError):
                pass
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        bot.trigger_shutdown()
        loop.run_until_complete(asyncio.sleep(0.5))
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
