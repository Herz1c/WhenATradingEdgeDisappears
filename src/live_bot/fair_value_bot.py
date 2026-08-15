"""Fair-value live bot — shadow/live runner for the RTDS-oracle strategy.

Mirrors live_bot.main.LiveBot's proven skeleton (file-tail data, dry-run-gated
PolymarketOrderRouter, PositionManager/RiskGate) but drives the SAME feature
extractor used to build the training dataset (src/fair_value/extractor.py), so
LIVE FEATURES == BACKTEST FEATURES by construction:

  * data comes from the very recorder files that produced fair_value_v1
    (MultiSourceTail with fair_value_sources=True adds RTDS + Coinbase);
  * each active market gets a FairValueState whose Binance(=CEX) + Chainlink
    sub-states are SHARED globals (one Coinbase/RTDS stream feeds them once),
    and a per-market Polymarket book + strike;
  * the decision is the trained model (artifacts/fair_value_v1) + the exact
    gates validated in the backtest (band [0.30,0.70], EV>=0.10, ttc 20-90s,
    one entry per market).

Default is DRY-RUN (shadow): logs every decision + the quote it would have
taken, so we can measure live data freshness and would-be PnL with zero risk.
Real orders require --live AND ENABLE_REAL_ORDERS=1 AND a signing key (the
order_router enforces this).

Run:
  py -3 -m live_bot.fair_value_bot                 # shadow (dry-run)
  $env:ENABLE_REAL_ORDERS="1"; py -3 -m live_bot.fair_value_bot --live
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import signal
import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from fair_value.extractor import BinanceState, ChainlinkState, FairValueState, _f
from poly_l2_only.extractor import MarketState
from live_bot.file_tail import MultiSourceTail
from live_bot.poly_ws_direct import PolyDirectWS
from live_bot.fair_value_feeds import RtdsDirectWS, CoinbaseDirectWS
from live_bot.order_router import PolymarketCreds, PolymarketOrderRouter
from live_bot.position_manager import OpenPosition, PositionManager
from live_bot.risk_gate import RiskGate

NS = 1_000_000_000


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


def _parse_allowed_utc_hours(raw: str) -> set[int] | None:
    raw = raw.strip()
    if not raw or raw.lower() in {"all", "*"}:
        return None
    hours: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        h = int(part)
        if h < 0 or h > 23:
            raise ValueError(f"allowed UTC hours contains invalid hour: {h}")
        hours.add(h)
    return hours


def _allowed_utc_hours_for_mode(live_orders: bool) -> tuple[set[int] | None, str, str]:
    """Mode-specific trading hours.

    Shadow needs volume for parity testing, so it defaults to all day. Live
    keeps the validated 15:00-20:00 Prague summer window unless explicitly
    overridden.
    """
    if live_orders:
        name = "FV_LIVE_ALLOWED_UTC_HOURS"
        default_raw = "13,14,15,16,17"
    else:
        name = "FV_SHADOW_ALLOWED_UTC_HOURS"
        default_raw = "all"
    raw = os.getenv(name, default_raw)
    return _parse_allowed_utc_hours(raw), name if name in os.environ else "default", raw


ART = _env_path("FV_ARTIFACTS_DIR", REPO / "artifacts" / "fair_value_v2_source_time")

# Gates — the exact validated config (band [0.30,0.70], EV>=0.10, ttc 20-90s).
TTC_MIN = float(os.getenv("FV_TTC_MIN", "15"))
TTC_MAX = float(os.getenv("FV_TTC_MAX", "75"))
EV_THR = float(os.getenv("FV_EV_THR", "0.02"))
PRICE_LO = float(os.getenv("FV_PRICE_LO", "0.10"))
PRICE_HI = float(os.getenv("FV_PRICE_HI", "0.90"))
FIXED_SHARES = float(os.getenv("FV_FIXED_SHARES", "5.1"))
MAX_SLIPPAGE = float(os.getenv("FV_MAX_SLIPPAGE", "0.05"))  # max c above quote we'll cross to fill
MAX_EXEC_PM_LAG_S = float(os.getenv("FV_MAX_EXEC_PM_LAG_S", "1.0"))
MAX_EXEC_SOURCE_LAG_S = float(os.getenv("FV_MAX_EXEC_SOURCE_LAG_S", "2.0"))
EXEC_RETRY_COOLDOWN_S = float(os.getenv("FV_EXEC_RETRY_COOLDOWN_S", "2.0"))

# BOOK-QUALITY GATE (anti-"bullshit trade" guard). The live PolyDirectWS book can
# desync from the true book (a separate WS feed; thin 5-min markets have sparse
# updates), and since implied_p_up IS the model's init_score, a broken book makes
# the model over-confident and the bot enters trades that don't exist in the real
# market (the 2026-06-27 UP bleed). DEFENCE: only enter on a COHERENT, FRESH,
# two-sided book. up/down are complementary, so up_mid+down_mid==1 in any healthy
# book (verified: dataset mid_sum is 1.000 at p99 -> this gate removes ZERO
# legitimate backtest trades while catching every desynced live book).
BOOK_COHERENCE_TOL = float(os.getenv("FV_BOOK_COHERENCE_TOL", "0.03"))  # max |1 - (up_mid+dn_mid)|
BOOK_REQUIRE_BOTH_ACTIVE = os.getenv("FV_BOOK_REQUIRE_BOTH_ACTIVE", "1") == "1"  # both sides traded/updated in last 5s
# Seconds before the trading window (ttc=TTC_MAX) to fire a HARD book resync, so
# decisions run on a freshly-dumped, un-drifted book.
RESYNC_LEAD_S = float(os.getenv("FV_RESYNC_LEAD_S", "15"))

# TRAIN/SERVE PARITY: v2 source-time datasets make RTDS available at
# chainlink_ts + rtds_latency_s. Live features must use that same availability
# clock, not local receive time, or RTDS age / derived model features can flip
# marginal actions. FV_RTDS_LATENCY_S defaults to the v2 builder value.
RTDS_FEATURE_LAG_S = float(os.getenv("FV_RTDS_LATENCY_S", os.getenv("FV_RTDS_LAG_S", "2.5")))

# Dataset builder keeps the final price_change / Coinbase top-of-book update in
# each 200 ms bucket. Live must use the same cadence or rolling event-count and
# CEX microstructure features cannot match the backtest.
PM_PRICE_CHANGE_BUCKET_NS = int(float(os.getenv("FV_PM_PRICE_CHANGE_BUCKET_MS", "200")) * 1_000_000)
CEX_TICKER_BUCKET_NS = int(float(os.getenv("FV_CEX_TICKER_BUCKET_MS", "200")) * 1_000_000)
PM_BBA_AUTHORITATIVE = os.getenv("FV_PM_BBA_AUTHORITATIVE", "0") == "1"

LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s :: %(message)s"


def _now_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1e9)


def _ts_iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat(timespec="milliseconds")


def _iso_to_ns(text: object) -> int | None:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * NS)


def _logit(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, p))
    return math.log(p / (1 - p))


class FairValueBot:
    def __init__(self, *, raw_root: Path = Path("data"),
                 decisions_dir: Path = Path("logs/fair_value_bot"),
                 live_orders: bool = False,
                 logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("fv_bot")
        self.raw_root = raw_root if raw_root.is_absolute() else REPO / raw_root
        self.raw_root = self.raw_root.resolve()
        self.decisions_dir = decisions_dir if decisions_dir.is_absolute() else REPO / decisions_dir
        self.decisions_dir = self.decisions_dir.resolve()
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

        # SHARED global CEX + Chainlink state — one Coinbase/RTDS stream feeds
        # these once; every market's FairValueState references them.
        self.bin = BinanceState()
        self.cl = ChainlinkState()

        self.creds = PolymarketCreds.from_keyfile_and_env()
        self.router = PolymarketOrderRouter(creds=self.creds, live=live_orders, logger=self.logger)
        self.positions = PositionManager(raw_root=raw_root)
        self.risk_gate = RiskGate()
        self._allowed_utc_hours, self._allowed_utc_hours_source, self._allowed_utc_hours_raw = (
            _allowed_utc_hours_for_mode(live_orders)
        )

        # Direct-WS hot data plane (book + RTDS + Coinbase) — bypasses the
        # file-tail, which re-decompresses the huge L2 file each poll and lagged
        # everything 10-18s. Registration (REST + strike) stays on a light tail.
        self._parity_strict = os.getenv("FV_PARITY_STRICT", "0") == "1"
        self._pm_book_source = os.getenv("FV_PM_BOOK_SOURCE", "recorder").strip().lower()
        feature_source = os.getenv("FV_FEATURE_SOURCE", "direct").strip().lower()
        if self._parity_strict:
            self._pm_book_source = "recorder"
            feature_source = "recorder"
        if self._pm_book_source not in {"recorder", "direct"}:
            raise ValueError("FV_PM_BOOK_SOURCE must be 'recorder' or 'direct'")
        if feature_source not in {"recorder", "direct"}:
            raise ValueError("FV_FEATURE_SOURCE must be 'recorder' or 'direct'")
        self._feature_source = feature_source
        self._rtds_source = os.getenv("FV_RTDS_SOURCE", feature_source).strip().lower()
        self._cex_source = os.getenv("FV_CEX_SOURCE", feature_source).strip().lower()
        if self._parity_strict:
            self._rtds_source = "recorder"
            self._cex_source = "recorder"
        if self._rtds_source not in {"recorder", "direct"}:
            raise ValueError("FV_RTDS_SOURCE must be 'recorder' or 'direct'")
        if self._cex_source in {"none", "disabled"}:
            self._cex_source = "off"
        if self._cex_source not in {"recorder", "direct", "off"}:
            raise ValueError("FV_CEX_SOURCE must be 'recorder', 'direct', or 'off'")
        self.poly_ws = (PolyDirectWS(on_l2=self._on_l2, logger=self.logger)
                        if self._pm_book_source == "direct" else None)
        self.rtds_ws = RtdsDirectWS(on_price=self._on_rtds, logger=self.logger)
        self.coinbase_ws = CoinbaseDirectWS(on_ticker=self._buffer_cb_ticker,
                                            on_trade=self._on_cb_trade, logger=self.logger)

        self._markets: dict[str, FairValueState] = {}        # slug -> state
        self._asset_ids: dict[str, tuple[str | None, str | None]] = {}
        self._pm_watermark_by_slug: dict[str, int] = {}
        self._pm_source_watermark_by_slug: dict[str, int] = {}
        self._pm_pc_pending_by_slug: dict[str, tuple[int, dict]] = {}
        self._cb_ticker_pending: tuple[int, tuple[int, float, float, float, float]] | None = None
        self._entered: set[str] = set()                      # one entry per market
        self._exec_retry_after_by_slug: dict[str, int] = {}
        self._resync_armed: set[str] = set()                 # markets we've fired a pre-window resync for
        self._expired: set[str] = set()                      # closed/dropped — never re-register (anti-thrash)
        self._pending_meta: dict[str, dict] = {}
        # RTDS staleness-alignment buffer (train/serve parity, see RTDS_FEATURE_LAG_S):
        # holds (recv_ts_ns, cl_ts_ns, price); released into ChainlinkState once
        # the tick has aged RTDS_FEATURE_LAG_S, so the served feed matches training.
        self._rtds_buf: deque[tuple[int, int, int, float]] = deque()
        self._rtds_lag_ns = int(RTDS_FEATURE_LAG_S * NS)
        self._last_rtds_actual_recv_ns = 0
        # Rolling RTDS (chainlink) price history (cl_ts_ns, price) for deriving a
        # market's STRIKE from the oracle at market-open — same logic the dataset
        # builder intends (_rtds_price_at(open_s)). Used when the strike-recorder
        # HTML scrape is unavailable (price_to_beat=None, as on 2026-06-28).
        self._rtds_hist: deque[tuple[int, float]] = deque(maxlen=2400)  # ~40 min @1/s
        # Rolling RTDS (chainlink) price history for deriving a market's STRIKE
        # from the oracle at market-open — used when the strike recorder's HTML
        # scrape is unavailable (price_to_beat=None). This is exactly how the
        # dataset builder derives strike (_rtds_price_at(open_s)), so parity holds.
        self._rtds_hist: deque[tuple[int, float]] = deque(maxlen=1800)  # ~30 min @1/s
        self._shutdown = asyncio.Event()
        self._stats = {"poly_l2": 0, "rtds": 0, "coinbase": 0, "strike": 0,
                       "rest": 0, "ticks": 0, "enters": 0, "skips": 0}
        self._l2_audit_enabled = os.getenv("FV_L2_AUDIT", "0") == "1"
        self._decision_feature_log_enabled = os.getenv("FV_DECISION_FEATURE_LOG", "1") == "1"
        self._l2_audit_path: Path | None = None
        self._l2_audit_fh = None
        self._l2_audit_writes = 0
        self._l2_audit_flush_n = max(1, int(os.getenv("FV_L2_AUDIT_FLUSH_N", "1000")))

        self._model = None
        self._calibrator = None
        self._features: list[str] = []
        self._feature_policy: dict = {}
        self._require_cex = True
        self._load_model()
        if not self._require_cex:
            self._cex_source = "off"

    def _load_model(self) -> None:
        try:
            import lightgbm as lgb
            import joblib
            self._model = lgb.Booster(model_file=str(ART / "model.txt"))
            self._calibrator = joblib.load(ART / "calibrator.pkl")
            self._features = json.loads((ART / "features.json").read_text())
            policy_path = ART / "feature_policy.json"
            self._feature_policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
            require_cex_raw = os.getenv("FV_REQUIRE_CEX", "").strip()
            if require_cex_raw:
                self._require_cex = require_cex_raw not in {"0", "false", "False", "off", "OFF", "no", "NO"}
            else:
                self._require_cex = bool(self._feature_policy.get("require_cex", True))
            self.logger.info("loaded fair_value model (%d features) from %s feature_set=%s require_cex=%s",
                             len(self._features), ART,
                             self._feature_policy.get("feature_set", "legacy_full"),
                             self._require_cex)
        except Exception as exc:
            self.logger.error("could not load model from %s: %r — bot cannot decide", ART, exc)
            raise

    # ── market registration (mirrors LiveBot) ─────────────────────────────────
    def _accumulate_from_rest(self, rec: dict) -> None:
        slug = str(rec.get("selected_market_slug") or "").strip()
        if not slug or slug in self._markets or slug in self._expired:
            return
        b = self._pending_meta.setdefault(slug, {})
        try:
            if rec.get("market_open_s"):  b["open_s"] = int(rec["market_open_s"])
            if rec.get("market_close_s"): b["close_s"] = int(rec["market_close_s"])
        except (TypeError, ValueError):
            pass
        aids = list(rec.get("selected_asset_ids") or [])
        outs = [str(o).lower() for o in (rec.get("selected_outcomes") or [])]
        if len(aids) == 2 and len(outs) == 2:
            for o, a in zip(outs, aids):
                if o == "up":   b["up_asset_id"] = str(a)
                if o == "down": b["down_asset_id"] = str(a)
        self._try_register(slug)

    def _accumulate_from_strike(self, rec: dict) -> None:
        slug = str(rec.get("selected_market_slug") or "").strip()
        if not slug or slug in self._expired:
            return
        strike = _f(rec.get("price_to_beat"), 0.0)
        if strike <= 0:
            return
        # If already registered with a PENDING (RTDS) strike, prefer the
        # authoritative scraped strike when it finally arrives.
        st = self._markets.get(slug)
        if st is not None:
            if st.strike <= 0:
                st.strike = float(strike)
                self.logger.info("strike (scraped) applied to %s: %.2f", slug, strike)
            return
        self._pending_meta.setdefault(slug, {})["strike"] = strike
        self._try_register(slug)

    def _try_register(self, slug: str) -> None:
        b = self._pending_meta.get(slug, {})
        # Strike is NO LONGER required to register — if the scrape gave us one we
        # use it; otherwise we register with strike=0 (pending) and derive it from
        # the RTDS oracle at market-open in the tick loop. This decouples the bot
        # from the fragile HTML strike scrape.
        if not all(k in b for k in ("open_s", "close_s", "up_asset_id", "down_asset_id")):
            return
        # Skip markets with no tradeable window left (close already past, or under
        # TTC_MIN to go) — otherwise the rest tail re-registers closed markets and
        # the tick loop drops them, thrashing the WS connection.
        if (int(b["close_s"]) * NS - _now_ns()) / NS < TTC_MIN:
            self._pending_meta.pop(slug, None)
            self._expired.add(slug)
            return
        strike = float(b.get("strike") or 0.0)
        # New FairValueState that SHARES the global cex + chainlink sub-states.
        # Default PM top-of-book semantics must match the v2 builder/model.
        # FV_PM_BBA_AUTHORITATIVE=1 is only a diagnostic escape hatch.
        st = FairValueState(pm=MarketState(bba_authoritative=PM_BBA_AUTHORITATIVE), bin=self.bin, cl=self.cl,
                            strike=strike, open_s=int(b["open_s"]),
                            close_s=int(b["close_s"]),
                            require_cex=self._require_cex)
        # pm_state_to_features() owns ttc_log/tick/meta features; keep its
        # embedded MarketState in sync with the FairValueState wrapper.
        st.pm.market_open_s = int(b["open_s"])
        st.pm.market_close_s = int(b["close_s"])
        self._markets[slug] = st
        self._asset_ids[slug] = (b["up_asset_id"], b["down_asset_id"])
        del self._pending_meta[slug]
        self.logger.info("registered %s strike=%s close=%s", slug,
                         f"{strike:.2f}" if strike > 0 else "PENDING(rtds@open)",
                         _ts_iso(int(b["close_s"]) * NS))
        if self.poly_ws is not None:
            try:
                self.poly_ws.add_market(market_slug=slug, market_id="", condition_id="",
                                        up_asset_id=b["up_asset_id"], down_asset_id=b["down_asset_id"])
            except Exception as exc:
                self.logger.warning("poly_ws add_market failed %s: %r", slug, exc)

    # ── direct-WS hot-data handlers ──────────────────────────────────────────
    def _on_l2(self, rec: dict) -> None:
        """L2 record from the selected PM book source."""
        self._stats["poly_l2_raw"] = self._stats.get("poly_l2_raw", 0) + 1
        source_ts_ns = self._pm_source_ts_ns(rec)
        if source_ts_ns <= 0:
            self._stats["poly_l2_missing_source_ts"] = self._stats.get("poly_l2_missing_source_ts", 0) + 1
            return
        if PM_PRICE_CHANGE_BUCKET_NS > 0 and rec.get("event_type") == "price_change":
            self._buffer_pm_price_change(rec)
            return
        self._flush_pm_price_change_until(source_ts_ns or int(rec.get("recv_ts_ns") or _now_ns()))
        self._apply_l2(rec)

    @staticmethod
    def _pm_source_ts_ns(rec: dict) -> int:
        """Exchange/event timestamp for a PM L2 record.

        Polymarket can deliver a large websocket backlog after subscription.
        `recv_ts_ns` then looks fresh even though the payload's own timestamp is
        tens of seconds old; feature/execution freshness must use the payload
        clock and never silently fall back to recv.
        """
        st = rec.get("source_timestamps") or {}
        for key in ("timestamp_ns", "message_timestamp_ns"):
            try:
                v = int(st.get(key) or 0)
            except (TypeError, ValueError):
                v = 0
            if v > 0:
                return v
        for key in ("timestamp_ms", "message_timestamp_ms"):
            try:
                v = int(st.get(key) or 0)
            except (TypeError, ValueError):
                v = 0
            if v > 0:
                return v * 1_000_000
        return 0

    def _apply_l2(self, rec: dict) -> None:
        slug = str(rec.get("selected_market_slug") or "")
        recv_ts_ns = int(rec.get("recv_ts_ns") or 0)
        source_ts_ns = self._pm_source_ts_ns(rec)
        if source_ts_ns <= 0:
            self._stats["poly_l2_missing_source_ts"] = self._stats.get("poly_l2_missing_source_ts", 0) + 1
            return
        if slug and source_ts_ns and source_ts_ns < self._pm_source_watermark_by_slug.get(slug, 0):
            self._stats["poly_l2_out_of_order"] = self._stats.get("poly_l2_out_of_order", 0) + 1
            return
        if slug and recv_ts_ns > self._pm_watermark_by_slug.get(slug, 0):
            self._pm_watermark_by_slug[slug] = recv_ts_ns
        if slug and source_ts_ns > self._pm_source_watermark_by_slug.get(slug, 0):
            self._pm_source_watermark_by_slug[slug] = source_ts_ns
        st = self._markets.get(slug)
        if st is not None:
            state_rec = rec
            if source_ts_ns > 0:
                state_rec = dict(rec)
                state_rec["raw_recv_ts_ns"] = recv_ts_ns
                state_rec["recv_ts_ns"] = source_ts_ns
                state_rec["source_ts_ns"] = source_ts_ns
                if recv_ts_ns > 0:
                    state_rec["pm_delivery_lag_s"] = (recv_ts_ns - source_ts_ns) / NS
            st.update_pm(state_rec)
            if self._l2_audit_enabled:
                self._append_l2_audit(st, state_rec)
        self._stats["poly_l2"] += 1

    def _buffer_pm_price_change(self, rec: dict) -> None:
        slug = str(rec.get("selected_market_slug") or "")
        if not slug:
            return
        ts_ns = self._pm_source_ts_ns(rec)
        if ts_ns <= 0:
            return
        bucket = ts_ns // PM_PRICE_CHANGE_BUCKET_NS
        pending = self._pm_pc_pending_by_slug.get(slug)
        if pending is not None and pending[0] < bucket:
            self._apply_l2(pending[1])
            self._stats["poly_l2_pc_kept"] = self._stats.get("poly_l2_pc_kept", 0) + 1
        elif pending is not None and pending[0] > bucket:
            return
        self._pm_pc_pending_by_slug[slug] = (bucket, rec)

    def _flush_pm_price_change_until(self, cutoff_ns: int) -> None:
        if PM_PRICE_CHANGE_BUCKET_NS <= 0:
            return
        cutoff_bucket = int(cutoff_ns) // PM_PRICE_CHANGE_BUCKET_NS
        for slug, pending in list(self._pm_pc_pending_by_slug.items()):
            bucket, rec = pending
            if bucket < cutoff_bucket:
                self._pm_pc_pending_by_slug.pop(slug, None)
                self._apply_l2(rec)
                self._stats["poly_l2_pc_kept"] = self._stats.get("poly_l2_pc_kept", 0) + 1

    def _pm_clock_fields(self, slug: str, now_ns: int) -> dict:
        recv_watermark = int(self._pm_watermark_by_slug.get(slug, 0) or 0)
        source_watermark = int(self._pm_source_watermark_by_slug.get(slug, 0) or 0)
        recv_lag_s = round(max(0.0, (now_ns - recv_watermark) / NS), 3) if recv_watermark else None
        source_lag_s = round(max(0.0, (now_ns - source_watermark) / NS), 3) if source_watermark else None
        return {
            "pm_book_source": self._pm_book_source,
            "pm_watermark_ns": recv_watermark,
            "pm_lag_s": source_lag_s if source_lag_s is not None else recv_lag_s,
            "pm_recv_lag_s": recv_lag_s,
            "pm_source_ns": source_watermark,
            "pm_source_lag_s": source_lag_s,
        }

    def _append_l2_audit(self, st: FairValueState, rec: dict) -> None:
        """Compact direct-L2 event log for diagnosing live-vs-recorder book drift.

        Off by default. With FV_L2_AUDIT=1 this records only markets near the
        decision/audit window, not the full day-long firehose.
        """
        now = int(rec.get("recv_ts_ns") or _now_ns())
        ttc = (st.close_s * NS - now) / NS
        if not (TTC_MIN - 5 <= ttc <= TTC_MAX + RESYNC_LEAD_S + 10):
            return
        date_iso = datetime.fromtimestamp(now / 1e9, UTC).date().isoformat()
        path = self.decisions_dir / f"l2_audit_{date_iso}.jsonl"
        if path != self._l2_audit_path:
            if self._l2_audit_fh:
                self._l2_audit_fh.close()
            self._l2_audit_fh = path.open("a", encoding="utf-8")
            self._l2_audit_path = path
        changed = rec.get("changed_levels") or []
        payload = rec.get("event_payload") or {}
        digest_src = json.dumps(changed or payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.blake2b(digest_src.encode("utf-8"), digest_size=8).hexdigest()
        lc = rec.get("level_counts") or {}
        dt = rec.get("depth_totals") or {}
        out = {
            "ts": now,
            "slug": rec.get("selected_market_slug"),
            "pm_book_source": self._pm_book_source,
            "ttc": round(ttc, 3),
            "conn": rec.get("connection_id"),
            "seq": rec.get("local_msg_seq"),
            "event_type": rec.get("event_type"),
            "token_outcome": rec.get("token_outcome"),
            "token_asset_id": rec.get("token_asset_id"),
            "book_state_seq": rec.get("book_state_seq"),
            "snapshot_origin": rec.get("snapshot_origin"),
            "book_hash": rec.get("book_hash"),
            "event_digest": digest,
            "best_bid": rec.get("best_bid"),
            "best_ask": rec.get("best_ask"),
            "level_bid": lc.get("bid"),
            "level_ask": lc.get("ask"),
            "depth_bid": dt.get("bid_size_total"),
            "depth_ask": dt.get("ask_size_total"),
            "source_timestamps": rec.get("source_timestamps"),
        }
        self._l2_audit_fh.write(json.dumps(out, separators=(",", ":")) + "\n")
        self._l2_audit_writes += 1
        if self._l2_audit_writes % self._l2_audit_flush_n == 0:
            self._l2_audit_fh.flush()

    def _on_rtds(self, recv_ts_ns: int, cl_ts_ns: int, price: float) -> None:
        self._stats["rtds"] += 1
        if price > 0:
            self._rtds_hist.append((cl_ts_ns, price))   # for RTDS-derived strike
        if recv_ts_ns and self._last_rtds_actual_recv_ns and recv_ts_ns < self._last_rtds_actual_recv_ns:
            self._stats["rtds_out_of_order"] = self._stats.get("rtds_out_of_order", 0) + 1
            return
        if recv_ts_ns > self._last_rtds_actual_recv_ns:
            self._last_rtds_actual_recv_ns = recv_ts_ns
        if self._rtds_lag_ns <= 0:
            self.cl.update(recv_ts_ns, cl_ts_ns, price, self.bin)  # legacy: no alignment
            return
        avail_ns = cl_ts_ns + self._rtds_lag_ns
        self._rtds_buf.append((avail_ns, recv_ts_ns, cl_ts_ns, price))
        self._flush_rtds(_now_ns())

    def _rtds_price_at(self, target_ns: int) -> float | None:
        """RTDS price at a target Chainlink timestamp, matching the builder.

        Wait until the RTDS history has advanced to/past target_ns, then return
        the latest Chainlink tick at or before target_ns. This avoids locking a
        market's strike to the previous second just because the true open tick is
        delivered ~2s after market open.
        """
        if not self._rtds_hist:
            return None
        if max(ts for ts, _ in self._rtds_hist) < target_ns:
            return None
        best_ts = None
        best_px = None
        for ts, px in self._rtds_hist:
            if ts <= target_ns and (best_ts is None or ts > best_ts):
                best_ts = ts
                best_px = px
        if best_ts is not None and (target_ns - best_ts) <= 30 * NS:
            return best_px
        return None

    def _flush_rtds(self, now_ns: int) -> None:
        """Release RTDS ticks at the same source-time availability used by v2."""
        buf = self._rtds_buf
        while buf and buf[0][0] <= now_ns:
            avail_ns, _recv_ts_ns, cl_ts_ns, price = buf.popleft()
            self.cl.update(avail_ns, cl_ts_ns, price, self.bin)

    def _on_cb_ticker(self, ts: int, bid: float, ask: float, bq: float, aq: float) -> None:
        if ts and self.bin.last_ts_ns and ts < self.bin.last_ts_ns:
            self._stats["coinbase_out_of_order"] = self._stats.get("coinbase_out_of_order", 0) + 1
            return
        self.bin.update_bookticker(ts, bid, ask, bq, aq)
        self._stats["coinbase"] += 1

    def _on_cb_trade(self, ts: int, qty: float, side: float) -> None:
        self.bin.update_trade(ts, qty, side)

    def _buffer_cb_ticker(self, ts: int, bid: float, ask: float, bq: float, aq: float) -> None:
        if CEX_TICKER_BUCKET_NS <= 0:
            self._on_cb_ticker(ts, bid, ask, bq, aq)
            return
        bucket = int(ts) // CEX_TICKER_BUCKET_NS
        pending = self._cb_ticker_pending
        if pending is not None and pending[0] < bucket:
            _, data = pending
            self._on_cb_ticker(*data)
        elif pending is not None and pending[0] > bucket:
            return
        self._cb_ticker_pending = (bucket, (int(ts), float(bid), float(ask), float(bq), float(aq)))

    def _flush_cb_ticker_until(self, cutoff_ns: int) -> None:
        if CEX_TICKER_BUCKET_NS <= 0 or self._cb_ticker_pending is None:
            return
        cutoff_bucket = int(cutoff_ns) // CEX_TICKER_BUCKET_NS
        bucket, data = self._cb_ticker_pending
        if bucket < cutoff_bucket:
            self._cb_ticker_pending = None
            self._on_cb_ticker(*data)

    def _on_rtds_record(self, rec: dict) -> None:
        try:
            self._on_rtds(int(rec["recv_ts_ns"]), int(rec["chainlink_ts_ms"]) * 1_000_000,
                          float(rec["btc_usd_price"]))
        except (KeyError, TypeError, ValueError):
            return

    @staticmethod
    def _coinbase_source_ts_ns(rec: dict) -> int | None:
        st = rec.get("source_timestamps") or {}
        try:
            v = int(st.get("message_timestamp_ns") or 0)
        except (TypeError, ValueError):
            v = 0
        if v > 0:
            return v
        payload = rec.get("payload") or {}
        if isinstance(payload, dict):
            return _iso_to_ns(payload.get("timestamp"))
        return None

    def _on_coinbase_record(self, rec: dict) -> None:
        ts = self._coinbase_source_ts_ns(rec)
        if not ts or ts <= 0:
            return
        channel = rec.get("channel")
        payload = rec.get("payload") or {}
        if channel == "ticker":
            for ev in payload.get("events", ()):
                for tk in ev.get("tickers", ()):
                    bb = tk.get("best_bid")
                    ba = tk.get("best_ask")
                    if bb and ba:
                        try:
                            self._buffer_cb_ticker(
                                ts, float(bb), float(ba),
                                float(tk.get("best_bid_quantity") or 0.0),
                                float(tk.get("best_ask_quantity") or 0.0))
                        except (TypeError, ValueError):
                            continue
        elif channel == "market_trades":
            self._flush_cb_ticker_until(ts)
            for ev in payload.get("events", ()):
                if ev.get("type") == "snapshot":
                    continue
                for tr in ev.get("trades", ()):
                    try:
                        size = float(tr.get("size") or 0.0)
                    except (TypeError, ValueError):
                        continue
                    side_raw = tr.get("side")
                    side = 1.0 if side_raw == "BUY" else (-1.0 if side_raw == "SELL" else 0.0)
                    if size > 0:
                        self._on_cb_trade(ts, size, side)

    async def _ingest_loop(self, tail: MultiSourceTail) -> None:
        """LIGHT tail — only market discovery (REST) + strike. Hot feeds
        (book, RTDS, Coinbase) arrive via direct WS, not files."""
        processed = 0
        async for source, _path, rec in tail.stream():
            try:
                if source == MultiSourceTail.SOURCE_POLY_L2:
                    if self._pm_book_source == "recorder":
                        self._accumulate_from_rest(rec)
                        self._on_l2(rec)
                elif source == MultiSourceTail.SOURCE_POLY_REST:
                    self._accumulate_from_rest(rec); self._stats["rest"] += 1
                elif source == MultiSourceTail.SOURCE_POLY_STRIKE:
                    self._accumulate_from_strike(rec); self._stats["strike"] += 1
                elif source == MultiSourceTail.SOURCE_RTDS:
                    if self._rtds_source == "recorder":
                        self._on_rtds_record(rec)
                elif source == MultiSourceTail.SOURCE_COINBASE:
                    if self._cex_source == "recorder":
                        self._on_coinbase_record(rec)
            except Exception as exc:
                self.logger.debug("ingest err src=%s: %r", source, exc)
            processed += 1
            if processed % 500 == 0:
                await asyncio.sleep(0)

    # ── decision (mirrors trainer/backtest exactly) ────────────────────────────
    def _predict(self, feats: dict) -> float:
        x = np.array([[float(feats.get(c, np.nan)) for c in self._features]], dtype=float)
        raw = float(self._model.predict(x, raw_score=True)[0])
        p = 1.0 / (1.0 + math.exp(-(_logit(float(feats["implied_p_up"])) + raw)))
        return float(self._calibrator.transform([p])[0])

    def _decision_meta(self) -> dict:
        return {
            "artifacts_dir": str(ART),
            "parity_strict": self._parity_strict,
            "feature_source": self._feature_source,
            "rtds_source": self._rtds_source,
            "cex_source": self._cex_source,
            "require_cex": self._require_cex,
            "feature_set": self._feature_policy.get("feature_set", "legacy_full"),
            "decision_feature_log": self._decision_feature_log_enabled,
            "allowed_utc_hours": sorted(self._allowed_utc_hours) if self._allowed_utc_hours is not None else None,
            "allowed_utc_hours_source": self._allowed_utc_hours_source,
            "allowed_utc_hours_raw": self._allowed_utc_hours_raw,
        }

    def _feature_snapshot(self, feats: dict) -> dict:
        if not self._decision_feature_log_enabled:
            return {}
        out = {}
        for k in self._features:
            if k not in feats:
                continue
            v = feats.get(k)
            if isinstance(v, (int, float, np.integer, np.floating)):
                fv = float(v)
                out[k] = fv if math.isfinite(fv) else None
            else:
                out[k] = v
        return out

    async def _evaluate(self, slug: str, st: FairValueState, now_ns: int) -> None:
        if slug in self._entered:
            return
        if self._allowed_utc_hours is not None:
            hour_utc = datetime.fromtimestamp(now_ns / NS, tz=UTC).hour
            if hour_utc not in self._allowed_utc_hours:
                return
        ttc = (st.close_s * NS - now_ns) / NS
        if not (TTC_MIN <= ttc <= TTC_MAX):
            return
        feats = st.features(now_ns)
        if feats is None:
            return
        p = self._predict(feats)
        up_ask = float(feats.get("up_best_ask") or 0.0)
        dn_ask = float(feats.get("down_best_ask") or 0.0)
        ev_up = p - up_ask
        ev_dn = (1 - p) - dn_ask
        side = fill = None
        if ev_up >= EV_THR and ev_up >= ev_dn and PRICE_LO < up_ask < PRICE_HI:
            side, fill = "UP", up_ask
        elif ev_dn >= EV_THR and PRICE_LO < dn_ask < PRICE_HI:
            side, fill = "DOWN", dn_ask
        if side is None:
            self._stats["skips"] += 1
            self._stats["no_edge"] = self._stats.get("no_edge", 0) + 1
            return
        # ── BOOK-QUALITY GATE: refuse to trade on a broken/stale book. This is
        # what stops the "bullshit trades": the EV edge above is only real if the
        # book we computed it from is real. ───────────────────────────────────
        bad = self._book_quality_reason(feats)
        if bad is not None:
            self._stats["skips"] += 1
            self._stats["book_blocked"] = self._stats.get("book_blocked", 0) + 1
            self._log_book_block(slug, now_ns, feats, side, fill, p, ttc, bad)
            return
        await self._enter(slug, st, now_ns, feats, side, fill, p, ttc)

    @staticmethod
    def _book_quality_reason(feats: dict) -> str | None:
        """Return a reason string if the book is NOT trustworthy, else None."""
        up_mid = float(feats.get("up_mid") or 0.0)
        dn_mid = float(feats.get("down_mid") or 0.0)
        # both sides must have a real two-sided mid (no one-sided/empty fallback)
        if up_mid <= 0.0 or dn_mid <= 0.0:
            return "one_sided_book"
        # complementary outcomes -> a coherent book sums to ~1; a desynced/stale
        # side breaks this. THE key anti-bullshit check.
        incoherence = abs(1.0 - (up_mid + dn_mid))
        if incoherence > BOOK_COHERENCE_TOL:
            return f"incoherent_mid_sum:{up_mid + dn_mid:.3f}"
        if BOOK_REQUIRE_BOTH_ACTIVE:
            if float(feats.get("up_book_evts_5s") or 0.0) <= 0.0 or \
               float(feats.get("down_book_evts_5s") or 0.0) <= 0.0:
                return "stale_book_side_inactive_5s"
        return None

    def _log_book_block(self, slug, now_ns, feats, side, fill, p, ttc, reason) -> None:
        rec = {
            "logged_at_ns": _now_ns(), "snapshot_ts_ns": now_ns, "iso": _ts_iso(now_ns),
            "market_slug": slug, "decision": "SKIP_BOOK", "book_reason": reason,
            "side": side, "ttc_s": round(ttc, 1), "p_model": round(p, 4),
            "fill_quote": round(fill, 4),
            "up_mid": round(float(feats.get("up_mid") or 0), 4),
            "down_mid": round(float(feats.get("down_mid") or 0), 4),
            "mid_sum": round(float(feats.get("up_mid") or 0) + float(feats.get("down_mid") or 0), 4),
            "up_best_ask": round(float(feats.get("up_best_ask") or 0), 4),
            "down_best_ask": round(float(feats.get("down_best_ask") or 0), 4),
            "up_book_evts_5s": float(feats.get("up_book_evts_5s") or 0),
            "down_book_evts_5s": float(feats.get("down_book_evts_5s") or 0),
            "implied_p_up": round(float(feats.get("implied_p_up") or 0), 4),
        }
        rec.update(self._pm_clock_fields(slug, now_ns))
        rec.update(self._decision_meta())
        rec["f"] = self._feature_snapshot(feats)
        self._write(now_ns, rec)
        self.logger.info("SKIP_BOOK %s [%s] reason=%s mid_sum=%.3f p=%.3f fill=%.3f",
                         slug, side, reason, rec["mid_sum"], p, fill)

    async def _enter(self, slug, st, now_ns, feats, side, fill, p, ttc) -> None:
        retry_after = int(self._exec_retry_after_by_slug.get(slug, 0) or 0)
        if retry_after > now_ns:
            return
        block, reason = self.risk_gate.should_block(now_ns=now_ns)
        up_id, dn_id = self._asset_ids.get(slug, (None, None))
        token_id = up_id if side == "UP" else dn_id
        rec = {
            "logged_at_ns": _now_ns(), "snapshot_ts_ns": now_ns, "iso": _ts_iso(now_ns),
            "market_slug": slug, "side": side, "ttc_s": round(ttc, 1),
            "p_model": round(p, 4), "fill_quote": round(fill, 4),
            "ev": round((p - fill) if side == "UP" else ((1 - p) - fill), 4),
            "rtds_age_s": round(float(feats.get("rtds_age_s") or 0), 2),
            "delta_to_strike_rtds": round(float(feats.get("delta_to_strike_rtds") or 0), 2),
            "implied_p_up": round(float(feats.get("implied_p_up") or 0), 4),
            "mid_sum": round(float(feats.get("up_mid") or 0) + float(feats.get("down_mid") or 0), 4),
            "up_mid": round(float(feats.get("up_mid") or 0), 4),
            "down_mid": round(float(feats.get("down_mid") or 0), 4),
            "up_book_evts_5s": float(feats.get("up_book_evts_5s") or 0),
            "down_book_evts_5s": float(feats.get("down_book_evts_5s") or 0),
            "book_vs_oracle_gap": round(float(feats.get("book_vs_oracle_gap") or 0), 4),
            "market_close_ts_ns": st.close_s * NS,
            "risk_blocked": block, "risk_reason": reason,
            "router_mode": "live" if self.router.live else "dry_run",
        }
        rec.update(self._pm_clock_fields(slug, now_ns))
        rec.update(self._decision_meta())
        timing_sources = {
            "pm_book_ns": int(self._pm_watermark_by_slug.get(slug, 0) or 0),
            "rtds_recv_ns": int(self.cl.latest_recv_ns or 0),
            "rtds_source_ns": int(self.cl.latest_cl_ts_ns or 0),
            "cex_ns": int(self.bin.last_ts_ns or 0),
        }
        rec["timing"] = timing_sources | {
            "eval_snapshot_ns": int(now_ns),
        }
        if block:
            rec["decision"] = "SKIP_RISK"
            rec["f"] = self._feature_snapshot(feats)
            self._write(now_ns, rec); return
        pm_lag_s = rec.get("pm_lag_s")
        if pm_lag_s is None or float(pm_lag_s) > MAX_EXEC_PM_LAG_S:
            rec["decision"] = "SKIP_EXEC_STALE_BOOK"
            rec["exec_reason"] = f"pm_lag_s>{MAX_EXEC_PM_LAG_S:.3f}"
            rec["receipt"] = {"success": False, "dry_run": not self.router.live,
                              "filled_size": 0.0, "avg_price": 0.0,
                              "status": "not_submitted", "order_id": None,
                              "err": rec["exec_reason"]}
            rec["f"] = self._feature_snapshot(feats)
            self._stats["skips"] += 1
            self._stats["exec_blocked"] = self._stats.get("exec_blocked", 0) + 1
            self._exec_retry_after_by_slug[slug] = now_ns + int(EXEC_RETRY_COOLDOWN_S * NS)
            self._write(now_ns, rec)
            self.logger.warning("SKIP_EXEC_STALE_BOOK %s [%s] p=%.3f fill=%.3f ttc=%.0fs pm_lag=%s",
                                slug, side, p, fill, ttc, pm_lag_s)
            return
        stale_sources = []
        freshness_names = ("rtds_recv_ns", "cex_ns") if self._require_cex else ("rtds_recv_ns",)
        for name in freshness_names:
            ts_ns = int(timing_sources.get(name) or 0)
            if ts_ns <= 0:
                stale_sources.append(f"{name}=missing")
                continue
            age_s = (now_ns - ts_ns) / NS
            rec["timing"][name.replace("_ns", "_age_at_eval_s")] = round(age_s, 6)
            if age_s > MAX_EXEC_SOURCE_LAG_S:
                stale_sources.append(f"{name.replace('_ns', '')}_age_s={age_s:.3f}")
        if stale_sources:
            rec["decision"] = "SKIP_EXEC_STALE_SOURCE"
            rec["exec_reason"] = " ".join(stale_sources)
            rec["receipt"] = {"success": False, "dry_run": not self.router.live,
                              "filled_size": 0.0, "avg_price": 0.0,
                              "status": "not_submitted", "order_id": None,
                              "err": rec["exec_reason"]}
            rec["f"] = self._feature_snapshot(feats)
            self._stats["skips"] += 1
            self._stats["exec_blocked"] = self._stats.get("exec_blocked", 0) + 1
            self._exec_retry_after_by_slug[slug] = now_ns + int(EXEC_RETRY_COOLDOWN_S * NS)
            self._write(now_ns, rec)
            self.logger.warning("SKIP_EXEC_STALE_SOURCE %s [%s] p=%.3f fill=%.3f ttc=%.0fs %s",
                                slug, side, p, fill, ttc, rec["exec_reason"])
            return

        size_shares = FIXED_SHARES
        # Crossing LIMIT (taker) with FAK/IOC semantics: the router sends
        # quote+MAX_SLIPPAGE as the hard cap, fills immediately if liquidity is
        # there, and cancels any remainder instead of resting a ghost order.
        # CRITICAL: run the order in a THREAD (asyncio.to_thread), NOT inline —
        # submit_buy_gtc does blocking synchronous HTTP (4 round-trips). On the
        # event loop it FREEZES the PolyDirectWS consumer for hundreds of ms ->
        # book messages overflow the queue and drop -> the book reconstruction
        # drifts (this is how the live book reached corr~0 vs the recorder, while
        # dry-run shadow — no real HTTP — stayed coherent). to_thread keeps the
        # loop free so the WS keeps reading. (Matches poly_l2_only_bot.)
        order_submit_start_ns = _now_ns()
        rec["timing"].update({
            "order_submit_start_ns": int(order_submit_start_ns),
            "snapshot_to_order_submit_s": round((order_submit_start_ns - now_ns) / NS, 6),
        })
        for label, ts_ns in timing_sources.items():
            if ts_ns > 0:
                rec["timing"][f"{label[:-3]}_to_order_submit_s"] = round(
                    (order_submit_start_ns - ts_ns) / NS, 6)
        required_fresh = [v for k, v in timing_sources.items()
                          if k in ({"pm_book_ns", "rtds_recv_ns", "cex_ns"}
                                   if self._require_cex else {"pm_book_ns", "rtds_recv_ns"})
                          and v > 0]
        if required_fresh:
            rec["timing"]["freshest_required_data_to_order_submit_s"] = round(
                (order_submit_start_ns - max(required_fresh)) / NS, 6)
            rec["timing"]["oldest_required_data_to_order_submit_s"] = round(
                (order_submit_start_ns - min(required_fresh)) / NS, 6)
            # Conventional tick-to-trade: latest PM book tick used -> order submit.
            rec["timing"]["tick_to_trade_s"] = rec["timing"].get("pm_book_to_order_submit_s")
        receipt = await asyncio.to_thread(
            self.router.submit_buy_gtc, token_id=token_id, limit_price=fill,
            size_shares=size_shares, side_label=side,
            buffer=MAX_SLIPPAGE, fixed_shares=size_shares)
        order_submit_done_ns = _now_ns()
        rec["timing"].update({
            "order_submit_done_ns": int(order_submit_done_ns),
            "order_submit_roundtrip_s": round((order_submit_done_ns - order_submit_start_ns) / NS, 6),
            "snapshot_to_receipt_s": round((order_submit_done_ns - now_ns) / NS, 6),
        })
        rec["decision"] = "ENTER"
        rec["size_shares"] = round(size_shares, 4)
        rec["receipt"] = {"success": receipt.success, "dry_run": receipt.dry_run,
                          "filled_size": receipt.filled_size, "avg_price": receipt.avg_price,
                          "status": receipt.status, "order_id": receipt.order_id, "err": receipt.err,
                          "raw_response": receipt.raw_response}
        rec["f"] = self._feature_snapshot(feats)
        if receipt.success and receipt.filled_size > 0:
            self._entered.add(slug)
            self._exec_retry_after_by_slug.pop(slug, None)
            rec["decision"] = "ENTER"
            self._write(now_ns, rec)
            self._stats["enters"] += 1
            self.logger.info("ENTER %s [%s] p=%.3f fill=%.3f ttc=%.0fs rtds_age=%.1fs mode=%s",
                             slug, side, p, fill, ttc, rec["rtds_age_s"],
                             "LIVE" if self.router.live else "DRY")
            pos = OpenPosition(
                market_slug=slug, down_token_id=token_id, shares=float(receipt.filled_size),
                entry_price=float(receipt.avg_price), notional_usd=float(receipt.filled_size * receipt.avg_price),
                fee_usd_estimated=0.072 * fill * (1 - fill) * receipt.filled_size,
                snapshot_ts_ns=now_ns, market_close_ts_ns=st.close_s * NS,
                order_id=receipt.order_id, edge_dn=rec["ev"],
                side=side, strike=float(st.strike))   # side+strike now real fields (persist + RTDS reconcile)
            self.positions.add_position(pos)
            return
        rec["decision"] = "SKIP_EXEC"
        self._stats["skips"] += 1
        self._stats["exec_blocked"] = self._stats.get("exec_blocked", 0) + 1
        self._exec_retry_after_by_slug[slug] = now_ns + int(EXEC_RETRY_COOLDOWN_S * NS)
        self._write(now_ns, rec)
        self.logger.warning("SKIP_EXEC %s [%s] p=%.3f fill=%.3f ttc=%.0fs status=%s err=%s order_id=%s",
                            slug, side, p, fill, ttc, receipt.status, receipt.err, receipt.order_id)

    def _write(self, now_ns: int, rec: dict) -> None:
        date_iso = datetime.fromtimestamp(now_ns / 1e9, UTC).date().isoformat()
        path = self.decisions_dir / f"fv_decisions_{date_iso}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # ── loops ────────────────────────────────────────────────────────────────
    async def _tick_loop(self) -> None:
        tick_s = float(os.getenv("FV_TICK_S", "1.0"))
        while not self._shutdown.is_set():
            now = _now_ns()
            self._flush_pm_price_change_until(now)
            self._flush_cb_ticker_until(now)
            self._flush_rtds(now)   # advance the aligned RTDS feed up to `now`
            for slug, st in list(self._markets.items()):
                ttc = (st.close_s * NS - now) / NS
                # drop finished markets to keep the loop light
                if ttc < TTC_MIN - 5:
                    self._markets.pop(slug, None)
                    self._pm_pc_pending_by_slug.pop(slug, None)
                    self._exec_retry_after_by_slug.pop(slug, None)
                    self._resync_armed.discard(slug)
                    self._expired.add(slug)   # never re-register this closed market
                    if self.poly_ws is not None:
                        try: self.poly_ws.remove_market(slug)
                        except Exception: pass
                    continue
                # STRIKE from RTDS oracle at market-open, if the scrape didn't give
                # us one. Snapshot the chainlink price as of open_s (= price-to-beat).
                if st.strike <= 0.0 and now >= st.open_s * NS:
                    px = self._rtds_price_at(st.open_s * NS)
                    if px is not None:
                        st.strike = float(px)
                        self.logger.info("strike from RTDS@open for %s: %.2f (scrape unavailable)", slug, px)
                # HARD book resync ~15s before this market's trading window opens
                # (window is ttc (TTC_MIN, TTC_MAX]; arm once at ttc in (MAX, MAX+RESYNC_LEAD]).
                if slug not in self._resync_armed and TTC_MAX < ttc <= TTC_MAX + RESYNC_LEAD_S:
                    self._resync_armed.add(slug)
                    if self.poly_ws is not None:
                        try:
                            self.poly_ws.request_resync()
                            self.logger.info("pre-window HARD resync armed for %s (ttc=%.0fs)", slug, ttc)
                        except Exception as exc:
                            self.logger.debug("resync request err %s: %r", slug, exc)
                try:
                    await self._evaluate(slug, st, now)
                except Exception as exc:
                    self.logger.debug("evaluate err %s: %r", slug, exc)
            self._stats["ticks"] += 1
            await asyncio.sleep(tick_s)

    async def _book_audit_loop(self) -> None:
        """STRICT book logging for verification: every few seconds snapshot the
        LIVE book (up/dn mid, implied_p_up, asks, evts) for every active market
        into book_audit_<date>.jsonl. Offline we compare these to the recorder's
        book at the same instants — the REAL test (implied_p_up corr > 0.95),
        not mid_sum. Off via FV_BOOK_AUDIT=0."""
        period = float(os.getenv("FV_BOOK_AUDIT_S", "3.0"))
        path0 = None; fh = None
        try:
            while not self._shutdown.is_set():
                now = _now_ns()
                self._flush_pm_price_change_until(now)
                self._flush_cb_ticker_until(now)
                date_iso = datetime.fromtimestamp(now / 1e9, UTC).date().isoformat()
                p = self.decisions_dir / f"book_audit_{date_iso}.jsonl"
                if p != path0:
                    if fh: fh.close()
                    fh = p.open("a", encoding="utf-8"); path0 = p
                for slug, st in list(self._markets.items()):
                    ttc = (st.close_s * NS - now) / NS
                    if not (TTC_MIN - 5 <= ttc <= TTC_MAX + RESYNC_LEAD_S + 10):
                        continue  # only log around the tradeable window
                    try:
                        feats = st.features(now)
                    except Exception:
                        feats = None
                    if not feats:
                        continue
                    um = float(feats.get("up_mid") or 0); dm = float(feats.get("down_mid") or 0)
                    rec = {"ts": now, "slug": slug, "ttc": round(ttc, 1),
                           "up_mid": round(um, 4), "down_mid": round(dm, 4),
                           "mid_sum": round(um + dm, 4),
                           "implied_p_up": round(float(feats.get("implied_p_up") or 0), 4),
                           "up_best_ask": round(float(feats.get("up_best_ask") or 0), 4),
                           "down_best_ask": round(float(feats.get("down_best_ask") or 0), 4),
                           "up_evts5": float(feats.get("up_book_evts_5s") or 0),
                           "dn_evts5": float(feats.get("down_book_evts_5s") or 0)}
                    rec.update(self._pm_clock_fields(slug, now))
                    # FULL feature vector + live p_model, so we can compare EVERY
                    # model feature live-vs-dataset and find what still diverges.
                    try:
                        rec["p_model"] = round(self._predict(feats), 4)
                    except Exception:
                        rec["p_model"] = None
                    rec["f"] = {k: (round(float(feats[k]), 5) if isinstance(feats.get(k), (int, float)) else feats.get(k))
                                for k in self._features if k in feats}
                    fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                fh.flush()
                await asyncio.sleep(period)
        finally:
            if fh: fh.close()

    async def _stats_loop(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(15.0)
            now_ns = _now_ns()
            cl_age = (now_ns - self.cl.latest_recv_ns) / NS if self.cl.latest_recv_ns else None
            cl_src_age = (now_ns - self.cl.latest_cl_ts_ns) / NS if self.cl.latest_cl_ts_ns else None
            cb_age = (now_ns - self.bin.last_ts_ns) / NS if self.bin.last_ts_ns else None
            self.logger.info("stats: markets=%d entered=%d | poly_l2=%d rtds=%d cb=%d strike=%d rest=%d "
                             "| cex_mid=%s cl_px=%s cl_recv_age=%s cl_src_age=%s cb_age=%s "
                             "| ws_reconn(poly/rtds/cb)=%d/%d/%d resync=%d unseeded=%d "
                             "| ooo(poly/rtds/cb)=%d/%d/%d pm_missing_ts=%d "
                             "| ticks=%d enters=%d skips=%d no_edge=%d book_blocked=%d exec_blocked=%d",
                             len(self._markets), len(self._entered),
                             self._stats["poly_l2"], self._stats["rtds"], self._stats["coinbase"],
                             self._stats["strike"], self._stats["rest"],
                             f"{self.bin.last_mid:.1f}" if self.bin.last_mid else "?",
                             f"{self.cl.latest_price:.1f}" if self.cl.latest_price else "?",
                             f"{cl_age:.1f}s" if cl_age is not None else "?",
                             f"{cl_src_age:.1f}s" if cl_src_age is not None else "?",
                             f"{cb_age:.1f}s" if cb_age is not None else "?",
                             (self.poly_ws.stats["ws_reconnects"] if self.poly_ws is not None else 0),
                             self.rtds_ws.stats["reconnects"],
                             self.coinbase_ws.stats["reconnects"],
                             (self.poly_ws.stats.get("ws_resyncs", 0) if self.poly_ws is not None else 0),
                             (self.poly_ws.stats.get("ws_unseeded_records_skipped", 0) if self.poly_ws is not None else 0),
                             self._stats.get("poly_l2_out_of_order", 0),
                             self._stats.get("rtds_out_of_order", 0),
                             self._stats.get("coinbase_out_of_order", 0),
                             self._stats.get("poly_l2_missing_source_ts", 0),
                             self._stats["ticks"], self._stats["enters"], self._stats["skips"],
                             self._stats.get("no_edge", 0),
                             self._stats.get("book_blocked", 0),
                             self._stats.get("exec_blocked", 0))

    async def _reconcile_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                # reconcile_resolved reads zstd resolution files (blocking I/O) —
                # offload so it doesn't freeze the WS consumer (same reason as the
                # order submission).
                now = _now_ns()
                closed = await asyncio.to_thread(self.positions.reconcile_resolved, now)
                # RTDS fallback for positions the resolution recorder missed —
                # without it those losses never reach the daily loss-limit.
                closed += self.positions.reconcile_resolved_rtds(now, self._rtds_price_at)
                for c in closed:
                    self.risk_gate.record_realized(c.resolved_ts_ns, c.realized_pnl_usd)
                    self.logger.info("RESOLVED %s: %s pnl=$%+.4f day=$%+.2f", c.open.market_slug,
                                     "WIN" if c.won else "LOSE", c.realized_pnl_usd, self.risk_gate.today_pnl())
            except Exception as exc:
                self.logger.debug("reconcile err: %r", exc)
            await asyncio.sleep(60.0)

    async def run(self) -> None:
        self.logger.info("=== FAIR-VALUE BOT (mode=%s) ===", "LIVE" if self.router.live else "DRY-RUN/SHADOW")
        self.logger.info("gates: ttc[%g,%g] EV>=%g band[%.2f,%.2f] fixed_shares=%.1f one-entry/market",
                         TTC_MIN, TTC_MAX, EV_THR, PRICE_LO, PRICE_HI, FIXED_SHARES)
        self.logger.info("strategy hours UTC: %s (source=%s raw=%s)",
                         sorted(self._allowed_utc_hours) if self._allowed_utc_hours is not None else "all",
                         self._allowed_utc_hours_source, self._allowed_utc_hours_raw)
        self.logger.info("artifacts: %s", ART)
        self.logger.info("RTDS source-time availability latency: %.1fs", RTDS_FEATURE_LAG_S)
        self.logger.info("PM top-of-book mode: bba_authoritative=%s", PM_BBA_AUTHORITATIVE)
        self.logger.info("book-quality gate: |1-mid_sum|<=%.3f, both-sides-active=%s (anti-bullshit guard)",
                         BOOK_COHERENCE_TOL, BOOK_REQUIRE_BOTH_ACTIVE)
        self.logger.info("execution freshness gate: pm_lag<=%.3fs source_recv_lag<=%.3fs",
                         MAX_EXEC_PM_LAG_S, MAX_EXEC_SOURCE_LAG_S)
        if self._parity_strict:
            self.logger.info("STRICT PARITY: forcing PM book, RTDS, and Coinbase to recorder sources")
        self.logger.info("paths: raw_root=%s decisions_dir=%s", self.raw_root, self.decisions_dir)
        if not (self.raw_root / "raw").exists():
            self.logger.warning("raw_root/raw does not exist: %s", self.raw_root / "raw")
        if self.router.live:
            self.logger.info("preflight: %s", self.router.preflight())
        need_recorder_features = self._rtds_source == "recorder" or self._cex_source == "recorder"
        # Light file tail in direct mode; full fair-value recorder tail when any
        # model feature source is intentionally recorder-authoritative.
        tail = MultiSourceTail(raw_root=self.raw_root,
                               poll_interval_s=float(os.getenv("FV_TAIL_POLL_S", "1.0")),
                               logger=self.logger,
                               fair_value_sources=need_recorder_features,
                               registration_only=(self._pm_book_source == "direct" and not need_recorder_features),
                               polymarket_only=(self._pm_book_source == "recorder" and not need_recorder_features))
        self.logger.info("data plane: PM book=%s, RTDS=%s, Coinbase=%s",
                         self._pm_book_source, self._rtds_source, self._cex_source)
        tasks = [asyncio.create_task(self._ingest_loop(tail), name="registration"),
                 asyncio.create_task(self._tick_loop(), name="tick"),
                 asyncio.create_task(self._reconcile_loop(), name="reconcile"),
                 asyncio.create_task(self._stats_loop(), name="stats")]
        if self._rtds_source == "direct":
            tasks.append(asyncio.create_task(self.rtds_ws.run(), name="rtds_ws"))
        if self._cex_source == "direct":
            tasks.append(asyncio.create_task(self.coinbase_ws.run(), name="coinbase_ws"))
        if self.poly_ws is not None:
            tasks.append(asyncio.create_task(self.poly_ws.run(), name="poly_ws"))
        if os.getenv("FV_BOOK_AUDIT", "1") == "1":
            tasks.append(asyncio.create_task(self._book_audit_loop(), name="book_audit"))
        await self._shutdown.wait()
        if self.poly_ws is not None:
            self.poly_ws.stop()
        self.rtds_ws.stop(); self.coinbase_ws.stop()
        for t in tasks: t.cancel()
        for t in tasks:
            try: await t
            except asyncio.CancelledError: pass
        tail.close()
        if self._l2_audit_fh:
            self._l2_audit_fh.close()
            self._l2_audit_fh = None
        self.logger.info("=== FAIR-VALUE BOT STOPPED ===")

    def trigger_shutdown(self) -> None:
        self._shutdown.set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default="data", type=Path)
    ap.add_argument("--decisions-dir", default="logs/fair_value_bot", type=Path)
    ap.add_argument("--live", action="store_true",
                    help="allow live orders (also needs ENABLE_REAL_ORDERS=1 + signing key)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))
    bot = FairValueBot(raw_root=args.raw_root, decisions_dir=args.decisions_dir, live_orders=args.live)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sname in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sname):
            try: loop.add_signal_handler(getattr(signal, sname), bot.trigger_shutdown)
            except (NotImplementedError, RuntimeError): pass
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        bot.trigger_shutdown(); loop.run_until_complete(asyncio.sleep(0.3))
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
