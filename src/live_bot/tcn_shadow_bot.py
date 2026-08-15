"""Shadow/live runner for the locked BTC 5m TCN double strategy.

It reuses the live fair-value data plane and the same FairValueState extractor
that built the episode dataset, but the decision engine is the locked sequence
model portfolio:

  * early_tcn_75_120_utc02_13
  * late_tcn_50_75_all_day

Default mode is SHADOW: it logs per-strategy decisions for live-vs-replay
parity and never submits orders (no HTTP in the decision loop).

Live mode requires the full triple gate: `--live` flag AND ENABLE_REAL_ORDERS=1
AND signing credentials (the router stays dry-run otherwise). The live path
adds, per the lock's validation requirements: RiskGate (STOP_TRADING kill file
+ daily loss cap, fed by the inherited reconcile loop), a shared per-market
notional cap across both strategy slots (TCN_MAX_MARKET_NOTIONAL_USD), FAK/IOC
order submission off-loop, and position tracking/reconcile.
"""
from __future__ import annotations

import argparse
import asyncio
import heapq
import hashlib
import json
import logging
import math
import os
import signal
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from live_bot.fair_value_bot import (  # noqa: E402
    CEX_TICKER_BUCKET_NS,
    LOG_FORMAT,
    NS,
    PM_BBA_AUTHORITATIVE,
    PM_PRICE_CHANGE_BUCKET_NS,
    RESYNC_LEAD_S,
    RTDS_FEATURE_LAG_S,
    FairValueBot,
    _logit,
    _now_ns,
    _ts_iso,
)
from binance_recorder.compression import ZstdJsonlAppender  # noqa: E402
from fair_value.extractor import BinanceState, ChainlinkState, FairValueState, MarketState  # noqa: E402
from live_bot.file_tail import MultiSourceTail  # noqa: E402
from live_bot.position_manager import OpenPosition  # noqa: E402
from train_episode_tcn import ResidualTCN  # noqa: E402

DEFAULT_DATASET = REPO / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_TCN_ART = REPO / "artifacts" / "btc_5m_episode_tcn_c64_b7_cal_ttc15_90"
DEFAULT_LOCK = REPO / "artifacts" / "strategy_locks" / "tcn_double_strategy_v1_lock.json"

STEP_NS = int(float(os.getenv("TCN_CADENCE_S", "0.2")) * NS)
SEQ_LEN = int(os.getenv("TCN_SEQ_LEN", "1500"))
LOG_SKIPS = os.getenv("TCN_LOG_SKIPS", "1") == "1"
MAX_PM_SOURCE_AGE_S = float(os.getenv("TCN_MAX_PM_SOURCE_AGE_S", "2.0"))
MAX_CEX_AGE_S = float(os.getenv("TCN_MAX_CEX_AGE_S", "2.0"))
MAX_RTDS_SOURCE_AGE_S = float(os.getenv("TCN_MAX_RTDS_SOURCE_AGE_S", "60.0"))
EV_MARGIN = float(os.getenv("TCN_EV_MARGIN", "0.0"))
SOURCE_REPLAY_SETTLE_NS = int(float(os.getenv("TCN_SOURCE_REPLAY_SETTLE_MS", "300")) * 1_000_000)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    path = Path(raw) if raw else default
    return path if path.is_absolute() else REPO / path


DIRECT_CAPTURE_ENABLED = os.getenv("TCN_DIRECT_CAPTURE", "1") == "1"
DIRECT_CAPTURE_ROOT = _env_path("TCN_DIRECT_CAPTURE_ROOT", REPO / "data" / "tcn_direct_capture")
DIRECT_CAPTURE_FLUSH_S = float(os.getenv("TCN_DIRECT_CAPTURE_FLUSH_S", "0.05"))

# live-path knobs (only used with --live); small-stakes defaults per gate 6.3
LIVE_SHARES = float(os.getenv("TCN_LIVE_SHARES", "1.0"))
MAX_MARKET_NOTIONAL_USD = float(os.getenv("TCN_MAX_MARKET_NOTIONAL_USD", "5.0"))
LIVE_ORDER_BUFFER = float(os.getenv("TCN_LIVE_ORDER_BUFFER", "0.03"))  # lock slippage cap


def _sigmoid(x: float) -> float:
    x = max(-50.0, min(50.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _finite_float(v: Any) -> float | None:
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return x if math.isfinite(x) else None
    return None


def _hash_float_row(row: np.ndarray) -> str:
    arr = np.asarray(row, dtype=np.float32)
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()


@dataclass(frozen=True)
class LockedStrategy:
    strategy_id: str
    beta: float
    ttc_min: float
    ttc_max: float
    ev_min: float
    price_lo: float
    price_hi: float
    utc_hours: set[int] | None
    size_k: float | None = None      # per-slot linear-EV sizing override
    size_max: float | None = None

    @classmethod
    def from_lock(cls, row: dict[str, Any]) -> "LockedStrategy":
        hours_raw = row.get("utc_hours_allowed")
        hours = None if hours_raw == "all" else {int(x) for x in hours_raw}
        sizing = row.get("sizing") or {}
        return cls(
            strategy_id=str(row["strategy_id"]),
            beta=float(row["beta"]),
            ttc_min=float(row["ttc_min_s_exclusive"]),
            ttc_max=float(row["ttc_max_s_inclusive"]),
            ev_min=float(row["ev_min"]),
            price_lo=float(row["price_min_exclusive"]),
            price_hi=float(row["price_max_exclusive"]),
            utc_hours=hours,
            size_k=float(sizing["k"]) if "k" in sizing else None,
            size_max=float(sizing["s_max"]) if "s_max" in sizing else None,
        )

    def allowed_at(self, now_ns: int) -> bool:
        if self.utc_hours is None:
            return True
        return datetime.fromtimestamp(now_ns / NS, tz=UTC).hour in self.utc_hours

    def in_ttc(self, ttc_s: float) -> bool:
        return self.ttc_min < ttc_s <= self.ttc_max


@dataclass
class EpisodeBuffer:
    norm_x: np.ndarray
    valid: np.ndarray
    filled: np.ndarray
    raw_x: np.ndarray
    p_market: np.ndarray
    last_step: int = -1
    last_delta: float = float("nan")


class DirectFeedCapture:
    """Write the exact direct feed used by TCN shadow into builder-compatible raw files."""

    def __init__(self, root: Path, *, flush_s: float, logger: logging.Logger) -> None:
        self.root = (root if root.name == "raw" else root / "raw").resolve()
        self.flush_ns = max(0, int(float(flush_s) * NS))
        self.logger = logger
        self.session = f"{datetime.now(tz=UTC).strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self._writers: dict[Path, ZstdJsonlAppender] = {}
        self._last_flush_ns = 0
        self.rows = 0

    @staticmethod
    def _date_from_ns(ts_ns: int) -> str:
        ts = int(ts_ns or _now_ns())
        return datetime.fromtimestamp(ts / NS, UTC).date().isoformat()

    @staticmethod
    def _hour_from_ns(ts_ns: int) -> str:
        ts = int(ts_ns or _now_ns())
        return datetime.fromtimestamp(ts / NS, UTC).strftime("%H")

    def _writer(self, path: Path) -> ZstdJsonlAppender:
        writer = self._writers.get(path)
        if writer is None:
            writer = ZstdJsonlAppender(path)
            self._writers[path] = writer
        return writer

    def _write_record(self, path: Path, rec: dict[str, Any]) -> None:
        try:
            self._writer(path).write_json(rec)
            self.rows += 1
            now = _now_ns()
            if self.flush_ns == 0 or now - self._last_flush_ns >= self.flush_ns:
                for writer in self._writers.values():
                    writer.flush(frame=True)
                self._last_flush_ns = now
        except Exception as exc:
            self.logger.warning("direct_capture write failed path=%s err=%r", path, exc)

    def write_pm(self, rec: dict[str, Any]) -> None:
        slug = str(rec.get("selected_market_slug") or "")
        if not slug:
            return
        recv_ns = int(rec.get("recv_ts_ns") or _now_ns())
        d = self._date_from_ns(recv_ns)
        hour = self._hour_from_ns(recv_ns)
        path = (
            self.root / "polymarket" / "btc_updown_5m" / d
            / f"{hour}__{slug}__tcn-direct-{self.session}.l2.jsonl.zst"
        )
        self._write_record(path, rec)

    def write_coinbase_ticker(self, ts_ns: int, bid: float, ask: float, bq: float, aq: float) -> None:
        recv_ns = _now_ns()
        d = self._date_from_ns(recv_ns)
        hour = self._hour_from_ns(recv_ns)
        rec = {
            "recv_ts_ns": recv_ns,
            "channel": "ticker",
            "source_timestamps": {"message_timestamp_ns": int(ts_ns)},
            "payload": {
                "channel": "ticker",
                "events": [{
                    "type": "update",
                    "tickers": [{
                        "product_id": "BTC-USD",
                        "best_bid": f"{float(bid):.2f}",
                        "best_ask": f"{float(ask):.2f}",
                        "best_bid_quantity": f"{float(bq):.8f}",
                        "best_ask_quantity": f"{float(aq):.8f}",
                    }],
                }],
            },
            "capture_source": "tcn_direct",
            "capture_session": self.session,
        }
        path = self.root / "coinbase" / "advanced" / "BTC-USD" / d / f"{hour}__tcn-direct-{self.session}.ws.jsonl.zst"
        self._write_record(path, rec)

    def write_coinbase_trade(self, ts_ns: int, qty: float, side: float) -> None:
        recv_ns = _now_ns()
        d = self._date_from_ns(recv_ns)
        hour = self._hour_from_ns(recv_ns)
        side_s = "BUY" if side > 0 else ("SELL" if side < 0 else "")
        rec = {
            "recv_ts_ns": recv_ns,
            "channel": "market_trades",
            "source_timestamps": {"message_timestamp_ns": int(ts_ns)},
            "payload": {
                "channel": "market_trades",
                "events": [{
                    "type": "update",
                    "trades": [{
                        "product_id": "BTC-USD",
                        "size": f"{float(qty):.8f}",
                        "side": side_s,
                    }],
                }],
            },
            "capture_source": "tcn_direct",
            "capture_session": self.session,
        }
        path = self.root / "coinbase" / "advanced" / "BTC-USD" / d / f"{hour}__tcn-direct-{self.session}.ws.jsonl.zst"
        self._write_record(path, rec)

    def write_rtds(self, recv_ts_ns: int, cl_ts_ns: int, price: float) -> None:
        recv_ns = int(recv_ts_ns or _now_ns())
        d = self._date_from_ns(recv_ns)
        hour = self._hour_from_ns(recv_ns)
        rec = {
            "record_type": "chainlink_live_reference",
            "recv_ts_ns": recv_ns,
            "chainlink_ts_ms": int(cl_ts_ns) // 1_000_000,
            "btc_usd_price": float(price),
            "capture_source": "tcn_direct",
            "capture_session": self.session,
        }
        path = (
            self.root / "polymarket" / "rtds" / "crypto_prices_chainlink" / "btc_usd" / d
            / f"{hour}__tcn-direct-{self.session}.ws.jsonl.zst"
        )
        self._write_record(path, rec)

    def close(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception as exc:
                self.logger.warning("direct_capture close failed path=%s err=%r", writer.path, exc)
        self._writers.clear()


class TCNShadowBot(FairValueBot):
    """FairValueBot data plane with a TCN sequence decision engine."""

    def __init__(
        self,
        *,
        raw_root: Path = Path("data"),
        decisions_dir: Path = Path("logs/tcn_shadow_bot"),
        live_orders: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self._dataset_dir = _env_path("TCN_DATASET_DIR", DEFAULT_DATASET)
        self._tcn_art_dir = _env_path("TCN_ARTIFACTS_DIR", DEFAULT_TCN_ART)
        self._lock_path = _env_path("TCN_STRATEGY_LOCK", DEFAULT_LOCK)
        self._episode_buffers: dict[str, EpisodeBuffer] = {}
        self._entered_strategy: set[tuple[str, str]] = set()
        self._last_eval_step_by_slug: dict[str, int] = {}
        self._last_replay_grid_ns: int | None = None
        self._receptive_steps = SEQ_LEN
        # --live enables the order path (risk gate + exposure cap + router);
        # the router itself stays dry-run unless ENABLE_REAL_ORDERS=1 + creds
        self._live_path = bool(live_orders)
        self._market_notional_usd: dict[str, float] = {}
        super().__init__(raw_root=raw_root, decisions_dir=decisions_dir,
                         live_orders=live_orders, logger=logger)
        self._replay_bin = BinanceState()
        self._replay_cl = ChainlinkState()
        self._replay_markets: dict[str, FairValueState] = {}
        self._replay_events: list[tuple[int, int, int, Any]] = []
        self._replay_seq = 0
        self._replay_pm_recv_watermark_by_slug: dict[str, int] = {}
        self._replay_pm_source_watermark_by_slug: dict[str, int] = {}
        self._replay_applied_until_ns = 0
        self._direct_capture = (
            DirectFeedCapture(DIRECT_CAPTURE_ROOT, flush_s=DIRECT_CAPTURE_FLUSH_S, logger=self.logger)
            if DIRECT_CAPTURE_ENABLED else None
        )
        self._stats["tcn_eval"] = 0
        self._stats["tcn_invalid"] = 0
        self._stats["tcn_warmup_blocked"] = 0
        self._stats["tcn_scheduler_dropped_steps"] = 0
        self._stats["tcn_source_replay_settle_ms"] = SOURCE_REPLAY_SETTLE_NS / 1_000_000
        if self._direct_capture is not None:
            self.logger.info("TCN direct feed capture enabled: root=%s session=%s",
                             self._direct_capture.root, self._direct_capture.session)

    def _load_model(self) -> None:
        norm = json.loads((self._dataset_dir / "normalization.json").read_text(encoding="utf-8"))
        self._features = [str(x) for x in norm["feature_names"]]
        self._norm_mean = np.asarray(norm["mean"], dtype=np.float32)
        self._norm_std = np.asarray(norm["std"], dtype=np.float32)
        self._norm_std[self._norm_std <= 1e-12] = 1.0
        self._require_cex = True
        self._feature_policy = {
            "feature_set": "btc_5m_episodes_v1_200ms",
            "require_cex": True,
            "normalization": str(self._dataset_dir / "normalization.json"),
        }

        # ensemble support: TCN_ARTIFACTS_DIRS (';'-separated) overrides the
        # single TCN_ARTIFACTS_DIR; deltas are averaged across members
        dirs_raw = os.getenv("TCN_ARTIFACTS_DIRS", "").strip()
        if dirs_raw:
            self._tcn_art_dirs = [
                (Path(p) if Path(p).is_absolute() else REPO / p)
                for p in dirs_raw.split(";") if p.strip()
            ]
        else:
            self._tcn_art_dirs = [self._tcn_art_dir]
        self._device = torch.device("cuda" if torch.cuda.is_available()
                                    and os.getenv("TCN_FORCE_CPU", "0") != "1" else "cpu")
        self._tcn_models = []
        receptive = SEQ_LEN
        for art_dir in self._tcn_art_dirs:
            report = json.loads((art_dir / "tcn_report.json").read_text(encoding="utf-8"))
            model_cfg = report["model"]
            model = ResidualTCN(
                n_features=int(model_cfg["n_features"]),
                channels=int(model_cfg["channels"]),
                blocks=int(model_cfg["blocks"]),
                kernel_size=int(model_cfg["kernel_size"]),
                dropout=0.0,
                residual_scale=float(model_cfg.get("residual_scale", 0.5)),
            ).to(self._device)
            state = torch.load(art_dir / "model.pt", map_location=self._device)
            model.load_state_dict(state)
            model.eval()
            self._tcn_models.append(model)
            receptive = min(receptive, int(getattr(model, "receptive_steps", SEQ_LEN)))
        self._tcn_model = self._tcn_models[0]   # kept for backward compat
        self._receptive_steps = receptive

        lock = json.loads(self._lock_path.read_text(encoding="utf-8"))
        self._lock_id = str(lock.get("lock_id", "unknown_lock"))
        # optional per-member Platt calibration (v2.1 locks)
        prob_cfg = lock.get("probability") or {}
        self._platt_params: list[tuple[float, float]] | None = None
        if str(prob_cfg.get("kind", "")) == "per_member_platt":
            params = []
            for art_dir in self._tcn_art_dirs:
                rep = json.loads((art_dir / "tcn_report.json").read_text(encoding="utf-8"))
                cal = rep["calibrators"]["val_platt_l2"]
                params.append((float(cal["coef"]), float(cal["intercept"])))
            self._platt_params = params
            self.logger.info("per-member platt calibration enabled (%d members)", len(params))
        # optional EV-proportional sizing (v2.1 locks)
        sizing_cfg = lock.get("sizing") or {}
        self._sizing = None
        if str(sizing_cfg.get("kind", "")) in ("linear_ev", "base_linear_ev"):
            self._sizing = {
                "base": float(sizing_cfg.get("base", 0.0)),
                "k": float(sizing_cfg["k"]),
                "s_max": float(sizing_cfg["s_max"]),
                "size_scale": float(sizing_cfg.get("size_scale", 1.0)),
            }
            self.logger.info("EV sizing enabled: %s", self._sizing)
        self._strategies = [
            LockedStrategy.from_lock(x)
            for x in lock["strategies"]
            if bool(x.get("enabled", True))
        ]
        self._min_ttc = min(s.ttc_min for s in self._strategies)
        self._max_ttc = max(s.ttc_max for s in self._strategies)
        self.logger.info(
            "loaded TCN shadow model from %s dataset=%s features=%d strategies=%s device=%s",
            self._tcn_art_dir, self._dataset_dir, len(self._features),
            [s.strategy_id for s in self._strategies], self._device,
        )
        self._startup_selfchecks(lock)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    def _startup_selfchecks(self, lock: dict) -> None:
        """Bind the run to exact artifact versions and refuse obvious config
        drift. Hashes land in every decision record via _decision_meta."""
        self._artifact_hashes = {
            "model_pt_sha256": [self._file_sha256(d / "model.pt") for d in self._tcn_art_dirs],
            "model_dirs": [str(d) for d in self._tcn_art_dirs],
            "normalization_sha256": self._file_sha256(self._dataset_dir / "normalization.json"),
            "lock_sha256": self._file_sha256(self._lock_path),
        }
        expected = lock.get("model_artifacts") or (
            [lock["model_artifact"]] if lock.get("model_artifact") else [])
        expected = [str(x).replace("\\", "/") for x in expected]
        running = [str(d).replace("\\", "/") for d in self._tcn_art_dirs]
        if expected and (
            len(expected) != len(running)
            or any(not r.endswith(e) for r, e in zip(running, expected))
        ):
            msg = (f"model artifact mismatch: lock expects {expected} "
                   f"but running {running}")
            if self._live_path:
                raise RuntimeError(msg + " — refusing to start with --live")
            self.logger.warning("%s (shadow mode, continuing)", msg)
        self.logger.info("artifact hashes: %s", self._artifact_hashes)

    def _decision_meta(self) -> dict:
        base = super()._decision_meta()
        base.update({
            "bot": "tcn_shadow_bot",
            "lock_id": self._lock_id,
            "dataset_dir": str(self._dataset_dir),
            "tcn_artifacts_dir": str(self._tcn_art_dir),
            "strategy_lock": str(self._lock_path),
            "cadence_s": STEP_NS / NS,
            "seq_len": SEQ_LEN,
            "receptive_steps": self._receptive_steps,
            "max_pm_source_age_s": MAX_PM_SOURCE_AGE_S,
            "max_cex_age_s": MAX_CEX_AGE_S,
            "max_rtds_source_age_s": MAX_RTDS_SOURCE_AGE_S,
            "ev_margin": EV_MARGIN,
            "source_replay_settle_ms": SOURCE_REPLAY_SETTLE_NS / 1_000_000,
            "direct_capture_root": str(self._direct_capture.root) if self._direct_capture is not None else None,
            "direct_capture_session": self._direct_capture.session if self._direct_capture is not None else None,
            "live_path": self._live_path,
            "artifact_hashes": getattr(self, "_artifact_hashes", None),
        })
        return base

    def _ensure_replay_market(self, slug: str) -> FairValueState | None:
        hot = self._markets.get(slug)
        if hot is None:
            return None
        st = self._replay_markets.get(slug)
        if st is None:
            st = FairValueState(
                pm=MarketState(bba_authoritative=PM_BBA_AUTHORITATIVE),
                bin=self._replay_bin,
                cl=self._replay_cl,
                strike=float(hot.strike),
                open_s=int(hot.open_s),
                close_s=int(hot.close_s),
                require_cex=self._require_cex,
            )
            st.pm.market_open_s = int(hot.open_s)
            st.pm.market_close_s = int(hot.close_s)
            self._replay_markets[slug] = st
        else:
            st.strike = float(hot.strike)
            st.open_s = int(hot.open_s)
            st.close_s = int(hot.close_s)
            st.pm.market_open_s = int(hot.open_s)
            st.pm.market_close_s = int(hot.close_s)
        return st

    def _queue_replay_event(self, ts_ns: int, kind: int, payload: Any) -> None:
        if int(ts_ns or 0) <= 0:
            return
        self._replay_seq += 1
        heapq.heappush(self._replay_events, (int(ts_ns), int(kind), self._replay_seq, payload))

    def _queue_replay_l2(self, rec: dict) -> None:
        source_ts_ns = self._pm_source_ts_ns(rec)
        if source_ts_ns > 0:
            self._queue_replay_event(source_ts_ns, 0, rec)

    def _queue_replay_ticker(self, ts: int, bid: float, ask: float, bq: float, aq: float) -> None:
        self._queue_replay_event(int(ts), 1, (int(ts), float(bid), float(ask), float(bq), float(aq)))

    def _queue_replay_trade(self, ts: int, qty: float, side: float) -> None:
        self._queue_replay_event(int(ts), 2, (int(ts), float(qty), float(side)))

    def _queue_replay_rtds(self, recv_ts_ns: int, cl_ts_ns: int, price: float) -> None:
        if not (price > 0 and cl_ts_ns > 0):
            return
        avail_ns = int(cl_ts_ns) + int(self._rtds_lag_ns) if self._rtds_lag_ns > 0 else int(recv_ts_ns)
        self._queue_replay_event(avail_ns, 3, (avail_ns, int(cl_ts_ns), float(price)))

    def _flush_source_replay_until(self, cutoff_ns: int) -> None:
        while self._replay_events and self._replay_events[0][0] <= cutoff_ns:
            ts, kind, _seq, payload = heapq.heappop(self._replay_events)
            if kind == 0:
                rec = payload
                slug = str(rec.get("selected_market_slug") or "")
                st = self._ensure_replay_market(slug)
                if st is None:
                    continue
                source_ts_ns = self._pm_source_ts_ns(rec)
                if source_ts_ns <= 0:
                    continue
                if source_ts_ns < self._replay_pm_source_watermark_by_slug.get(slug, 0):
                    self._stats["tcn_replay_poly_ooo"] = self._stats.get("tcn_replay_poly_ooo", 0) + 1
                    continue
                recv_ts_ns = int(rec.get("recv_ts_ns") or 0)
                state_rec = dict(rec)
                state_rec["raw_recv_ts_ns"] = recv_ts_ns
                state_rec["recv_ts_ns"] = source_ts_ns
                state_rec["source_ts_ns"] = source_ts_ns
                if recv_ts_ns > 0:
                    state_rec["pm_delivery_lag_s"] = (recv_ts_ns - source_ts_ns) / NS
                    self._replay_pm_recv_watermark_by_slug[slug] = max(
                        recv_ts_ns,
                        self._replay_pm_recv_watermark_by_slug.get(slug, 0),
                    )
                self._replay_pm_source_watermark_by_slug[slug] = max(
                    source_ts_ns,
                    self._replay_pm_source_watermark_by_slug.get(slug, 0),
                )
                st.update_pm(state_rec)
            elif kind == 1:
                tick_ts, bid, ask, bq, aq = payload
                if tick_ts >= int(self._replay_bin.last_ts_ns or 0):
                    self._replay_bin.update_bookticker(tick_ts, bid, ask, bq, aq)
            elif kind == 2:
                trade_ts, qty, side = payload
                self._replay_bin.update_trade(trade_ts, qty, side)
            elif kind == 3:
                avail_ns, cl_ts_ns, price = payload
                self._replay_cl.update(avail_ns, cl_ts_ns, price, self._replay_bin)
            self._replay_applied_until_ns = max(self._replay_applied_until_ns, int(ts))
        self._stats["tcn_replay_pending"] = len(self._replay_events)

    def _replay_pm_clock_fields(self, slug: str, now_ns: int) -> dict[str, Any]:
        recv_watermark = int(self._replay_pm_recv_watermark_by_slug.get(slug, 0) or 0)
        source_watermark = int(self._replay_pm_source_watermark_by_slug.get(slug, 0) or 0)
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

    def _on_l2(self, rec: dict) -> None:
        super()._on_l2(rec)

    def _apply_l2(self, rec: dict) -> None:
        before = int(self._stats.get("poly_l2", 0))
        super()._apply_l2(rec)
        applied = int(self._stats.get("poly_l2", 0)) > before
        if applied:
            self._queue_replay_l2(rec)
        capture = getattr(self, "_direct_capture", None)
        if capture is not None and self._pm_book_source == "direct" and applied:
            capture.write_pm(rec)
            self._stats["direct_capture_rows"] = capture.rows

    def _buffer_cb_ticker(self, ts: int, bid: float, ask: float, bq: float, aq: float) -> None:
        capture = getattr(self, "_direct_capture", None)
        if capture is not None and self._cex_source == "direct":
            capture.write_coinbase_ticker(ts, bid, ask, bq, aq)
            self._stats["direct_capture_rows"] = capture.rows
        super()._buffer_cb_ticker(ts, bid, ask, bq, aq)

    def _on_cb_ticker(self, ts: int, bid: float, ask: float, bq: float, aq: float) -> None:
        before = int(self._stats.get("coinbase", 0))
        super()._on_cb_ticker(ts, bid, ask, bq, aq)
        if int(self._stats.get("coinbase", 0)) > before:
            self._queue_replay_ticker(ts, bid, ask, bq, aq)

    def _on_cb_trade(self, ts: int, qty: float, side: float) -> None:
        capture = getattr(self, "_direct_capture", None)
        if capture is not None and self._cex_source == "direct":
            capture.write_coinbase_trade(ts, qty, side)
            self._stats["direct_capture_rows"] = capture.rows
        super()._on_cb_trade(ts, qty, side)
        self._queue_replay_trade(ts, qty, side)

    def _on_rtds(self, recv_ts_ns: int, cl_ts_ns: int, price: float) -> None:
        capture = getattr(self, "_direct_capture", None)
        if capture is not None and self._rtds_source == "direct":
            capture.write_rtds(recv_ts_ns, cl_ts_ns, price)
            self._stats["direct_capture_rows"] = capture.rows
        accepted = not (
            recv_ts_ns
            and self._last_rtds_actual_recv_ns
            and recv_ts_ns < self._last_rtds_actual_recv_ns
        )
        super()._on_rtds(recv_ts_ns, cl_ts_ns, price)
        if accepted:
            self._queue_replay_rtds(recv_ts_ns, cl_ts_ns, price)

    def _close_direct_capture(self) -> None:
        capture = getattr(self, "_direct_capture", None)
        if capture is not None:
            capture.close()
            self.logger.info("TCN direct feed capture closed: rows=%d root=%s",
                             capture.rows, capture.root)

    def _buffer_for(self, slug: str) -> EpisodeBuffer:
        buf = self._episode_buffers.get(slug)
        if buf is None:
            n = len(self._features)
            buf = EpisodeBuffer(
                norm_x=np.zeros((SEQ_LEN, n), dtype=np.float32),
                valid=np.zeros(SEQ_LEN, dtype=bool),
                filled=np.zeros(SEQ_LEN, dtype=bool),
                raw_x=np.zeros((SEQ_LEN, n), dtype=np.float32),
                p_market=np.full(SEQ_LEN, 0.5, dtype=np.float32),
            )
            self._episode_buffers[slug] = buf
        return buf

    def _slot_for(self, st, now_ns: int) -> tuple[int, int] | None:
        if STEP_NS <= 0:
            return None
        rel = int(now_ns) - int(st.open_s) * NS
        if rel < 0:
            return None
        step = rel // STEP_NS
        if step < 0 or step >= SEQ_LEN:
            return None
        grid_ns = int(st.open_s) * NS + int(step) * STEP_NS
        return int(step), int(grid_ns)

    def _valid_source(self, slug: str, now_ns: int, raw: np.ndarray) -> tuple[bool, dict[str, Any]]:
        pm = self._replay_pm_clock_fields(slug, now_ns)
        pm_age = pm.get("pm_source_lag_s")
        cex_age = (
            (now_ns - int(self._replay_bin.last_ts_ns or 0)) / NS
            if self._replay_bin.last_ts_ns else float("nan")
        )
        rtds_source_age = (
            (now_ns - int(self._replay_cl.latest_cl_ts_ns or 0)) / NS
            if self._replay_cl.latest_cl_ts_ns else float("nan")
        )
        finite = bool(np.isfinite(raw).all())
        valid = finite
        if pm_age is None or not math.isfinite(float(pm_age)) or float(pm_age) > MAX_PM_SOURCE_AGE_S:
            valid = False
        if not math.isfinite(cex_age) or cex_age > MAX_CEX_AGE_S:
            valid = False
        if not math.isfinite(rtds_source_age) or rtds_source_age > MAX_RTDS_SOURCE_AGE_S:
            valid = False
        audit = {
            **pm,
            "cex_age_s": round(cex_age, 6) if math.isfinite(cex_age) else None,
            "rtds_source_age_s": round(rtds_source_age, 6) if math.isfinite(rtds_source_age) else None,
            "finite_features": finite,
            "source_valid": valid,
            "source_replay_applied_until_ns": self._replay_applied_until_ns,
            "source_replay_pending": len(self._replay_events),
        }
        return valid, audit

    def _append_step(self, slug: str, st, now_ns: int, feats: dict[str, Any]) -> tuple[int, bool, dict[str, Any]]:
        slot = self._slot_for(st, now_ns)
        if slot is None:
            return -1, False, {"source_valid": False, "invalid_reason": "outside_episode_grid"}
        step, grid_ns = slot
        buf = self._buffer_for(slug)
        raw = np.asarray([float(feats.get(c, np.nan)) for c in self._features], dtype=np.float32)
        valid, audit = self._valid_source(slug, grid_ns, raw)
        if valid:
            norm = (raw - self._norm_mean) / self._norm_std
            norm[~np.isfinite(norm)] = 0.0
            buf.norm_x[step, :] = norm
            buf.raw_x[step, :] = raw
            buf.valid[step] = True
            buf.p_market[step] = float(np.clip(float(feats["implied_p_up"]), 1e-5, 1.0 - 1e-5))
        else:
            buf.norm_x[step, :] = 0.0
            buf.raw_x[step, :] = 0.0
            buf.valid[step] = False
            buf.p_market[step] = 0.5
        buf.last_step = max(buf.last_step, step)
        audit["step_idx"] = step
        audit["grid_ns"] = grid_ns
        audit["raw_feature_hash"] = _hash_float_row(raw)
        audit["norm_feature_hash"] = _hash_float_row(buf.norm_x[step, :])
        buf.filled[step] = True
        return step, valid, audit

    def _append_invalid_step(self, slug: str, st, now_ns: int, reason: str) -> tuple[int, dict[str, Any]]:
        slot = self._slot_for(st, now_ns)
        if slot is None:
            return -1, {"source_valid": False, "invalid_reason": "outside_episode_grid"}
        step, grid_ns = slot
        buf = self._buffer_for(slug)
        buf.norm_x[step, :] = 0.0
        buf.raw_x[step, :] = 0.0
        buf.valid[step] = False
        buf.filled[step] = True
        buf.p_market[step] = 0.5
        buf.last_step = max(buf.last_step, step)
        audit = {
            "source_valid": False,
            "invalid_reason": reason,
            "step_idx": step,
            "grid_ns": grid_ns,
            "raw_feature_hash": _hash_float_row(buf.raw_x[step, :]),
            "norm_feature_hash": _hash_float_row(buf.norm_x[step, :]),
        }
        return step, audit

    def _history_status(self, slug: str, step: int) -> tuple[bool, dict[str, Any]]:
        buf = self._buffer_for(slug)
        receptive = max(1, min(int(self._receptive_steps), SEQ_LEN))
        start = max(0, int(step) - receptive + 1)
        window = buf.filled[start:int(step) + 1]
        filled_count = int(window.sum())
        needed = int(window.size)
        missing = max(0, needed - filled_count)
        ready = missing == 0
        return ready, {
            "history_ready": ready,
            "history_missing_steps": missing,
            "history_window_start_step": start,
            "history_window_steps": needed,
            "receptive_steps": receptive,
        }

    @torch.no_grad()
    def _predict_delta_for_step(self, slug: str, step: int,
                                base_logit: float | None = None) -> float:
        """Mean ensemble delta; with per-member Platt calibration enabled (lock
        `probability.kind == "per_member_platt"`) returns the EFFECTIVE delta
        logit(p_ensemble) - base_logit, so downstream p_strategy code is
        unchanged (beta should then be 1.0)."""
        buf = self._buffer_for(slug)
        x = np.concatenate([buf.norm_x, buf.valid[:, None].astype(np.float32)], axis=1)
        xb = torch.from_numpy(x[None, :, :]).to(self._device)
        deltas = [float(m(xb)[0, step].detach().cpu().item()) for m in self._tcn_models]
        if self._platt_params is not None and base_logit is not None:
            probs = [
                _sigmoid(coef * (base_logit + d) + icpt)
                for d, (coef, icpt) in zip(deltas, self._platt_params)
            ]
            p_ens = min(max(float(np.mean(probs)), 1e-5), 1.0 - 1e-5)
            value = float(_logit(p_ens) - base_logit)
        else:
            value = float(np.mean(deltas))
        buf.last_delta = value
        return value

    def _log_decision(self, now_ns: int, rec: dict[str, Any]) -> None:
        rec.update(self._decision_meta())
        self._write(now_ns, rec)

    async def _evaluate_grid(self, slug: str, st, grid_ns: int) -> None:
        slot = self._slot_for(st, grid_ns)
        if slot is None:
            return
        step, grid_ns = slot
        if self._last_eval_step_by_slug.get(slug) == step:
            return
        self._last_eval_step_by_slug[slug] = step

        ttc = (st.close_s * NS - grid_ns) / NS
        feats = st.features(grid_ns)
        if feats is None:
            step, audit = self._append_invalid_step(slug, st, grid_ns, "features_unavailable")
            self._stats["tcn_invalid"] += 1
            active = [s for s in self._strategies if s.in_ttc(ttc) and s.allowed_at(grid_ns)]
            if LOG_SKIPS and active and step >= 0:
                for strat in active:
                    self._stats["skips"] += 1
                    self._log_decision(grid_ns, {
                        "logged_at_ns": _now_ns(),
                        "snapshot_ts_ns": grid_ns,
                        "iso": _ts_iso(grid_ns),
                        "market_slug": slug,
                        "strategy_id": strat.strategy_id,
                        "decision": "SKIP_SOURCE",
                        "ttc_s": round(ttc, 3),
                        "audit": audit,
                    })
            return

        step, source_valid, audit = self._append_step(slug, st, grid_ns, feats)
        if step < 0:
            return
        history_ready, history_audit = self._history_status(slug, step)
        audit.update(history_audit)
        active = [s for s in self._strategies if s.in_ttc(ttc) and s.allowed_at(grid_ns)]
        if not active:
            return

        self._stats["tcn_eval"] += len(active)
        if not source_valid or not history_ready:
            if source_valid and not history_ready:
                self._stats["tcn_warmup_blocked"] = self._stats.get("tcn_warmup_blocked", 0) + len(active)
                audit["invalid_reason"] = "history_warmup"
            if LOG_SKIPS:
                for strat in active:
                    self._stats["skips"] += 1
                    self._log_decision(grid_ns, {
                        "logged_at_ns": _now_ns(),
                        "snapshot_ts_ns": grid_ns,
                        "iso": _ts_iso(grid_ns),
                        "market_slug": slug,
                        "strategy_id": strat.strategy_id,
                        "decision": "SKIP_SOURCE",
                        "ttc_s": round(ttc, 3),
                        "audit": audit,
                    })
            return

        p_market = float(np.clip(float(feats["implied_p_up"]), 1e-5, 1.0 - 1e-5))
        delta = self._predict_delta_for_step(slug, step, base_logit=_logit(p_market))
        up_ask = float(feats.get("up_best_ask") or 0.0)
        dn_ask = float(feats.get("down_best_ask") or 0.0)
        book_reason = self._book_quality_reason(feats)

        for strat in active:
            key = (slug, strat.strategy_id)
            if key in self._entered_strategy:
                continue
            p = _sigmoid(_logit(p_market) + strat.beta * delta)
            ev_gate = strat.ev_min + EV_MARGIN
            ev_up = p - up_ask
            ev_dn = (1.0 - p) - dn_ask
            side: str | None = None
            fill: float | None = None
            ev: float | None = None
            if (
                ev_up >= ev_gate
                and ev_up >= ev_dn
                and strat.price_lo < up_ask < strat.price_hi
            ):
                side, fill, ev = "UP", up_ask, ev_up
            elif (
                ev_dn >= ev_gate
                and ev_dn > ev_up
                and strat.price_lo < dn_ask < strat.price_hi
            ):
                side, fill, ev = "DOWN", dn_ask, ev_dn

            rec = {
                "logged_at_ns": _now_ns(),
                "snapshot_ts_ns": grid_ns,
                "iso": _ts_iso(grid_ns),
                "market_slug": slug,
                "strategy_id": strat.strategy_id,
                "step_idx": step,
                "ttc_s": round(ttc, 3),
                "p_market": round(p_market, 6),
                "tcn_delta": round(delta, 6),
                "beta": strat.beta,
                "p_strategy": round(p, 6),
                "ev_min": strat.ev_min,
                "ev_gate": round(ev_gate, 6),
                "ev_margin": EV_MARGIN,
                "up_best_ask": round(up_ask, 4),
                "down_best_ask": round(dn_ask, 4),
                "ev_up": round(ev_up, 6),
                "ev_down": round(ev_dn, 6),
                "side": side,
                "fill_quote": None if fill is None else round(fill, 4),
                "ev": None if ev is None else round(ev, 6),
                "mid_sum": round(float(feats.get("up_mid") or 0.0) + float(feats.get("down_mid") or 0.0), 6),
                "up_book_evts_5s": _finite_float(feats.get("up_book_evts_5s")),
                "down_book_evts_5s": _finite_float(feats.get("down_book_evts_5s")),
                "audit": audit,
                "f": self._feature_snapshot(feats),
            }
            if side is None:
                if LOG_SKIPS:
                    rec["decision"] = "SKIP_NO_EDGE"
                    self._stats["skips"] += 1
                    self._stats["no_edge"] = self._stats.get("no_edge", 0) + 1
                    self._log_decision(grid_ns, rec)
                continue
            if book_reason is not None:
                rec["decision"] = "SKIP_BOOK"
                rec["book_reason"] = book_reason
                self._stats["skips"] += 1
                self._stats["book_blocked"] = self._stats.get("book_blocked", 0) + 1
                self._log_decision(grid_ns, rec)
                continue
            stale_sources = []
            pm_age = audit.get("pm_source_lag_s")
            if pm_age is None or float(pm_age) > MAX_PM_SOURCE_AGE_S:
                stale_sources.append(f"pm_lag_s>{MAX_PM_SOURCE_AGE_S:.3f}")
            cex_age = audit.get("cex_age_s")
            if cex_age is None or float(cex_age) > MAX_CEX_AGE_S:
                stale_sources.append(f"cex_age_s>{MAX_CEX_AGE_S:.3f}")
            if stale_sources:
                rec["decision"] = "SKIP_EXEC_STALE_SOURCE"
                rec["exec_reason"] = " ".join(stale_sources)
                self._stats["skips"] += 1
                self._stats["exec_blocked"] = self._stats.get("exec_blocked", 0) + 1
                self._log_decision(grid_ns, rec)
                continue

            if not self._live_path:
                rec["decision"] = "ENTER"
                rec["router_mode"] = "shadow_no_order"
                rec["size_shares"] = round(self._shares_for(strat, float(ev)), 4)
                self._entered_strategy.add(key)
                self._entered.add(f"{slug}:{strat.strategy_id}")
                self._stats["enters"] += 1
                self._log_decision(grid_ns, rec)
                self.logger.info(
                    "TCN ENTER %s %s [%s] p=%.3f fill=%.3f ev=%.3f ttc=%.0fs",
                    strat.strategy_id, slug, side, p, float(fill), float(ev), ttc,
                )
                continue
            await self._enter_live(slug, st, grid_ns, rec, key,
                                   strat=strat, side=side, fill=float(fill), ttc=ttc)

    def _shares_for(self, strat: "LockedStrategy", ev: float) -> float:
        """Shares per entry: base + linear-in-EV boost when the lock defines
        sizing (per-slot overrides win), else the lock's fixed shares (5.1 for
        v1-era locks). base=0 reproduces pure linear_ev."""
        if self._sizing is None:
            return 5.1
        k = strat.size_k if strat.size_k is not None else self._sizing["k"]
        s_max = strat.size_max if strat.size_max is not None else self._sizing["s_max"]
        raw = self._sizing["base"] + k * max(0.0, ev - strat.ev_min)
        return min(s_max, raw) * self._sizing["size_scale"]

    async def _enter_live(self, slug: str, st, grid_ns: int, rec: dict,
                          key: tuple[str, str], *, strat: "LockedStrategy",
                          side: str, fill: float, ttc: float) -> None:
        """Order path for --live: RiskGate -> shared per-market notional cap ->
        FAK/IOC submit off-loop -> position tracking. The inherited reconcile
        loop resolves positions and feeds realized PnL back into the RiskGate
        daily-loss cap."""
        rec["router_mode"] = "live" if self.router.live else "dry_run"
        block, reason = self.risk_gate.should_block(now_ns=grid_ns)
        if block:
            rec["decision"] = "SKIP_RISK"
            rec["risk_reason"] = reason
            self._stats["skips"] += 1
            self._log_decision(grid_ns, rec)
            self.logger.warning("SKIP_RISK %s %s: %s", strat.strategy_id, slug, reason)
            return
        size_shares = (self._shares_for(strat, float(rec.get("ev") or 0.0))
                       if self._sizing is not None else LIVE_SHARES)
        new_notional = size_shares * fill
        cur_notional = self._market_notional_usd.get(slug, 0.0)
        if cur_notional + new_notional > MAX_MARKET_NOTIONAL_USD:
            rec["decision"] = "SKIP_EXPOSURE"
            rec["exposure_reason"] = (
                f"market_notional {cur_notional:.2f}+{new_notional:.2f}"
                f">{MAX_MARKET_NOTIONAL_USD:.2f}")
            self._stats["skips"] += 1
            self._log_decision(grid_ns, rec)
            return
        up_id, dn_id = self._asset_ids.get(slug, (None, None))
        token_id = up_id if side == "UP" else dn_id
        if token_id is None:
            rec["decision"] = "SKIP_EXEC"
            rec["exec_reason"] = "missing_token_id"
            self._stats["skips"] += 1
            self._log_decision(grid_ns, rec)
            return
        # crossing FAK/IOC in a thread so blocking HTTP never stalls the WS
        # consumers (see fair_value_bot._enter for the 2026-06-27 lesson)
        submit_start_ns = _now_ns()
        receipt = await asyncio.to_thread(
            self.router.submit_buy_gtc, token_id=token_id, limit_price=fill,
            size_shares=size_shares, side_label=side,
            buffer=LIVE_ORDER_BUFFER, fixed_shares=size_shares)
        rec["order_submit_roundtrip_s"] = round((_now_ns() - submit_start_ns) / NS, 6)
        rec["size_shares"] = round(size_shares, 4)
        rec["receipt"] = {
            "success": receipt.success, "dry_run": receipt.dry_run,
            "filled_size": receipt.filled_size, "avg_price": receipt.avg_price,
            "status": receipt.status, "order_id": receipt.order_id, "err": receipt.err,
        }
        if receipt.success and receipt.filled_size > 0:
            rec["decision"] = "ENTER"
            self._entered_strategy.add(key)
            self._entered.add(f"{slug}:{strat.strategy_id}")
            self._market_notional_usd[slug] = cur_notional + float(
                receipt.filled_size * receipt.avg_price)
            self._stats["enters"] += 1
            self._log_decision(grid_ns, rec)
            self.logger.info(
                "TCN ENTER(%s) %s %s [%s] fill=%.3f size=%.2f ttc=%.0fs",
                "LIVE" if self.router.live else "DRY", strat.strategy_id, slug,
                side, receipt.avg_price or fill, receipt.filled_size, ttc)
            pos = OpenPosition(
                market_slug=slug, down_token_id=token_id,
                shares=float(receipt.filled_size),
                entry_price=float(receipt.avg_price),
                notional_usd=float(receipt.filled_size * receipt.avg_price),
                fee_usd_estimated=0.072 * fill * (1 - fill) * receipt.filled_size,
                snapshot_ts_ns=grid_ns, market_close_ts_ns=st.close_s * NS,
                order_id=receipt.order_id, edge_dn=float(rec.get("ev") or 0.0),
                side=side, strike=float(st.strike))
            self.positions.add_position(pos)
            return
        rec["decision"] = "SKIP_EXEC"
        self._stats["skips"] += 1
        self._stats["exec_blocked"] = self._stats.get("exec_blocked", 0) + 1
        self._log_decision(grid_ns, rec)
        self.logger.warning("SKIP_EXEC %s %s [%s] status=%s err=%s",
                            strat.strategy_id, slug, side, receipt.status, receipt.err)

    async def _evaluate(self, slug: str, st, now_ns: int) -> None:
        slot = self._slot_for(st, now_ns)
        if slot is None:
            return
        _step, grid_ns = slot
        self._flush_source_replay_until(grid_ns)
        await self._evaluate_grid(slug, st, grid_ns)

    def _write(self, now_ns: int, rec: dict) -> None:
        date_iso = datetime.fromtimestamp(now_ns / 1e9, UTC).date().isoformat()
        path = self.decisions_dir / f"tcn_decisions_{date_iso}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    async def _tick_loop(self) -> None:
        tick_s = float(os.getenv("TCN_TICK_S", os.getenv("FV_TICK_S", "0.2")))
        max_catchup_steps = max(1, int(os.getenv("TCN_MAX_CATCHUP_STEPS", "50")))
        while not self._shutdown.is_set():
            now = _now_ns()
            replay_now = max(0, now - SOURCE_REPLAY_SETTLE_NS)
            target_grid_ns = (replay_now // STEP_NS) * STEP_NS if STEP_NS > 0 else replay_now
            if self._last_replay_grid_ns is None:
                self._last_replay_grid_ns = target_grid_ns - STEP_NS
            next_grid_ns = int(self._last_replay_grid_ns) + STEP_NS
            min_grid_ns = target_grid_ns - (max_catchup_steps - 1) * STEP_NS
            if next_grid_ns < min_grid_ns:
                dropped = int((min_grid_ns - next_grid_ns) // STEP_NS)
                self._stats["tcn_scheduler_dropped_steps"] = (
                    self._stats.get("tcn_scheduler_dropped_steps", 0) + max(0, dropped)
                )
                next_grid_ns = min_grid_ns
                self._last_replay_grid_ns = next_grid_ns - STEP_NS
            self._flush_pm_price_change_until(now)
            self._flush_cb_ticker_until(now)
            self._flush_rtds(now)
            while next_grid_ns <= target_grid_ns:
                grid_ns = int(next_grid_ns)
                self._flush_source_replay_until(grid_ns)
                for slug, st in list(self._markets.items()):
                    ttc_now = (st.close_s * NS - now) / NS
                    if ttc_now < self._min_ttc - 5:
                        self._markets.pop(slug, None)
                        self._episode_buffers.pop(slug, None)
                        self._replay_markets.pop(slug, None)
                        self._replay_pm_recv_watermark_by_slug.pop(slug, None)
                        self._replay_pm_source_watermark_by_slug.pop(slug, None)
                        self._pm_pc_pending_by_slug.pop(slug, None)
                        self._exec_retry_after_by_slug.pop(slug, None)
                        self._resync_armed.discard(slug)
                        self._last_eval_step_by_slug.pop(slug, None)
                        self._expired.add(slug)
                        if self.poly_ws is not None:
                            try:
                                self.poly_ws.remove_market(slug)
                            except Exception:
                                pass
                        continue
                    if st.strike <= 0.0 and grid_ns >= st.open_s * NS:
                        px = self._rtds_price_at(st.open_s * NS)
                        if px is not None:
                            st.strike = float(px)
                            replay_st = self._ensure_replay_market(slug)
                            if replay_st is not None:
                                replay_st.strike = float(px)
                            self.logger.info("strike from RTDS@open for %s: %.2f", slug, px)
                    if (
                        slug not in self._resync_armed
                        and self._max_ttc < ttc_now <= self._max_ttc + RESYNC_LEAD_S
                    ):
                        self._resync_armed.add(slug)
                        if self.poly_ws is not None:
                            try:
                                self.poly_ws.request_resync()
                                self.logger.info("pre-window HARD resync armed for %s (ttc=%.0fs)", slug, ttc_now)
                            except Exception as exc:
                                self.logger.debug("resync request err %s: %r", slug, exc)
                    try:
                        replay_st = self._ensure_replay_market(slug)
                        if replay_st is not None:
                            await self._evaluate_grid(slug, replay_st, grid_ns)
                    except Exception as exc:
                        self.logger.exception("TCN evaluate err %s: %r", slug, exc)
                self._last_replay_grid_ns = grid_ns
                next_grid_ns += STEP_NS
            self._stats["ticks"] += 1
            await asyncio.sleep(tick_s)

    async def run(self) -> None:
        self.logger.info("=== TCN SHADOW BOT (orders disabled) ===")
        self.logger.info("lock=%s", self._lock_path)
        for s in self._strategies:
            self.logger.info(
                "strategy %s: ttc(%.0f,%.0f] beta=%.3f EV>=%.3f px(%.2f,%.2f) utc=%s",
                s.strategy_id, s.ttc_min, s.ttc_max, s.beta, s.ev_min,
                s.price_lo, s.price_hi,
                sorted(s.utc_hours) if s.utc_hours is not None else "all",
            )
        self.logger.info("RTDS source-time availability latency: %.1fs", RTDS_FEATURE_LAG_S)
        self.logger.info("TCN source-time replay settle: %.0fms", SOURCE_REPLAY_SETTLE_NS / 1_000_000)
        self.logger.info("bucket cadence: pm=%dns cex=%dns model=%dns", PM_PRICE_CHANGE_BUCKET_NS,
                         CEX_TICKER_BUCKET_NS, STEP_NS)
        self.logger.info("paths: raw_root=%s decisions_dir=%s", self.raw_root, self.decisions_dir)
        need_recorder_features = self._rtds_source == "recorder" or self._cex_source == "recorder"
        tail = MultiSourceTail(
            raw_root=self.raw_root,
            poll_interval_s=float(os.getenv("FV_TAIL_POLL_S", "1.0")),
            logger=self.logger,
            fair_value_sources=need_recorder_features,
            registration_only=(self._pm_book_source == "direct" and not need_recorder_features),
            polymarket_only=(self._pm_book_source == "recorder" and not need_recorder_features),
        )
        self.logger.info("data plane: PM book=%s, RTDS=%s, Coinbase=%s",
                         self._pm_book_source, self._rtds_source, self._cex_source)
        tasks = [
            asyncio.create_task(self._ingest_loop(tail), name="registration"),
            asyncio.create_task(self._tick_loop(), name="tick"),
            asyncio.create_task(self._stats_loop(), name="stats"),
        ]
        if self._rtds_source == "direct":
            tasks.append(asyncio.create_task(self.rtds_ws.run(), name="rtds_ws"))
        if self._cex_source == "direct":
            tasks.append(asyncio.create_task(self.coinbase_ws.run(), name="coinbase_ws"))
        if self.poly_ws is not None:
            tasks.append(asyncio.create_task(self.poly_ws.run(), name="poly_ws"))
        try:
            await self._shutdown.wait()
        finally:
            if self.poly_ws is not None:
                self.poly_ws.stop()
            self.rtds_ws.stop()
            self.coinbase_ws.stop()
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            tail.close()
            self._close_direct_capture()
            self.logger.info("=== TCN SHADOW BOT STOPPED ===")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default="data", type=Path)
    ap.add_argument("--decisions-dir", default="logs/tcn_shadow_bot", type=Path)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--live", action="store_true",
                    help="enable the order path (RiskGate + exposure cap + router); "
                         "real orders additionally require ENABLE_REAL_ORDERS=1 + creds")
    args = ap.parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))
    bot = TCNShadowBot(raw_root=args.raw_root, decisions_dir=args.decisions_dir,
                       live_orders=args.live)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sname in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sname):
            try:
                loop.add_signal_handler(getattr(signal, sname), bot.trigger_shutdown)
            except (NotImplementedError, RuntimeError):
                pass
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        bot.trigger_shutdown()
        loop.run_until_complete(asyncio.sleep(0.3))
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
