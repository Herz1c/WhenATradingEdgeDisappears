"""Live feature runtime — incrementally builds per-market indices from
the tailed event stream and fires snapshot computations at the
dense_close grid times.

Pipeline:

  tailed L2 events -----> streaming_engine.feed_l2(record)
  tailed REST events ---> streaming_engine.feed_rest(record)
  per-second BTC stat --> LiveReferenceIndex (refreshed every ~1s)

  every ~100ms we call engine.tick(now_ns) which fires any snapshot
  whose grid time has passed. Each emitted row goes through:

      clean_features -> model.predict_proba -> decide() -> on_decision()

The runtime owns:
  - one StreamingFeatureEngine per UTC day (rolled at midnight)
  - the RiskState
  - the model + features list (loaded once at startup)

Hard contract: NEVER fires a snapshot for a market with ttc > 60 s.
The streaming_engine's grid already starts at T-60s, but we enforce
the gate again here as defense-in-depth (matches the "do not trade
the first 240 s" rule in LiveTradingBotPlan.md §1a).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from feature_cleanup import clean_features
from live.streaming_engine import StreamingFeatureEngine
from live_bot.btc_state import LiveBtcState
from live_bot.decision_engine import RiskState, TTC_MIN_S, TTC_MAX_S, decide
from polymarket_recorder.dataset_factory import MarketLabel, SECOND_NS


# Model selection. LIVE_BOT_MODEL = "no_bias" (default) uses the new
# bias-free calibrated model_no_bias_v1; "legacy" falls back to the original
# bias-corrected model_02 (still on disk for emergency rollback).
_MODEL_KEY = os.environ.get("LIVE_BOT_MODEL", "no_bias").strip().lower()
if _MODEL_KEY == "legacy":
    ART = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")
    _MODEL_REQUIRES_CALIBRATOR = False
    _FEATURE_LIST_SOURCE = "feature_importance.json"  # legacy file order
else:
    ART = Path("artifacts_cleaned/model_no_bias_v1")
    _MODEL_REQUIRES_CALIBRATOR = True
    _FEATURE_LIST_SOURCE = "experiment_manifest.json"  # carries true train order


def _load_model():
    if os.environ.get("FEATURE_CLEANUP_ENABLED") != "1":
        os.environ["FEATURE_CLEANUP_ENABLED"] = "1"
    model = joblib.load(ART / "model.pkl")
    calibrator = getattr(model, "_calibrator", None)
    if _MODEL_REQUIRES_CALIBRATOR and calibrator is None:
        raise RuntimeError(
            f"Model at {ART} is expected to have a calibrator attached "
            f"(_calibrator attribute). Re-train via tools/train_no_bias_model.py.")
    if (not _MODEL_REQUIRES_CALIBRATOR) and calibrator is not None:
        # Legacy contract: deployed model must NOT have a calibrator
        raise RuntimeError("Legacy model must have _calibrator stripped (LiveTradingBotPlan rule #1)")
    if _FEATURE_LIST_SOURCE == "experiment_manifest.json":
        feats = json.loads((ART / "experiment_manifest.json").read_text())["features"]
    else:
        feats = list(json.loads((ART / "feature_importance.json").read_text()).keys())
    return model, feats


def _build_X(row: dict, feats: list[str]) -> np.ndarray:
    # Wrap single row dict into a polars DataFrame for clean_features
    df = pl.DataFrame([row])
    df = clean_features(df)
    X = np.zeros((1, len(feats)), dtype=np.float32)
    for i, f in enumerate(feats):
        if f in df.columns:
            v = df.get_column(f)
            if v.dtype.is_numeric():
                arr = v.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
                if arr.size:
                    val = float(arr[0])
                    if np.isfinite(val):
                        X[0, i] = val
    return X


class FeatureRuntime:
    """Bridges the file-tail stream to the streaming engine + model + decision."""

    def __init__(self, *, btc: LiveBtcState, risk: RiskState,
                 on_decision, logger: logging.Logger | None = None) -> None:
        self.btc = btc
        self.risk = risk
        self.logger = logger or logging.getLogger("feature_runtime")
        self.on_decision = on_decision
        self._engine: StreamingFeatureEngine | None = None
        self._engine_day: date | None = None
        self.model, self.feats = _load_model()
        self._last_tick_ns = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def ensure_engine(self, now_ns: int) -> StreamingFeatureEngine:
        today = datetime.fromtimestamp(now_ns / 1e9, UTC).date()
        if self._engine is None or today != self._engine_day:
            self.logger.info("rolling streaming engine for new day %s", today)
            self._engine = StreamingFeatureEngine(live_reference=self.btc.to_live_reference_index())
            self._engine_day = today
        return self._engine

    def refresh_live_reference(self) -> None:
        if self._engine is None:
            return
        # Hot-swap the live_reference index — append-only so cheap.
        self._engine.live_reference = self.btc.to_live_reference_index()

    def register_market(self, label: MarketLabel) -> None:
        eng = self.ensure_engine(label.market_open_ts_ns)
        eng.register_market(label)

    # ── ingest ───────────────────────────────────────────────────────────────
    # Skip events older than this — they describe markets that already
    # resolved hours ago and would never become candidates for ENTER.
    INGEST_MAX_AGE_S = 600.0

    def _is_recent(self, recv: int) -> bool:
        import time as _time
        now_ns = int(_time.time() * 1e9)
        return (now_ns - recv) <= int(self.INGEST_MAX_AGE_S * SECOND_NS)

    def feed_l2(self, record: dict) -> None:
        recv = int(record.get("recv_ts_ns") or 0)
        if recv <= 0 or not self._is_recent(recv): return
        eng = self.ensure_engine(recv)
        eng.feed_l2(record)
        # Track the REAL DOWN-token best-ask per market (used only by the
        # sub-15c price gate in decision_engine; NOT a model feature, so it
        # does not affect parity with the training/test datasets).
        if str(record.get("token_outcome")) == "Down":
            slug = record.get("selected_market_slug")
            ba = record.get("best_ask")
            if slug and ba is not None:
                if not hasattr(self, "_down_ask_by_slug"):
                    self._down_ask_by_slug = {}
                try:
                    self._down_ask_by_slug[str(slug)] = (float(ba), recv)
                except (TypeError, ValueError):
                    pass

    def feed_rest(self, record: dict) -> None:
        recv = int(record.get("recv_ts_ns") or 0)
        if recv <= 0 or not self._is_recent(recv): return
        eng = self.ensure_engine(recv)
        eng.feed_rest(record)

    # ── tick ─────────────────────────────────────────────────────────────────
    def tick(self, now_ns: int) -> None:
        if self._engine is None:
            return
        # Refresh BTC indices ~once per second
        if now_ns - self._last_tick_ns > SECOND_NS:
            # Canonical-parquet bias is only used as a fallback ONCE -- if the
            # in-process _live_bias_loop hasn't populated _last_bias yet
            # (e.g. it's still computing on startup). After the live loop has
            # written a bias, we stop reading the parquet so we don't overwrite
            # the fresh in-process value with a stale on-disk one.
            # SKIPPED entirely for the no-bias model (calibrated v1).
            if (not _MODEL_REQUIRES_CALIBRATOR) and self.btc._last_bias_recv_ns == 0:
                try:
                    if self.btc.reload_canonical_bias():
                        self.logger.info("canonical bias (startup fallback): %+.2f (from %s)",
                                         self.btc._last_bias,
                                         datetime.fromtimestamp(self.btc._last_bias_recv_ns/1e9, UTC).isoformat())
                except Exception as exc:
                    self.logger.warning("canonical bias reload failed: %r", exc)
            self.refresh_live_reference()
            self._last_tick_ns = now_ns

        for row in self._engine.tick(now_ns):
            self._handle_snapshot(row, now_ns)

    def _handle_snapshot(self, row: dict, now_ns: int) -> None:
        # Skip snapshots for markets that have ALREADY closed in wall-clock
        # time. Polymarket pulls all asks the moment a market resolves, so
        # an order against a closed market either errors with
        # "orderbook does not exist" or "no orders found to match" because
        # the book is empty. Either way, the order is dead on arrival.
        # NOTE: we do NOT block stale snapshots (that was the over-aggressive
        # guard that previously prevented trading new markets); only
        # already-resolved ones.
        market_close_ns = int(row.get("market_close_ts_ns", 0))
        if market_close_ns > 0 and now_ns >= market_close_ns:
            return

        ttc = float(row.get("t_to_close_s", -1))
        # Defense-in-depth: never trade outside the band
        if ttc < TTC_MIN_S or ttc > TTC_MAX_S:
            return
        # Guard: do not run inference on an empty book — Polymarket
        # L2 events for this market may not have arrived yet.
        up_bid = row.get("up_token_best_bid")
        up_ask = row.get("up_token_best_ask")
        import math as _m
        def _bad(x): return x is None or (isinstance(x, float) and not _m.isfinite(x))
        if _bad(up_bid) or _bad(up_ask):
            return
        self.risk.release_closed(now_ns)
        # Inject the no-bias model's raw features (computed from
        # btc_state.history_binance_spot, not from synthetic_corrected).
        # On the legacy model this is harmless -- those columns just aren't
        # in the model's feature list, so _build_X ignores them.
        try:
            raw = self.btc.compute_raw_features(strike=row.get("price_to_beat"))
            for k, v in raw.items():
                row[k] = v
        except Exception as exc:
            self.logger.warning("compute_raw_features failed: %r", exc)
        try:
            X = _build_X(row, self.feats)
            raw_p_up = float(self.model.predict_proba(X)[0, 1])
            # Apply the calibrator if the deployed model has one. The new
            # no-bias model requires this; legacy is calibrator-stripped.
            cal = getattr(self.model, "_calibrator", None)
            if cal is not None:
                p_up = float(np.clip(cal.predict([raw_p_up])[0], 1e-6, 1 - 1e-6))
            else:
                p_up = raw_p_up
        except Exception as exc:
            self.logger.warning("model inference failed: %r", exc)
            return
        # Real DOWN best-ask for this market, if seen on the L2 feed within 30s.
        slug = str(row["market_slug"])
        down_ask = None
        da = getattr(self, "_down_ask_by_slug", {}).get(slug)
        if da is not None and (now_ns - da[1]) <= 30 * SECOND_NS:
            down_ask = da[0]
        snap = {
            "snapshot_ts_ns":     int(row["snapshot_ts_ns"]),
            "market_slug":        slug,
            "up_token_best_bid":  row.get("up_token_best_bid"),
            "up_token_best_ask":  row.get("up_token_best_ask"),
            "down_token_best_ask": down_ask,
            "t_to_close_s":       ttc,
            # Needed by decision_engine for the raw delta-to-strike guard
            # (refined-rule filter #2). Both are already present in `row`
            # from the streaming engine; we just expose them on the snap
            # dict so decide() can read them without poking the engine.
            "binance_spot_mid":   row.get("binance_spot_mid"),
            "price_to_beat":      row.get("price_to_beat"),
        }
        # Compute bias age (wall-clock seconds since the canonical
        # second the bias was derived from). None if never loaded.
        # For the no-bias model the bias is irrelevant -- short-circuit to 0
        # so the freshness gate never blocks. For legacy, keep current behavior.
        if _MODEL_REQUIRES_CALIBRATOR:
            bias_age: float | None = 0.0
        else:
            bias_age = None
            if self.btc._last_bias_recv_ns > 0:
                bias_age = (now_ns - self.btc._last_bias_recv_ns) / SECOND_NS
        decision = decide(snap=snap, p_up=p_up, state=self.risk,
                          bias_age_seconds=bias_age)
        decision["snap"] = snap
        decision["market_close_ts_ns"] = int(row.get("market_close_ts_ns", snap["snapshot_ts_ns"] + 60 * SECOND_NS))
        self.on_decision(decision)
