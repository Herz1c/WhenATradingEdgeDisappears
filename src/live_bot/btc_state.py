"""Live synthetic BTC mid for the bot — simplified version of the
canonical chainlink_recorder live_reference pipeline.

NOTE: events older than `INGEST_MAX_AGE_S` of wall-clock are dropped
at ingest. The bot only needs the most recent few minutes of price
history to build features; replaying half a day of stale events on
startup wastes CPU and (for the chainlink residual) produces a
nonsensical bias.

The model was trained on `synthetic_corrected`, the per-second
bias-corrected aggregate of binance_spot + binance_usdm + hyperliquid +
chainlink_public_delayed. The full offline pipeline applies a residual
bias correction to land within ±$5 of Chainlink — too heavyweight to
reproduce live in v1.

This module approximates `synthetic_corrected` as the **median of
available source mids** at each second. Empirically that's within
$5-15 of the calibrated value, which is a known accuracy haircut we
accept for v1 (documented in LIVE_BOT_README.md).

Outputs:
  - `current_mid()` -> latest live BTC mid (float | None)
  - `live_reference_index()` -> a LiveReferenceIndex covering the rolling
    last N hours, suitable for dataset_factory._build_snapshot_row().
"""
from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Import the canonical index dataclass so feature builders see the same shape
from polymarket_recorder.dataset_factory import LiveReferenceIndex, SECOND_NS


INGEST_MAX_AGE_S = 300.0   # drop any event older than 5 minutes of wall-clock


def _is_recent(recv_ns: int) -> bool:
    """True iff the event was received within INGEST_MAX_AGE_S of now."""
    now_ns = int(time.time() * 1e9)
    return (now_ns - recv_ns) <= int(INGEST_MAX_AGE_S * SECOND_NS)


@dataclass
class _SourceState:
    name: str
    last_mid: float | None = None
    last_recv_ns: int = 0


@dataclass
class LiveBtcState:
    """Tracks the most recent bid/ask from each source and produces a
    per-second synthetic price, bias-corrected against the canonical
    `live_reference_events_v1` parquet (rebuilt periodically by
    tools/canonical_daemon.py).

    Bias mechanism (matches what dataset_factory uses in backtests):

      1. The canonical builder produces `synthetic_corrected` per second
         (bias-corrected against chainlink_public_delayed historically).
      2. The bot reads the LATEST canonical row periodically:
              bias = synthetic_corrected − median(exchange_mids_at_same_second)
      3. The bot's runtime synthetic per second:
              synthetic = current_exchange_median + cached_bias

    This guarantees the bot's BTC feature is in the same distribution
    the model was trained on — which CANNOT be achieved with our own
    chainlink_onchain-derived bias (we lived this lesson on 2026-05-26)."""

    binance_spot: _SourceState = field(default_factory=lambda: _SourceState("binance_spot"))
    binance_usdm: _SourceState = field(default_factory=lambda: _SourceState("binance_usdm"))
    hyperliquid:  _SourceState = field(default_factory=lambda: _SourceState("hyperliquid"))

    # Kept for instrumentation only (sanity check vs canonical bias)
    chainlink_onchain: _SourceState = field(default_factory=lambda: _SourceState("chainlink_onchain"))

    # Canonical-derived bias state
    bias_max_age_seconds: float = 3600.0
    _last_bias: float = 0.0
    _last_bias_recv_ns: int = 0
    # Path is set by the orchestrator at startup based on UTC date
    canonical_parquet_path: Optional[object] = None     # Path or None
    _canonical_last_mtime: float = 0.0
    _canonical_load_errors: int = 0

    # rolling per-second history (we keep up to ~4 h = 14400 entries)
    history_seconds: deque[int]                  = field(default_factory=lambda: deque(maxlen=14_400))
    history_synthetic: deque[float]              = field(default_factory=lambda: deque(maxlen=14_400))
    history_price_valid: deque[bool]             = field(default_factory=lambda: deque(maxlen=14_400))
    history_source_count: deque[int]             = field(default_factory=lambda: deque(maxlen=14_400))
    history_max_spread: deque[float]             = field(default_factory=lambda: deque(maxlen=14_400))
    history_binance_spot: deque[float]           = field(default_factory=lambda: deque(maxlen=14_400))
    history_binance_usdm: deque[float]           = field(default_factory=lambda: deque(maxlen=14_400))
    history_hyperliquid:  deque[float]           = field(default_factory=lambda: deque(maxlen=14_400))

    _last_bucket_second: int = 0

    # ── ingest ───────────────────────────────────────────────────────────────
    def feed_binance_spot(self, record: dict) -> None:
        self._ingest(self.binance_spot, record, kind="bookTicker")

    def feed_binance_usdm(self, record: dict) -> None:
        self._ingest(self.binance_usdm, record, kind="bookTicker")

    def feed_hyperliquid(self, record: dict) -> None:
        self._ingest(self.hyperliquid, record, kind="hl")

    def feed_chainlink_onchain(self, record: dict) -> None:
        """Kept ONLY for instrumentation — we track the chainlink_onchain
        price for sanity checks but do NOT use it for bias correction.
        The bias is derived from the canonical live_reference_events_v1
        parquet (see reload_canonical_bias) which uses
        chainlink_public_delayed, matching what the model was trained on."""
        recv_ns = int(record.get("recv_ts_ns") or 0)
        if recv_ns <= 0 or not _is_recent(recv_ns): return
        rd = record.get("round_data") or {}
        price: float | None = None
        if rd.get("answer") is not None:
            try: price = float(rd["answer"])
            except (TypeError, ValueError): price = None
        if price is None and rd.get("answer_raw") is not None:
            try: price = float(int(rd["answer_raw"])) / 1e8
            except (TypeError, ValueError): price = None
        if price is None or not math.isfinite(price) or price <= 0:
            return
        self.chainlink_onchain.last_mid = price
        self.chainlink_onchain.last_recv_ns = recv_ns

    def reload_canonical_bias(self, *, force: bool = False) -> bool:
        """Read the latest valid row from the canonical parquet and extract
        bias = synthetic_corrected − median(exchange mids at same second).

        Falls back to the most recent past-date parquet (up to 7 days back)
        if today's doesn't exist yet. Returns True if bias was refreshed.
        Safe to call on every tick — skips work when the parquet's mtime
        hasn't changed unless `force=True`."""
        candidate = self._resolve_freshest_canonical()
        if candidate is None:
            return False
        path, mtime = candidate
        if not force and mtime <= self._canonical_last_mtime and path == getattr(self, "_canonical_last_path", None):
            return False
        try:
            import polars as pl
            df = pl.read_parquet(
                path,
                columns=["ts_seconds", "synthetic_corrected", "price_valid",
                         "binance_spot_mid", "binance_usdm_mid", "hyperliquid_mid"],
            )
            df = df.filter(pl.col("price_valid"))
            if len(df) == 0:
                return False
            last = df.tail(1).row(0, named=True)
        except Exception:
            self._canonical_load_errors += 1
            return False

        synthetic = last.get("synthetic_corrected")
        if synthetic is None or not math.isfinite(float(synthetic)):
            return False
        synthetic = float(synthetic)

        exch_mids = []
        for k in ("binance_spot_mid", "binance_usdm_mid", "hyperliquid_mid"):
            v = last.get(k)
            try:
                f = float(v)
                if math.isfinite(f) and f > 0:
                    exch_mids.append(f)
            except (TypeError, ValueError):
                pass
        if not exch_mids:
            return False
        canonical_exch_med = statistics.median(exch_mids)

        self._last_bias = synthetic - canonical_exch_med
        self._last_bias_recv_ns = int(last["ts_seconds"]) * SECOND_NS
        self._canonical_last_mtime = mtime
        self._canonical_last_path = path
        return True

    def reload_live_bias(self, raw_root, policy_path=None, *,
                         window_hours: float = 6.0,
                         max_workers: int = 4,
                         logger=None) -> dict | None:
        """Compute the bias fresh from recent recorder data using
        live_bot.live_bias.compute_live_bias_value (bit-identical to the
        canonical for the latest finalized second). On success, overwrites
        _last_bias / _last_bias_recv_ns. On failure or 'unavailable' result,
        leaves them untouched so the canonical-parquet-derived state remains.

        This is the path that lets the bot work as a single self-contained
        process (recorders -> bot) without anyone running compute_bias.ps1.

        Returns the full compute_live_bias_value result dict on success (so
        the caller can persist it for later canonical-comparison auditing),
        or None on any failure / no-usable-value path.
        """
        from live_bot.live_bias import compute_live_bias_value
        try:
            result = compute_live_bias_value(
                root=raw_root, policy_path=policy_path,
                window_hours=window_hours, max_workers=max_workers,
                include_active_hour=True,
            )
        except Exception as exc:
            if logger is not None:
                logger.warning("reload_live_bias compute failed: %r", exc)
            self._canonical_load_errors += 1
            return None
        if result is None:
            if logger is not None:
                logger.info("reload_live_bias: no eligible data in window")
            return None
        bot_bias = result.get("bot_extracted_bias")
        if bot_bias is None or not math.isfinite(float(bot_bias)):
            if logger is not None:
                logger.info("reload_live_bias: bias_mode=%s ts=%s "
                            "(no usable value to apply; keeping prior bias %+.2f)",
                            result.get("bias_mode"), result.get("ts"),
                            self._last_bias)
            return None
        self._last_bias = float(bot_bias)
        self._last_bias_recv_ns = int(result["ts"]) * SECOND_NS
        if logger is not None:
            logger.info("reload_live_bias: bias=%+.3f mode=%s active=%s "
                        "value_age=%.0fs elapsed=%.1fs",
                        self._last_bias, result.get("bias_mode"),
                        result.get("bias_active"),
                        float(result.get("value_age_seconds") or 0.0),
                        float(result.get("elapsed_seconds") or 0.0))
        return result

    # ── RAW features for the no-bias model ───────────────────────────────
    # The new no-bias model (model_no_bias_v1) takes raw-binance-derived
    # features instead of synthetic_corrected ones. The history_binance_spot
    # / history_seconds deques already accumulate per-second raw binance
    # mids -- this method computes the model's expected raw features from
    # them on demand, matching what tools/augment_dense_close_no_bias.py
    # produced for the training data.
    def compute_raw_features(self, *, strike: float | None,
                             now_seconds: int | None = None) -> dict:
        """Return a dict of raw features for the no-bias model:
           - delta_to_strike_raw, abs/sign/over_vol variants
           - btc_return_{1,3,5,10}s_raw      (from per-second history)
           - btc_realized_vol_{5,15}s_raw    (std of 1-s log returns)
           - time_spent_above/below_strike_recent_s_raw (60s window)
        Missing data -> NaN; clean_features will fill_null downstream.
        """
        out: dict = {}
        # Bring the deques into local lists once for indexed access
        ts_hist = list(self.history_seconds)
        px_hist = list(self.history_binance_spot)
        if not ts_hist or not px_hist or len(ts_hist) != len(px_hist):
            return out
        latest_px = px_hist[-1] if px_hist else None
        if latest_px is None or not math.isfinite(latest_px):
            return out

        # delta_to_strike_raw and variants
        if strike is not None and math.isfinite(float(strike)) and float(strike) > 0:
            delta = float(latest_px) - float(strike)
            out["delta_to_strike_raw"] = delta
            out["abs_delta_to_strike_raw"] = abs(delta)
            out["delta_sign_raw"] = 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)

        # Returns over horizons: (px_now - px_t_back) / px_t_back
        def ret(horizon_s: int) -> float:
            if len(px_hist) <= horizon_s: return math.nan
            cur = px_hist[-1]; prev = px_hist[-1 - horizon_s]
            if not (math.isfinite(cur) and math.isfinite(prev)) or prev == 0:
                return math.nan
            return (cur - prev) / prev

        out["btc_return_1s_raw"]  = ret(1)
        out["btc_return_3s_raw"]  = ret(3)
        out["btc_return_5s_raw"]  = ret(5)
        out["btc_return_10s_raw"] = ret(10)

        # Realized vol: std of 1-s log returns over the last N intervals
        def vol(window: int) -> float:
            if len(px_hist) <= window: return math.nan
            seg = px_hist[-(window + 1):]
            log_rets: list[float] = []
            for j in range(1, len(seg)):
                a, b = seg[j-1], seg[j]
                if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
                    continue
                log_rets.append(math.log(b / a))
            if len(log_rets) < 2: return math.nan
            mean = sum(log_rets) / len(log_rets)
            var = sum((x - mean)**2 for x in log_rets) / (len(log_rets) - 1)
            return math.sqrt(var)
        v5 = vol(5)
        v15 = vol(15)
        out["btc_realized_vol_5s_raw"]  = v5
        out["btc_realized_vol_15s_raw"] = v15
        # delta / vol -- guard against tiny vol
        if "delta_to_strike_raw" in out and math.isfinite(v15) and v15 > 0:
            out["delta_to_strike_over_vol_raw"] = out["delta_to_strike_raw"] / max(v15, 1e-9)
        else:
            out["delta_to_strike_over_vol_raw"] = math.nan

        # Time spent above/below strike (last 60 seconds)
        if strike is not None and math.isfinite(float(strike)) and float(strike) > 0:
            seg = px_hist[-60:] if len(px_hist) > 60 else px_hist
            above = sum(1 for p in seg if math.isfinite(p) and p > float(strike))
            below = sum(1 for p in seg if math.isfinite(p) and p < float(strike))
            out["time_spent_above_strike_recent_s_raw"] = float(above)
            out["time_spent_below_strike_recent_s_raw"] = float(below)
        return out

    def _resolve_freshest_canonical(self) -> Optional[tuple]:
        """Return (path, mtime) of the most recent canonical parquet to use.
        Tries today's first, then walks back up to 7 days for a fallback."""
        path = self.canonical_parquet_path
        if path is None:
            return None
        candidates: list[tuple] = []
        # today's path
        if path.exists():
            try: candidates.append((path, path.stat().st_mtime))
            except OSError: pass
        # walk yesterday back to 7 days back if today's is missing
        if not candidates:
            from datetime import datetime, timedelta, UTC as _UTC
            for delta in range(1, 8):
                d = (datetime.now(_UTC).date() - timedelta(days=delta)).isoformat()
                p = path.with_name(f"{d}.parquet")
                if p.exists():
                    try:
                        candidates.append((p, p.stat().st_mtime))
                        break
                    except OSError:
                        pass
        return candidates[0] if candidates else None

    def _ingest(self, src: _SourceState, record: dict, *, kind: str) -> None:
        recv_ns = int(record.get("recv_ts_ns") or 0)
        if recv_ns <= 0:
            return
        if not _is_recent(recv_ns):
            return
        mid: float | None = None
        if kind == "bookTicker":
            if str(record.get("event_type") or "") != "bookTicker":
                return
            pay = record.get("payload") or {}
            try:
                b = float(pay["b"]); a = float(pay["a"])
                mid = (b + a) / 2.0
            except (KeyError, TypeError, ValueError):
                return
        else:  # hl
            # Hyperliquid l2Book / bbo event — payload schema varies; try a few
            pay = record.get("payload") or {}
            for path in (
                ("levels", 0, 0, "px"),  # l2Book bid level 0
                ("bbo", 0, "px"),
                ("mid_px",),
            ):
                v = pay
                ok = True
                for k in path:
                    try:
                        v = v[k] if not isinstance(k, int) else v[k]
                    except (KeyError, IndexError, TypeError):
                        ok = False; break
                if ok and v is not None:
                    try:
                        mid = float(v); break
                    except (TypeError, ValueError):
                        pass
        if mid is None or not math.isfinite(mid) or mid <= 0:
            return
        src.last_mid = mid
        src.last_recv_ns = recv_ns
        self._maybe_emit(recv_ns)

    def _maybe_emit(self, now_ns: int) -> None:
        sec = now_ns // SECOND_NS
        if sec <= self._last_bucket_second:
            return
        # Fill any skipped seconds with the prior value (forward-fill, as the
        # canonical builder does when no fresh tick arrives in a given second).
        prev_synthetic = self.history_synthetic[-1] if self.history_synthetic else math.nan
        prev_valid     = self.history_price_valid[-1] if self.history_price_valid else False
        prev_count     = self.history_source_count[-1] if self.history_source_count else 0
        prev_spread    = self.history_max_spread[-1] if self.history_max_spread else math.nan
        prev_bs        = self.history_binance_spot[-1] if self.history_binance_spot else math.nan
        prev_bu        = self.history_binance_usdm[-1] if self.history_binance_usdm else math.nan
        prev_hl        = self.history_hyperliquid[-1]  if self.history_hyperliquid else math.nan

        start = self._last_bucket_second + 1 if self._last_bucket_second else sec
        for s in range(start, sec + 1):
            # 5-second freshness cutoff per source (matches canonical default)
            cutoff_ns = s * SECOND_NS - 5 * SECOND_NS
            mids: list[float] = []
            bs = bu = hl = math.nan
            if self.binance_spot.last_recv_ns >= cutoff_ns and self.binance_spot.last_mid is not None:
                mids.append(self.binance_spot.last_mid); bs = self.binance_spot.last_mid
            if self.binance_usdm.last_recv_ns >= cutoff_ns and self.binance_usdm.last_mid is not None:
                mids.append(self.binance_usdm.last_mid); bu = self.binance_usdm.last_mid
            if self.hyperliquid.last_recv_ns  >= cutoff_ns and self.hyperliquid.last_mid is not None:
                mids.append(self.hyperliquid.last_mid);  hl = self.hyperliquid.last_mid

            if mids:
                exch_median = statistics.median(mids)
                # Carry bias forward: use the bias captured at the last
                # chainlink update IF it's not too stale.
                bias_age_s = (s * SECOND_NS - self._last_bias_recv_ns) / SECOND_NS
                if self._last_bias_recv_ns > 0 and bias_age_s <= self.bias_max_age_seconds:
                    synthetic = exch_median + self._last_bias
                else:
                    synthetic = exch_median
                max_spread = max(mids) - min(mids) if len(mids) >= 2 else 0.0
                self.history_seconds.append(s)
                self.history_synthetic.append(synthetic)
                self.history_price_valid.append(True)
                self.history_source_count.append(len(mids))
                self.history_max_spread.append(max_spread)
                self.history_binance_spot.append(bs)
                self.history_binance_usdm.append(bu)
                self.history_hyperliquid.append(hl)
            else:
                # Forward-fill but mark invalid
                self.history_seconds.append(s)
                self.history_synthetic.append(prev_synthetic)
                self.history_price_valid.append(False)
                self.history_source_count.append(0)
                self.history_max_spread.append(math.nan)
                self.history_binance_spot.append(prev_bs)
                self.history_binance_usdm.append(prev_bu)
                self.history_hyperliquid.append(prev_hl)
        self._last_bucket_second = sec

    # ── output ───────────────────────────────────────────────────────────────
    def current_mid(self) -> float | None:
        if not self.history_synthetic:
            return None
        last = self.history_synthetic[-1]
        if not math.isfinite(last):
            return None
        return float(last)

    def to_live_reference_index(self) -> LiveReferenceIndex:
        """Build a LiveReferenceIndex of the in-memory history. Bias-related
        fields are filled with conservative defaults — model was trained
        with FEATURE_CLEANUP_ENABLED=1 which transforms
        `live_applied_bias_age_seconds` to a clipped log, so a constant
        sentinel (`MAX_BIAS_AGE_S`) collapses to the same model input."""
        n = len(self.history_seconds)
        return LiveReferenceIndex(
            ts_seconds              = list(self.history_seconds),
            synthetic_corrected     = list(self.history_synthetic),
            price_valid             = list(self.history_price_valid),
            degraded                = [False] * n,
            source_count            = list(self.history_source_count),
            bias_bootstrapped       = [True]  * n,
            bias_active             = [False] * n,
            bias_carried_forward    = [True]  * n,
            bias_mode               = ["carried_forward"] * n,
            active_window_bias      = [math.nan] * n,
            bias_state_stale        = [True]  * n,
            bias_neutralized        = [False] * n,
            bias_observation_count  = [0]     * n,
            bias_state_age_seconds  = [math.nan] * n,
            applied_bias_age_seconds= [600.0] * n,   # past MAX_BIAS_AGE_S clip
            binance_spot_mid        = list(self.history_binance_spot),
            binance_usdm_mid        = list(self.history_binance_usdm),
            hyperliquid_mid         = list(self.history_hyperliquid),
            max_cross_source_spread = list(self.history_max_spread),
        )
