"""Build fair_value_v2_source_time from raw source/event clocks.

This builder intentionally does not reuse the old fair_value_v1 recv-clock
Polymarket replay. Polymarket L2 frames are replayed at payload source time;
the recorder receive timestamp is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fair_value.extractor import FairValueState  # noqa: E402

NS = 1_000_000_000
OUT_DIR = ROOT / "data" / "datasets" / "fair_value_v2_source_time"
ART_DIR = ROOT / "artifacts" / "fair_value_v2_source_time"
FEATURE_MODES = {"full", "pm_rtds"}
CEX_DERIVED_OUTPUT_DROP = {
    "synthetic_chainlink_nowcast",
    "delta_to_strike_nowcast",
    "basis_binance_chainlink",
    "btc_realized_vol_15s",
    "delta_over_vol",
    "p_bs",
    "cex_mid",
    "cex_microprice_tilt",
    "cex_spread",
    "btc_ret_1s",
    "btc_ret_3s",
    "btc_ret_5s",
    "btc_ret_10s",
    "btc_depth_imbalance",
    "btc_ofi_1s",
    "btc_ofi_5s",
    "time_frac_above_strike_30s",
    "strike_crossings_30s",
    "book_vs_oracle_gap",
}

PM_BUCKET_NS = 200_000_000
CEX_BUCKET_NS = 200_000_000
WARMUP_S = 90
EMIT_TTC_HI = 90
EMIT_TTC_LO = 3
EMIT_STEP = 1

_RECV_RE = re.compile(rb'"recv_ts_ns":\s*(\d+)')
_SEQ_RE = re.compile(rb'"local_msg_seq":\s*(\d+)')
_PM_TS_NS_RE = re.compile(rb'"(?:timestamp_ns|message_timestamp_ns)":\s*(\d+)')
_PM_TS_MS_RE = re.compile(rb'"(?:timestamp_ms|message_timestamp_ms)":\s*(\d+)')
_CB_TS_NS_RE = re.compile(rb'"message_timestamp_ns":\s*(\d+)')
_CB_TS_ISO_RE = re.compile(rb'"timestamp"\s*:\s*"([^"]+)"')


def _expand_dates(spec: str | None) -> list[str]:
    if not spec or spec.lower() == "auto":
        return []
    if ":" in spec:
        a, b = spec.split(":", 1)
        d0 = date.fromisoformat(a)
        d1 = date.fromisoformat(b)
        out = []
        while d0 <= d1:
            out.append(d0.isoformat())
            d0 += timedelta(days=1)
        return out
    return [s.strip() for s in spec.split(",") if s.strip()]


def _raw_roots(spec: str) -> list[Path]:
    roots = []
    for item in spec.split(";"):
        item = item.strip()
        if item:
            roots.append(Path(item))
    return roots or [Path("data/raw"), Path("D:/RawDataStorage")]


def _rel(kind: str) -> Path:
    return {
        "pm": Path("polymarket/btc_updown_5m"),
        "rtds": Path("polymarket/rtds/crypto_prices_chainlink/btc_usd"),
        "res": Path("polymarket/resolution/btc_updown_5m"),
        "coinbase": Path("coinbase/advanced/BTC-USD"),
    }[kind]


def _best_dir(roots: list[Path], kind: str, d: str) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    suffix = {
        "pm": "*.l2.jsonl.zst",
        "rtds": "*.ws.jsonl.zst",
        "res": "*.resolution.jsonl.zst",
        "coinbase": "*.ws.jsonl.zst",
    }[kind]
    if os.getenv("FV_RAW_ROOT_PRIORITY", "0") == "1":
        for root in roots:
            path = root / _rel(kind) / d
            if path.exists() and any(path.glob(suffix)):
                return path
        return None
    for root in roots:
        path = root / _rel(kind) / d
        if path.exists():
            candidates.append((len(list(path.glob(suffix))), path))
    candidates = [c for c in candidates if c[0] > 0]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _auto_dates(roots: list[Path], *, require_coinbase: bool = True) -> list[str]:
    dates: set[str] = set()
    for root in roots:
        pm = root / _rel("pm")
        if pm.exists():
            dates.update(p.name for p in pm.iterdir() if p.is_dir() and len(p.name) == 10)
    complete = []
    required = ("pm", "rtds", "res", "coinbase") if require_coinbase else ("pm", "rtds", "res")
    for d in sorted(dates):
        if all(_best_dir(roots, kind, d) is not None for kind in required):
            complete.append(d)
    return complete


def _empty_cex_day() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict[str, Any]]:
    empty_i = np.array([], dtype=np.int64)
    empty_f = np.array([], dtype=np.float64)
    return (
        (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f),
        (empty_i, empty_f, empty_f),
        {"root": None, "disabled": True},
    )


def _iter_raw_lines(path: Path):
    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                try:
                    chunk = reader.read(1 << 20)
                except zstd.ZstdError:
                    break
                if not chunk:
                    break
                buf += chunk
                parts = buf.split(b"\n")
                buf = parts.pop()
                for ln in parts:
                    if ln:
                        yield ln
            if buf.strip():
                yield buf


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    return float(vals[min(len(vals) - 1, int(round((len(vals) - 1) * q)))])


def _lag_stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "p50": _pct(vals, 0.50),
        "p95": _pct(vals, 0.95),
        "p99": _pct(vals, 0.99),
        "max": float(max(vals)),
    }


def _parse_recv_ns(line: bytes) -> int | None:
    m = _RECV_RE.search(line)
    return int(m.group(1)) if m else None


def _parse_seq(line: bytes) -> int:
    m = _SEQ_RE.search(line)
    return int(m.group(1)) if m else 0


def _parse_pm_source_ns(line: bytes) -> int | None:
    m = _PM_TS_NS_RE.search(line)
    if m:
        return int(m.group(1))
    m = _PM_TS_MS_RE.search(line)
    if m:
        return int(m.group(1)) * 1_000_000
    return None


def _iso_to_ns(text: str) -> int | None:
    from datetime import datetime, timezone

    s = text.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * NS)


def _parse_coinbase_source_ns(line: bytes, rec: dict[str, Any] | None = None) -> int | None:
    m = _CB_TS_NS_RE.search(line)
    if m:
        return int(m.group(1))
    if rec is not None:
        st = rec.get("source_timestamps") or {}
        try:
            v = int(st.get("message_timestamp_ns") or 0)
        except (TypeError, ValueError):
            v = 0
        if v > 0:
            return v
        payload = rec.get("payload") or {}
        if isinstance(payload, dict) and payload.get("timestamp"):
            return _iso_to_ns(str(payload["timestamp"]))
    m = _CB_TS_ISO_RE.search(line)
    if m:
        try:
            return _iso_to_ns(m.group(1).decode("utf-8"))
        except UnicodeDecodeError:
            return None
    return None


def _flt(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def load_resolutions(d: str, roots: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    ddir = _best_dir(roots, "res", d)
    stats = {"root": str(ddir) if ddir else None, "rows": 0}
    if ddir is None:
        return out, stats
    for path in sorted(ddir.glob("*.resolution.jsonl.zst")):
        for line in _iter_raw_lines(path):
            if b'"record_type":"market_resolution"' not in line:
                continue
            try:
                rec = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if rec.get("record_type") != "market_resolution":
                continue
            slug = rec.get("market_slug")
            side = rec.get("resolved_side")
            if not slug or side not in ("up", "down"):
                continue
            stats["rows"] += 1
            out[str(slug)] = {
                "open_s": int(rec.get("market_open_s") or 0),
                "close_s": int(rec.get("market_close_s") or 0),
                "resolved_up": 1 if side == "up" else 0,
                "open_price": _flt(rec.get("page_resolution_open_price")),
                "close_price": _flt(rec.get("page_resolution_close_price")),
            }
    return out, stats


def load_rtds_day(d: str, roots: list[Path], latency_s: float) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict[str, Any]]:
    ddir = _best_dir(roots, "rtds", d)
    avail, cl, px, lag = [], [], [], []
    stats = {"root": str(ddir) if ddir else None, "rows": 0, "missing_ts": 0}
    if ddir is None:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=float)
        return (empty_i, empty_i, empty_f, empty_f), (empty_i, empty_f), stats
    latency_ns = int(round(float(latency_s) * NS))
    for path in sorted(ddir.glob("*.ws.jsonl.zst")):
        for line in _iter_raw_lines(path):
            if b"chainlink_live_reference" not in line:
                continue
            try:
                rec = orjson.loads(line)
                recv_ns = int(rec["recv_ts_ns"])
                cl_ns = int(rec["chainlink_ts_ms"]) * 1_000_000
                price = float(rec["btc_usd_price"])
            except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
                stats["missing_ts"] += 1
                continue
            avail.append(cl_ns + latency_ns)
            cl.append(cl_ns)
            px.append(price)
            lag.append((recv_ns - cl_ns) / NS)
            stats["rows"] += 1
    avail_a = np.asarray(avail, dtype=np.int64)
    cl_a = np.asarray(cl, dtype=np.int64)
    px_a = np.asarray(px, dtype=float)
    lag_a = np.asarray(lag, dtype=float)
    o = np.argsort(avail_a, kind="stable") if avail_a.size else np.array([], dtype=np.int64)
    oc = np.argsort(cl_a, kind="stable") if cl_a.size else np.array([], dtype=np.int64)
    stats["source_age_s"] = _lag_stats([float(x) for x in lag])
    return (avail_a[o], cl_a[o], px_a[o], lag_a[o]), (cl_a[oc], px_a[oc]), stats


def _rtds_price_at(cl_sorted: np.ndarray, px_sorted: np.ndarray, target_ns: int) -> float | None:
    if cl_sorted.size == 0:
        return None
    i = int(np.searchsorted(cl_sorted, target_ns, side="right")) - 1
    return float(px_sorted[i]) if i >= 0 else None


def load_coinbase_day(d: str, roots: list[Path]) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict[str, Any]]:
    ddir = _best_dir(roots, "coinbase", d)
    bt_by_bucket: dict[int, tuple[int, float, float, float, float, float]] = {}
    trades: list[tuple[int, float, float]] = []
    lags: list[float] = []
    stats = {"root": str(ddir) if ddir else None, "ticker_rows": 0, "trade_rows": 0, "missing_source_ts": 0}
    if ddir is None:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=float)
        return (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f), (empty_i, empty_f, empty_f), stats
    for path in sorted(ddir.glob("*.ws.jsonl.zst")):
        for line in _iter_raw_lines(path):
            if b'"channel":"ticker"' in line:
                try:
                    rec = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if rec.get("channel") != "ticker":
                    continue
                recv_ns = int(rec.get("recv_ts_ns") or 0)
                source_ns = _parse_coinbase_source_ns(line, rec)
                if source_ns is None:
                    stats["missing_source_ts"] += 1
                    continue
                payload = rec.get("payload") or {}
                for ev in payload.get("events", ()):
                    for tk in ev.get("tickers", ()):
                        bb = tk.get("best_bid")
                        ba = tk.get("best_ask")
                        if not (bb and ba):
                            continue
                        lag_s = (recv_ns - source_ns) / NS if recv_ns else float("nan")
                        bt_by_bucket[source_ns // CEX_BUCKET_NS] = (
                            source_ns, float(bb), float(ba),
                            float(tk.get("best_bid_quantity") or 0.0),
                            float(tk.get("best_ask_quantity") or 0.0),
                            lag_s,
                        )
                        if math.isfinite(lag_s):
                            lags.append(lag_s)
                        stats["ticker_rows"] += 1
            elif b'"channel":"market_trades"' in line:
                try:
                    rec = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if rec.get("channel") != "market_trades":
                    continue
                source_ns = _parse_coinbase_source_ns(line, rec)
                if source_ns is None:
                    stats["missing_source_ts"] += 1
                    continue
                payload = rec.get("payload") or {}
                for ev in payload.get("events", ()):
                    if ev.get("type") == "snapshot":
                        continue
                    for tr in ev.get("trades", ()):
                        sz = tr.get("size")
                        side = tr.get("side")
                        if not sz:
                            continue
                        signed = 1.0 if side == "BUY" else (-1.0 if side == "SELL" else 0.0)
                        trades.append((source_ns, float(sz), signed))
                        stats["trade_rows"] += 1
    bt_rows = sorted(bt_by_bucket.values(), key=lambda x: x[0])
    trades.sort()
    if bt_rows:
        bt = (
            np.asarray([r[0] for r in bt_rows], dtype=np.int64),
            np.asarray([r[1] for r in bt_rows], dtype=float),
            np.asarray([r[2] for r in bt_rows], dtype=float),
            np.asarray([r[3] for r in bt_rows], dtype=float),
            np.asarray([r[4] for r in bt_rows], dtype=float),
            np.asarray([r[5] for r in bt_rows], dtype=float),
        )
    else:
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.array([], dtype=float)
        bt = (empty_i, empty_f, empty_f, empty_f, empty_f, empty_f)
    if trades:
        ag = (
            np.asarray([r[0] for r in trades], dtype=np.int64),
            np.asarray([r[1] for r in trades], dtype=float),
            np.asarray([r[2] for r in trades], dtype=float),
        )
    else:
        ag = (np.array([], dtype=np.int64), np.array([], dtype=float), np.array([], dtype=float))
    stats["source_lag_s"] = _lag_stats(lags)
    return bt, ag, stats


def _slug_from_path(path: Path) -> str | None:
    return next((p for p in path.name.split("__") if p.startswith("btc-updown-5m-")), None)


def _read_pm_frames(path: Path) -> tuple[list[tuple[int, int, int, dict[str, Any], float]], dict[str, Any]]:
    keep_lines: list[tuple[int, int, int, bytes, float]] = []
    pc_by_bucket: dict[int, tuple[int, int, int, bytes, float]] = {}
    lags: list[float] = []
    missing_source_ts = 0
    parsed_lines = 0
    for line in _iter_raw_lines(path):
        if not (b'"event_type":"price_change"' in line
                or b'"event_type":"book"' in line
                or b'"event_type":"best_bid_ask"' in line
                or b'"event_type":"last_trade_price"' in line):
            continue
        recv_ns = _parse_recv_ns(line)
        source_ns = _parse_pm_source_ns(line)
        if recv_ns is None or source_ns is None:
            missing_source_ts += 1
            continue
        seq = _parse_seq(line)
        lag_s = (recv_ns - source_ns) / NS
        lags.append(lag_s)
        item = (source_ns, recv_ns, seq, line, lag_s)
        if b'"event_type":"price_change"' in line:
            pc_by_bucket[source_ns // PM_BUCKET_NS] = item
        else:
            keep_lines.append(item)
    frames: list[tuple[int, int, int, dict[str, Any], float]] = []
    for source_ns, recv_ns, seq, line, lag_s in keep_lines + list(pc_by_bucket.values()):
        try:
            rec = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        rec["raw_recv_ts_ns"] = int(rec.get("recv_ts_ns") or recv_ns)
        rec["recv_ts_ns"] = source_ns
        rec["source_ts_ns"] = source_ns
        rec["pm_delivery_lag_s"] = lag_s
        frames.append((source_ns, recv_ns, seq, rec, lag_s))
        parsed_lines += 1
    frames.sort(key=lambda x: (x[0], x[1], x[2]))
    return frames, {
        "missing_source_ts": missing_source_ts,
        "kept_frames": parsed_lines,
        "source_lag_s": _lag_stats(lags),
    }


_G: dict[str, Any] = {}


def _replay_init(res, rtds, rtds_cl, cexbt, cexag, rtds_latency_s, require_cex: bool = True):
    _G.clear()
    _G.update({
        "res": res,
        "rtds": rtds,
        "rtds_cl": rtds_cl,
        "cexbt": cexbt,
        "cexag": cexag,
        "rtds_latency_s": rtds_latency_s,
        "require_cex": bool(require_cex),
    })


def _build_market_worker(path_s: str):
    return build_market(Path(path_s), _G["res"], _G["rtds"], _G["rtds_cl"],
                        _G["cexbt"], _G["cexag"], _G["rtds_latency_s"],
                        require_cex=_G["require_cex"])


def build_market(pm_path: Path, res: dict[str, dict[str, Any]], rtds, rtds_cl,
                 cexbt, cexag, rtds_latency_s: float,
                 emit_ns: list[int] | None = None,
                 require_cex: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    slug = _slug_from_path(pm_path)
    audit: dict[str, Any] = {"path": str(pm_path), "slug": slug, "rows": 0, "skipped_reason": None}
    if slug is None or slug not in res:
        audit["skipped_reason"] = "missing_resolution"
        return [], audit
    meta = res[slug]
    open_s = int(meta["open_s"])
    close_s = int(meta["close_s"])
    if not (open_s and close_s and close_s > open_s):
        audit["skipped_reason"] = "bad_market_window"
        return [], audit
    strike = meta.get("open_price")
    if not strike or float(strike) <= 0:
        strike = _rtds_price_at(rtds_cl[0], rtds_cl[1], open_s * NS)
    if strike is None or float(strike) <= 0:
        audit["skipped_reason"] = "missing_strike"
        return [], audit

    lo_ns = (open_s - WARMUP_S) * NS
    hi_ns = close_s * NS
    pm_frames, pm_audit = _read_pm_frames(pm_path)
    pm_frames = [x for x in pm_frames if lo_ns <= x[0] <= hi_ns]
    audit["pm"] = pm_audit
    audit["pm_frames_in_window"] = len(pm_frames)

    cts, cbid, cask, cbq, caq, clag = cexbt
    ci0 = int(np.searchsorted(cts, lo_ns, "left"))
    ci1 = int(np.searchsorted(cts, hi_ns, "right"))
    ats, aq, asd = cexag
    ai0 = int(np.searchsorted(ats, lo_ns, "left"))
    ai1 = int(np.searchsorted(ats, hi_ns, "right"))
    ravail, rcl, rpx, rlag = rtds
    ri0 = int(np.searchsorted(ravail, lo_ns, "left"))
    ri1 = int(np.searchsorted(ravail, hi_ns, "right"))
    if not pm_frames:
        audit["skipped_reason"] = "no_pm_frames"
        return [], audit
    if require_cex and ci1 - ci0 == 0:
        audit["skipped_reason"] = "no_coinbase_window"
        return [], audit
    if ri1 - ri0 == 0:
        audit["skipped_reason"] = "no_rtds_window"
        return [], audit

    events: list[tuple[int, int, Any]] = []
    for source_ns, recv_ns, seq, rec, lag_s in pm_frames:
        events.append((int(source_ns), 0, (rec, float(lag_s), int(recv_ns))))
    if require_cex:
        for i in range(ci0, ci1):
            events.append((int(cts[i]), 1, (float(cbid[i]), float(cask[i]), float(cbq[i]), float(caq[i]), float(clag[i]))))
        for i in range(ai0, ai1):
            events.append((int(ats[i]), 2, (float(aq[i]), float(asd[i]))))
    for i in range(ri0, ri1):
        events.append((int(ravail[i]), 3, (int(rcl[i]), float(rpx[i]), float(rlag[i]))))
    events.sort(key=lambda x: (x[0], x[1]))

    st = FairValueState(require_cex=require_cex)
    st.strike = float(strike)
    st.open_s = open_s
    st.close_s = close_s
    st.pm.market_open_s = open_s
    st.pm.market_close_s = close_s

    rows: list[dict[str, Any]] = []
    latest_pm_source_ns = 0
    latest_pm_recv_ns = 0
    latest_pm_delivery_lag_s = float("nan")
    latest_cb_lag_s = float("nan")
    latest_rtds_source_age_s = float("nan")
    ei = 0
    n = len(events)
    if emit_ns is None:
        grid = [(close_s - ttc) * NS for ttc in range(EMIT_TTC_HI, EMIT_TTC_LO - 1, -EMIT_STEP)]
    else:
        grid = sorted(int(x) for x in emit_ns if lo_ns <= int(x) <= hi_ns)
    for now_ns in grid:
        while ei < n and events[ei][0] <= now_ns:
            ts, kind, data = events[ei]
            if kind == 0:
                rec, lag_s, recv_ns = data
                st.update_pm(rec)
                latest_pm_source_ns = int(ts)
                latest_pm_recv_ns = int(recv_ns)
                latest_pm_delivery_lag_s = float(lag_s)
            elif kind == 1:
                bid, ask, bq, aq_, lag_s = data
                st.update_binance_bookticker(ts, bid, ask, bq, aq_)
                latest_cb_lag_s = float(lag_s)
            elif kind == 2:
                st.update_binance_trade(ts, data[0], data[1])
            else:
                cl_ns, price, source_age_s = data
                st.update_rtds(ts, cl_ns, price)
                latest_rtds_source_age_s = float(source_age_s)
            ei += 1
        feat = st.features(now_ns)
        if feat is None:
            continue
        row = dict(feat)
        row.update({
            "market_slug": slug,
            "ttc_s": (close_s * NS - now_ns) / NS,
            "now_ns": int(now_ns),
            "resolved_up": int(meta["resolved_up"]),
            "chainlink_close_price": meta["close_price"] if meta["close_price"] is not None else float("nan"),
            "dataset_clock_mode": "source_time",
            "cex_source": "coinbase" if require_cex else "none",
            "pm_source_lag_s": (now_ns - latest_pm_source_ns) / NS if latest_pm_source_ns else float("nan"),
            "pm_recv_lag_s": (latest_pm_recv_ns - latest_pm_source_ns) / NS
            if latest_pm_recv_ns and latest_pm_source_ns else float("nan"),
            "pm_delivery_lag_s": latest_pm_delivery_lag_s,
            "coinbase_source_lag_s": latest_cb_lag_s,
            "rtds_source_age_s": latest_rtds_source_age_s,
            "rtds_latency_s": float(rtds_latency_s),
        })
        rows.append(row)
    audit["rows"] = len(rows)
    audit["skipped_reason"] = None if rows else "no_feature_rows"
    return rows, audit


def _merge_lag_stats(target: list[float], stats: dict[str, Any]) -> None:
    # The per-market audit stores compressed stats. Keep build_audit compact and
    # compute global lag stats from sampled row diagnostics instead.
    del target, stats


def build_one_date(d: str, roots: list[Path], workers: int, max_markets: int,
                   rebuild: bool, rtds_latency_s: float,
                   require_cex: bool) -> dict[str, Any]:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{d}.parquet"
    if out_path.exists() and not rebuild:
        return {"date": d, "status": "skipped_exists", "output": str(out_path), "elapsed_s": 0.0}

    pm_dir = _best_dir(roots, "pm", d)
    res, res_stats = load_resolutions(d, roots)
    rtds, rtds_cl, rtds_stats = load_rtds_day(d, roots, rtds_latency_s)
    if require_cex:
        cexbt, cexag, cex_stats = load_coinbase_day(d, roots)
    else:
        cexbt, cexag, cex_stats = _empty_cex_day()
    pm_files = sorted(pm_dir.glob("*.l2.jsonl.zst")) if pm_dir else []
    if max_markets:
        pm_files = pm_files[:max_markets]

    stats: dict[str, Any] = {
        "date": d,
        "status": "built",
        "roots": {
            "pm": str(pm_dir) if pm_dir else None,
            "resolution": res_stats.get("root"),
            "rtds": rtds_stats.get("root"),
            "coinbase": cex_stats.get("root"),
        },
        "pm_markets": len(pm_files),
        "resolutions": len(res),
        "coinbase_tickers": int(cexbt[0].size),
        "coinbase_trades": int(cexag[0].size),
        "require_cex": bool(require_cex),
        "rtds_rows": int(rtds[0].size),
        "rtds_cov_pct": round(float(np.unique(rtds_cl[0] // NS).size / 86400 * 100), 2) if rtds_cl[0].size else 0.0,
        "markets_with_rows": 0,
        "markets_skipped": 0,
        "rows": 0,
        "skips": {},
        "source_stats": {
            "coinbase": cex_stats,
            "rtds": rtds_stats,
        },
    }
    required_sources = [
        ("pm", pm_files),
        ("resolution", res),
        ("rtds", rtds[0].size),
    ]
    if require_cex:
        required_sources.append(("coinbase", cexbt[0].size))
    missing = [name for name, ok in required_sources if not ok]
    if missing:
        stats["status"] = "missing_sources"
        stats["note"] = f"missing {missing}"
        stats["elapsed_s"] = round(time.time() - t0, 2)
        return stats

    all_rows: list[dict[str, Any]] = []
    market_audits = []
    if workers > 1 and len(pm_files) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(pm_files)),
                                 initializer=_replay_init,
                                 initargs=(res, rtds, rtds_cl, cexbt, cexag,
                                           rtds_latency_s, require_cex)) as ex:
            for rows, audit in ex.map(_build_market_worker, [str(p) for p in pm_files], chunksize=4):
                market_audits.append(audit)
                if rows:
                    stats["markets_with_rows"] += 1
                    all_rows.extend(rows)
                else:
                    stats["markets_skipped"] += 1
                    reason = audit.get("skipped_reason") or "unknown"
                    stats["skips"][reason] = int(stats["skips"].get(reason, 0)) + 1
    else:
        _replay_init(res, rtds, rtds_cl, cexbt, cexag, rtds_latency_s, require_cex)
        for path in pm_files:
            rows, audit = _build_market_worker(str(path))
            market_audits.append(audit)
            if rows:
                stats["markets_with_rows"] += 1
                all_rows.extend(rows)
            else:
                stats["markets_skipped"] += 1
                reason = audit.get("skipped_reason") or "unknown"
                stats["skips"][reason] = int(stats["skips"].get(reason, 0)) + 1

    stats["rows_raw"] = len(all_rows)
    stats["rows"] = len(all_rows)
    if all_rows:
        import polars as pl

        frame = pl.DataFrame(all_rows)
        if not require_cex:
            drop_cols = [c for c in CEX_DERIVED_OUTPUT_DROP if c in frame.columns]
            if drop_cols:
                frame = frame.drop(drop_cols)
        before_dedup = frame.height
        frame = (
            frame
            .sort(["market_slug", "now_ns", "pm_source_lag_s", "coinbase_source_lag_s", "rtds_source_age_s"])
            .unique(subset=["market_slug", "now_ns"], keep="first", maintain_order=True)
            .sort(["market_slug", "now_ns"])
        )
        stats["duplicate_market_now_rows_removed"] = int(before_dedup - frame.height)
        stats["rows"] = int(frame.height)
        frame.write_parquet(out_path)
        stats["output"] = str(out_path)
        for col in ("pm_source_lag_s", "pm_recv_lag_s", "pm_delivery_lag_s",
                    "coinbase_source_lag_s", "rtds_source_age_s"):
            vals = [float(v) for v in frame[col].drop_nulls().to_list()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))]
            stats.setdefault("row_lag_stats", {})[col] = _lag_stats(vals)
    else:
        stats["status"] = "no_rows"
    stats["market_audit_sample"] = market_audits[:20]
    stats["elapsed_s"] = round(time.time() - t0, 2)
    return stats


def main() -> int:
    global OUT_DIR, ART_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="auto", help="auto, YYYY-MM-DD, comma list, or A:B range")
    ap.add_argument("--raw-roots", default="data/raw;D:/RawDataStorage",
                    help="semicolon-separated raw roots")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--max-markets", type=int, default=0)
    ap.add_argument("--rtds-latency-s", type=float, default=2.5)
    ap.add_argument("--feature-mode", choices=sorted(FEATURE_MODES), default="full",
                    help="full keeps CEX/Coinbase/Binance-derived features; pm_rtds uses only PM+RTDS clocks")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--artifacts-dir", type=Path, default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    require_cex = args.feature_mode == "full"
    default_out = ROOT / "data" / "datasets" / (
        "fair_value_v2_source_time" if require_cex else "fair_value_v2_pm_rtds"
    )
    default_art = ROOT / "artifacts" / (
        "fair_value_v2_source_time" if require_cex else "fair_value_v2_pm_rtds"
    )
    OUT_DIR = args.output_dir or default_out
    ART_DIR = args.artifacts_dir or default_art

    roots = _raw_roots(args.raw_roots)
    dates = _expand_dates(args.dates) or _auto_dates(roots, require_coinbase=require_cex)
    if not dates:
        raise SystemExit("no complete dates found")
    ART_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building fair_value_v2_source_time: mode={args.feature_mode} dates={len(dates)} "
          f"workers={args.workers} roots={[str(r) for r in roots]} "
          f"rtds_latency={args.rtds_latency_s}s", flush=True)

    results = []
    for d in dates:
        stats = build_one_date(d, roots, args.workers, args.max_markets, args.rebuild,
                               args.rtds_latency_s, require_cex)
        results.append(stats)
        print(
            f"  {d} {stats['status']}: rows={stats.get('rows', 0)} "
            f"used={stats.get('markets_with_rows', 0)} skip={stats.get('markets_skipped', 0)} "
            f"rtds_cov={stats.get('rtds_cov_pct', 0)}% {stats.get('elapsed_s', 0)}s"
            + (f" [{stats.get('note')}]" if stats.get("note") else ""),
            flush=True,
        )

    report = {
        "dataset": "fair_value_v2_source_time",
        "feature_mode": args.feature_mode,
        "require_cex": bool(require_cex),
        "clock_policy": "PM replay clock is payload source_ts_ns; recv_ts_ns is diagnostic only.",
        "raw_roots": [str(r) for r in roots],
        "rtds_latency_s": args.rtds_latency_s,
        "dates": results,
        "total_rows": int(sum(int(r.get("rows", 0)) for r in results)),
        "built_at_unix": time.time(),
    }
    (ART_DIR / "build_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TOTAL rows={report['total_rows']:,} -> {OUT_DIR}")
    print(f"Audit -> {ART_DIR / 'build_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
