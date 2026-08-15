"""Build the fair_value_v1 training dataset (Polymarket L2 + a CEX spot feed + RTDS).

Per market window we replay the time-ordered merged frame stream through the
single-source-of-truth extractor (src/fair_value/extractor.py) and emit one
feature row per second over a TTC band, joined to the resolution label.

CEX source is configurable (--cex): `binance` (bookTicker) or `coinbase`
(ticker best_bid/ask). Chainlink's resolution price is a CEX-aggregate, so either
spot feed LEADS it. Binance exists only 05-14+; Coinbase spans the whole history,
so Coinbase is the default (keeps train+test on one CEX source).

SPEED: each day's CEX + RTDS streams are loaded once into compact numpy arrays
(raw-line byte-prefilter skips heavy channels before orjson; top-of-book
downsampled to 200 ms). Polymarket L2 is the chatty one — price_change frames are
downsampled to 200 ms via a cheap recv_ts byte-regex so we orjson-parse ~2.5k
frames/market instead of ~110k. Days run in parallel across cores.

GAP-SAFE: if a required source has no data over a window (or is stale at an emit
point) that market/row is silently skipped — we never fabricate.

Usage:
  # train (Coinbase + RTDS from the D: archive)
  py -3 tools/build_fair_value_dataset.py --dates 2026-04-19:2026-05-13 \
        --raw-root D:/RawDataStorage --cex coinbase
  # june test (local)
  py -3 tools/build_fair_value_dataset.py --dates 2026-06-03:2026-06-16 \
        --raw-root data/raw --cex coinbase
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import orjson
import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fair_value.extractor import FairValueState  # noqa: E402

OUT_DIR = ROOT / "data" / "datasets" / "fair_value_v1"
BT_BUCKET_NS = 200_000_000  # downsample CEX top-of-book to 200 ms (features are ≥1 s)
PM_BUCKET_NS = 200_000_000  # downsample chatty Polymarket price_change to 200 ms
_RECV_RE = re.compile(rb'"recv_ts_ns":\s*(\d+)')

WARMUP_S = 90        # seed CEX vol / Chainlink basis before the emit band
EMIT_TTC_HI = 90     # earliest emit (s before close) — the tradeable window
EMIT_TTC_LO = 3      # latest emit
EMIT_STEP = 2        # seconds between emits (2 s is ample for training; ~2.7x fewer rows)


def _dirs(raw_root: str) -> dict:
    R = Path(raw_root)
    return {
        "pm": R / "polymarket" / "btc_updown_5m",
        "rtds": R / "polymarket" / "rtds" / "crypto_prices_chainlink" / "btc_usd",
        "res": R / "polymarket" / "resolution" / "btc_updown_5m",
        "binance": R / "binance" / "spot" / "BTCUSDT",
        "coinbase": R / "coinbase" / "advanced" / "BTC-USD",
    }


def _iter_raw_lines(path: Path):
    """Stream raw (bytes) JSONL lines from a .zst file, truncation-tolerant."""
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


# ----- CEX loaders (binance bookTicker / coinbase ticker) --------------------

def _load_binance_hour(path_str: str):
    path = Path(path_str)
    bt_by_bucket: dict[int, tuple] = {}
    ag_rows: list[tuple] = []
    try:
        for ln in _iter_raw_lines(path):
            if b'bookTicker' in ln:
                try:
                    r = orjson.loads(ln)
                    if r.get("event_type") != "bookTicker":
                        continue
                    p = r["payload"]; ts = int(r["recv_ts_ns"])
                    bt_by_bucket[ts // BT_BUCKET_NS] = (
                        ts, float(p["b"]), float(p["a"]), float(p["B"]), float(p["A"]))
                except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
            elif b'aggTrade' in ln:
                try:
                    r = orjson.loads(ln)
                    if r.get("event_type") != "aggTrade":
                        continue
                    p = r["payload"]
                    ag_rows.append((int(r["recv_ts_ns"]), float(p["q"]),
                                    -1.0 if p.get("m") else 1.0))
                except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    except Exception:
        pass
    return list(bt_by_bucket.values()), ag_rows


def _load_coinbase_hour(path_str: str):
    """Coinbase advanced ws: ticker -> top-of-book (best_bid/ask + qty),
    market_trades -> signed flow. l2_data/heartbeats skipped."""
    path = Path(path_str)
    bt_by_bucket: dict[int, tuple] = {}
    ag_rows: list[tuple] = []
    try:
        for ln in _iter_raw_lines(path):
            if b'"channel":"ticker"' in ln:
                try:
                    r = orjson.loads(ln)
                    if r.get("channel") != "ticker":
                        continue
                    ts = int(r["recv_ts_ns"])
                    for ev in r["payload"]["events"]:
                        for tk in ev.get("tickers", ()):
                            bb = tk.get("best_bid"); ba = tk.get("best_ask")
                            if bb and ba:
                                bt_by_bucket[ts // BT_BUCKET_NS] = (
                                    ts, float(bb), float(ba),
                                    float(tk.get("best_bid_quantity") or 0.0),
                                    float(tk.get("best_ask_quantity") or 0.0))
                except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
            elif b'"channel":"market_trades"' in ln:
                try:
                    r = orjson.loads(ln)
                    if r.get("channel") != "market_trades":
                        continue
                    ts = int(r["recv_ts_ns"])
                    for ev in r["payload"]["events"]:
                        if ev.get("type") == "snapshot":
                            continue  # historical dump on (re)connect
                        for tr in ev.get("trades", ()):
                            sz = tr.get("size"); side = tr.get("side")
                            if sz:
                                s = 1.0 if side == "BUY" else (-1.0 if side == "SELL" else 0.0)
                                ag_rows.append((ts, float(sz), s))
                except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    except Exception:
        pass
    return list(bt_by_bucket.values()), ag_rows


def load_cex_day(d: str, cex: str, workers: int, raw_root: str):
    """Top-of-book (ts,bid,ask,bq,aq) + trades (ts,qty,side) recv-sorted arrays."""
    dirs = _dirs(raw_root)
    if cex == "coinbase":
        ddir = dirs["coinbase"] / d; hourfn = _load_coinbase_hour
    else:
        ddir = dirs["binance"] / d; hourfn = _load_binance_hour
    files = [str(p) for p in sorted(ddir.glob("*.ws.jsonl.zst"))] if ddir.exists() else []
    bt_rows: list[tuple] = []
    ag_rows: list[tuple] = []
    if files:
        if workers > 1 and len(files) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(files))) as ex:
                for bt_h, ag_h in ex.map(hourfn, files):
                    bt_rows.extend(bt_h); ag_rows.extend(ag_h)
        else:
            for f in files:
                bt_h, ag_h = hourfn(f)
                bt_rows.extend(bt_h); ag_rows.extend(ag_h)
    bt_rows.sort()
    ag_rows.sort()
    if bt_rows:
        bt = (np.fromiter((r[0] for r in bt_rows), np.int64, len(bt_rows)),
              np.fromiter((r[1] for r in bt_rows), float, len(bt_rows)),
              np.fromiter((r[2] for r in bt_rows), float, len(bt_rows)),
              np.fromiter((r[3] for r in bt_rows), float, len(bt_rows)),
              np.fromiter((r[4] for r in bt_rows), float, len(bt_rows)))
    else:
        bt = (np.array([], np.int64), np.array([]), np.array([]), np.array([]), np.array([]))
    if ag_rows:
        ag = (np.fromiter((r[0] for r in ag_rows), np.int64, len(ag_rows)),
              np.fromiter((r[1] for r in ag_rows), float, len(ag_rows)),
              np.fromiter((r[2] for r in ag_rows), float, len(ag_rows)))
    else:
        ag = (np.array([], np.int64), np.array([]), np.array([]))
    return bt, ag


# ----- RTDS / resolution / PM loaders ----------------------------------------

def _flt(x):
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def load_resolutions(d: str, raw_root: str) -> dict:
    out: dict = {}
    ddir = _dirs(raw_root)["res"] / d
    if not ddir.exists():
        return out
    for path in sorted(ddir.glob("*.resolution.jsonl.zst")):
        try:
            for r in _iter_raw_lines(path):
                if b'"record_type":"market_resolution"' not in r:
                    continue
                try:
                    rec = orjson.loads(r)
                except orjson.JSONDecodeError:
                    continue
                if rec.get("record_type") != "market_resolution":
                    continue
                slug = rec.get("market_slug"); side = rec.get("resolved_side")
                if not slug or side not in ("up", "down"):
                    continue
                out[slug] = {
                    "open_s": int(rec.get("market_open_s") or 0),
                    "close_s": int(rec.get("market_close_s") or 0),
                    "resolved_up": 1 if side == "up" else 0,
                    "open_price": _flt(rec.get("page_resolution_open_price")),
                    "close_price": _flt(rec.get("page_resolution_close_price")),
                }
        except Exception:
            continue
    return out


def load_rtds_day(d: str, raw_root: str):
    ddir = _dirs(raw_root)["rtds"] / d
    recv, cl, px = [], [], []
    if ddir.exists():
        for path in sorted(ddir.glob("*.ws.jsonl.zst")):
            try:
                for ln in _iter_raw_lines(path):
                    if b'chainlink_live_reference' not in ln:
                        continue
                    try:
                        r = orjson.loads(ln)
                        recv.append(int(r["recv_ts_ns"]))
                        cl.append(int(r["chainlink_ts_ms"]) * 1_000_000)
                        px.append(float(r["btc_usd_price"]))
                    except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
            except Exception:
                continue
    recv = np.asarray(recv, np.int64); cl = np.asarray(cl, np.int64); px = np.asarray(px, float)
    o = np.argsort(recv, kind="stable")
    oc = np.argsort(cl, kind="stable")
    return (recv[o], cl[o], px[o]), (cl[oc], px[oc])


def _rtds_price_at(cl_sorted, px_sorted, target_ns):
    if cl_sorted.size == 0:
        return None
    i = int(np.searchsorted(cl_sorted, target_ns, side="right")) - 1
    return float(px_sorted[i]) if i >= 0 else None


def _read_pm_frames(path: Path):
    """Parse a market's L2 frames, DOWNSAMPLING the high-frequency price_change
    stream to 200 ms before orjson (it dominates: ~110k frames/market). recv_ts_ns
    is pulled with a byte-regex so we never JSON-parse the frames we drop.
    NOTE: live extractor must run at the same ≤200 ms cadence for parity."""
    keep_lines: list[tuple[int, bytes]] = []
    pc_by_bucket: dict[int, tuple[int, bytes]] = {}
    try:
        for ln in _iter_raw_lines(path):
            m = _RECV_RE.search(ln)
            if not m:
                continue
            ts = int(m.group(1))
            if b'"event_type":"price_change"' in ln:
                pc_by_bucket[ts // PM_BUCKET_NS] = (ts, ln)
            elif (b'"event_type":"book"' in ln
                  or b'"event_type":"best_bid_ask"' in ln
                  or b'"event_type":"last_trade_price"' in ln):
                keep_lines.append((ts, ln))
    except Exception:
        pass
    frames: list[tuple[int, dict]] = []
    for ts, ln in keep_lines:
        try:
            frames.append((ts, orjson.loads(ln)))
        except orjson.JSONDecodeError:
            pass
    for ts, ln in pc_by_bucket.values():
        try:
            frames.append((ts, orjson.loads(ln)))
        except orjson.JSONDecodeError:
            pass
    frames.sort(key=lambda x: x[0])
    return frames


# ----- per-market replay -----------------------------------------------------

def build_market(pm_path: Path, res: dict, rtds, rtds_cl, cexbt, cexag, rtds_lat_ns: int = -1,
                 emit_ns: list[int] | None = None) -> list:
    name = pm_path.name
    slug = next((p for p in name.split("__") if p.startswith("btc-updown-5m-")), None)
    if slug is None or slug not in res:
        return []
    meta = res[slug]
    open_s, close_s = meta["open_s"], meta["close_s"]
    if not (open_s and close_s and close_s > open_s):
        return []
    strike = meta["open_price"]
    # Fall back to the RTDS/chainlink price at open when the scraped page
    # open_price is missing OR 0.0 (the strike scrape can break — None comes
    # through _flt as 0.0). Same RTDS-derived strike the live bot uses, so
    # backtest==live. For days where the scrape worked, open_price>0 and this
    # branch is skipped -> training data unchanged.
    if not strike or strike <= 0:
        strike = _rtds_price_at(rtds_cl[0], rtds_cl[1], open_s * 1_000_000_000)
    if strike is None or strike <= 0:
        return []

    NS = 1_000_000_000
    lo_ns = (open_s - WARMUP_S) * NS
    hi_ns = close_s * NS

    pm_frames = [(ts, fr) for (ts, fr) in _read_pm_frames(pm_path) if lo_ns <= ts <= hi_ns]
    recv, cl_r, px_r = rtds
    ri0 = int(np.searchsorted(recv, lo_ns, "left")); ri1 = int(np.searchsorted(recv, hi_ns, "right"))
    bts, bb, ba, bbq, baq = cexbt
    bi0 = int(np.searchsorted(bts, lo_ns, "left")); bi1 = int(np.searchsorted(bts, hi_ns, "right"))
    ats, aq, asd = cexag
    ai0 = int(np.searchsorted(ats, lo_ns, "left")); ai1 = int(np.searchsorted(ats, hi_ns, "right"))

    if (bi1 - bi0) == 0 or (ri1 - ri0) == 0:
        return []  # gap-safe: no CEX or no RTDS over this window -> skip market

    events = []
    for ts, fr in pm_frames:
        events.append((ts, 0, fr))
    for i in range(bi0, bi1):
        events.append((int(bts[i]), 1, (float(bb[i]), float(ba[i]), float(bbq[i]), float(baq[i]))))
    for i in range(ai0, ai1):
        events.append((int(ats[i]), 2, (float(aq[i]), float(asd[i]))))
    for i in range(ri0, ri1):
        # LIVE-PARITY OPTION (rtds_lat_ns >= 0): the recorder feed (recv_ts) is
        # ~1s SLOWER than the live bot's direct WS, so training on recv_ts bakes
        # in extra RTDS staleness the bot never has live -> the model over-enters
        # live. Instead make each tick AVAILABLE at chainlink_ts + live_latency
        # (the bot's measured feed lag), not the recorder's recv_ts. chainlink_ts
        # is the publish time, so we still only use prices published >= latency
        # before the emit -> causally faithful to live, NOT a true future leak.
        # rtds_lat_ns < 0 keeps the legacy recv_ts (recorder-latency) behaviour.
        avail = (int(cl_r[i]) + rtds_lat_ns) if rtds_lat_ns >= 0 else int(recv[i])
        events.append((avail, 3, (int(cl_r[i]), float(px_r[i]))))
    events.sort(key=lambda e: e[0])

    st = FairValueState()
    st.strike = float(strike); st.open_s = open_s; st.close_s = close_s

    if emit_ns is None:
        grid = [(close_s - ttc) * NS for ttc in range(EMIT_TTC_HI, EMIT_TTC_LO - 1, -EMIT_STEP)]
    else:
        grid = sorted(int(t) for t in emit_ns if lo_ns <= int(t) <= hi_ns)
    rows = []
    ei = 0
    n = len(events)
    for now_ns in grid:
        while ei < n and events[ei][0] <= now_ns:
            ts, kind, data = events[ei]
            if kind == 0:
                st.update_pm(data)
            elif kind == 1:
                st.update_binance_bookticker(ts, data[0], data[1], data[2], data[3])
            elif kind == 2:
                st.update_binance_trade(ts, data[0], data[1])
            else:
                st.update_rtds(ts, data[0], data[1])
            ei += 1
        feat = st.features(now_ns)
        if feat is None:
            continue
        feat["market_slug"] = slug
        feat["ttc_s"] = (close_s * NS - now_ns) / NS
        feat["now_ns"] = now_ns
        feat["resolved_up"] = meta["resolved_up"]
        feat["chainlink_close_price"] = meta["close_price"] if meta["close_price"] is not None else float("nan")
        rows.append(feat)
    return rows


# Per-worker day data for the replay pool (single-day path only).
_G: dict = {}


def _replay_init(res, rtds, rtds_cl, cexbt, cexag, rtds_lat_ns):
    _G["res"], _G["rtds"], _G["rtds_cl"], _G["cexbt"], _G["cexag"] = res, rtds, rtds_cl, cexbt, cexag
    _G["rtds_lat_ns"] = rtds_lat_ns


def _replay_market(pm_path_str: str):
    return build_market(Path(pm_path_str), _G["res"], _G["rtds"], _G["rtds_cl"],
                        _G["cexbt"], _G["cexag"], _G["rtds_lat_ns"])


def build_one_date(d: str, load_workers: int, max_markets: int, raw_root: str, cex: str,
                   rtds_lat_ns: int = -1) -> dict:
    t0 = time.time()
    res = load_resolutions(d, raw_root)
    rtds, rtds_cl = load_rtds_day(d, raw_root)
    cexbt, cexag = load_cex_day(d, cex, load_workers, raw_root)
    pm_dir = _dirs(raw_root)["pm"] / d
    pm_files = sorted(pm_dir.glob("*.l2.jsonl.zst")) if pm_dir.exists() else []
    if max_markets:
        pm_files = pm_files[:max_markets]
    t_load = round(time.time() - t0, 1)

    # health: RTDS second-coverage over the day (feed ≈ 1 update/s)
    rtds_cov = 0.0
    if rtds_cl[0].size:
        rtds_cov = round(np.unique(rtds_cl[0] // 1_000_000_000).size / 86400 * 100, 1)

    stats = {"date": d, "pm_markets": len(pm_files), "resolutions": len(res),
             "rtds_rows": int(rtds[0].size), "rtds_cov": rtds_cov, "cex_bt": int(cexbt[0].size),
             "markets_with_rows": 0, "rows": 0, "skipped": 0, "t_load": t_load}
    if not pm_files or cexbt[0].size == 0 or rtds[0].size == 0:
        miss = [n for n, ok in (("pm", pm_files), (cex, cexbt[0].size), ("rtds", rtds[0].size)) if not ok]
        stats["note"] = f"missing {miss} -> 0 rows"
        stats["elapsed_s"] = round(time.time() - t0, 1)
        return stats

    all_rows = []

    def _account(r):
        if r:
            stats["markets_with_rows"] += 1
            all_rows.extend(r)
        else:
            stats["skipped"] += 1

    if load_workers > 1 and len(pm_files) > 1:
        with ProcessPoolExecutor(max_workers=load_workers, initializer=_replay_init,
                                 initargs=(res, rtds, rtds_cl, cexbt, cexag, rtds_lat_ns)) as ex:
            for r in ex.map(_replay_market, [str(p) for p in pm_files], chunksize=4):
                _account(r)
    else:
        for pmf in pm_files:
            _account(build_market(pmf, res, rtds, rtds_cl, cexbt, cexag, rtds_lat_ns))
    stats["rows"] = len(all_rows)

    if all_rows:
        import polars as pl
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(all_rows).with_columns(pl.lit(cex).alias("cex_source")).write_parquet(
            OUT_DIR / f"{d}.parquet")
    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats


def _expand_dates(spec: str) -> list:
    if ":" in spec:
        a, b = spec.split(":")
        d0 = date.fromisoformat(a); d1 = date.fromisoformat(b)
        out = []
        while d0 <= d1:
            out.append(d0.isoformat()); d0 += timedelta(days=1)
        return out
    return [s.strip() for s in spec.split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True, help="YYYY-MM-DD, comma list, or A:B range")
    ap.add_argument("--raw-root", default="data/raw", help="root holding polymarket/, binance/, coinbase/")
    ap.add_argument("--cex", choices=("coinbase", "binance"), default="coinbase")
    # 8 cores / 16 GB: ~0.4 GB per concurrent day-process -> all cores fit comfortably
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--max-markets", type=int, default=0, help="cap markets/day (debug)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild days whose parquet already exists")
    ap.add_argument("--rtds-live-latency", type=float, default=-1.0,
                    help="if >=0, make RTDS ticks available at chainlink_ts + this many SECONDS "
                         "(models the live direct-WS feed lag) instead of the slower recorder recv_ts. "
                         "Use ~1.0 to match the live bot's ~2.2s rtds_age. -1 = legacy recorder behaviour.")
    args = ap.parse_args()
    dates = _expand_dates(args.dates)
    rtds_lat_ns = int(args.rtds_live_latency * 1_000_000_000) if args.rtds_live_latency >= 0 else -1

    print(f"Building fair_value_v1: {len(dates)} day(s), cex={args.cex}, "
          f"raw_root={args.raw_root}, workers={args.workers}", flush=True)
    # Sequential days, parallel WITHIN each day (1 day in RAM at a time -> no OOM;
    # one disk read at a time -> no thrash). Crash-proof: a bad day is logged and
    # skipped, not fatal. Completed days are skipped on re-run.
    results = []
    for d in dates:
        out = OUT_DIR / f"{d}.parquet"
        if (not args.rebuild) and out.exists():
            print(f"  skip {d} (exists)", flush=True)
            continue
        try:
            s = build_one_date(d, args.workers, args.max_markets, args.raw_root, args.cex, rtds_lat_ns)
            results.append(s)
            print(f"  done {s['date']}: rows={s['rows']} rtds_cov={s['rtds_cov']}% "
                  f"used={s['markets_with_rows']} skip={s['skipped']} "
                  f"{s.get('elapsed_s',0)}s" + (f" [{s['note']}]" if s.get('note') else ""), flush=True)
        except Exception as e:
            print(f"  FAILED {d}: {e!r}", flush=True)

    results.sort(key=lambda s: s["date"])
    print(f"\n{'date':12s} {'mkts':>5s} {'rtds%':>6s} {'cexBT':>8s} {'used':>5s} "
          f"{'skip':>5s} {'rows':>8s} {'sec':>6s}")
    tot = 0
    for s in results:
        tot += s.get("rows", 0)
        print(f"{s['date']:12s} {s['pm_markets']:>5d} {s['rtds_cov']:>6.1f} {s['cex_bt']:>8d} "
              f"{s['markets_with_rows']:>5d} {s['skipped']:>5d} {s['rows']:>8d} "
              f"{s.get('elapsed_s',0):>6.1f}" + (f"  [{s['note']}]" if s.get("note") else ""))
    print(f"\nTOTAL rows: {tot:,}  -> {OUT_DIR}")


if __name__ == "__main__":
    main()
