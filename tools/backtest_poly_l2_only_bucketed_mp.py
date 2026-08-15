"""Bucketed backtest (multiprocessing variant) — generic over v1/v2 models.

Speed plan:
  - Each market is independent → trivially parallelizable across processes.
  - Worker init loads model + calibrator once, holds them for the worker's
    lifetime. maxtasksperchild caps memory drift.
  - With 7 workers, ~7x speedup vs serial. Single-day run drops from ~17 min
    to ~2.5 min.

Model variants:
  v1 (artifacts_cleaned/poly_l2_only_v1/):
    predict path:  p_up = calibrator(model.predict(X))
  v2 (artifacts_cleaned/poly_l2_only_v2/):
    predict path:  p_up = sigmoid( logit(implied_p_up) + model.predict(X, raw_score=True) )
    detected via `model._uses_init_score = True` attribute set at training.

Trading rule (same as previous bucketed run):
  - At most one entry per market per TTC bucket (5 buckets of 10s, [10..60]).
  - First qualifying tick within each bucket wins; further ticks skip the
    predict() call (the optimisation we added last iteration).
  - Qualify: EV/share >= MIN_EDGE AND EV/cost >= MIN_EV_FRAC AND
             fill in [FILL_PRICE_MIN, FILL_PRICE_MAX].
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import orjson
import zstandard as zstd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from poly_l2_only.extractor import (  # noqa: E402
    EMIT_EVENT_TYPES, MarketState, NS_PER_S,
    state_to_features, update_state,
)
from poly_l2_only.resolution import load_resolution_map  # noqa: E402

DATA_RAW = REPO_ROOT / "data" / "raw" / "polymarket" / "btc_updown_5m"
RESOLUTION_ROOT = REPO_ROOT / "data" / "raw" / "polymarket" / "resolution" / "btc_updown_5m"

# Trading params (defaults; CLI can override).
NOTIONAL_USD = 5.0
MIN_EDGE_PER_SHARE = 0.04
MIN_EV_FRAC = 0.06
TTC_MIN_S = 10.0
TTC_MAX_S = 60.0
FILL_PRICE_MIN = 0.30
FILL_PRICE_MAX = 0.70
COMMISSION_BPS = 0.0
BUCKET_EDGES = [(50, 60), (40, 50), (30, 40), (20, 30), (10, 20)]
N_BUCKETS = len(BUCKET_EDGES)
LOGIT_CLIP = (1e-4, 1 - 1e-4)


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# Worker globals — set by _worker_init.
_MODEL = None
_CALIB = None
_FEATURE_ORDER: List[str] = []
_USES_INIT_SCORE = False
_IMPLIED_P_UP_IDX = -1
_USE_CALIBRATOR = True
_PARAMS = {}


def _worker_init(artifact_dir_str: str, use_calibrator: bool, params: dict) -> None:
    global _MODEL, _CALIB, _FEATURE_ORDER, _USES_INIT_SCORE, _IMPLIED_P_UP_IDX, _USE_CALIBRATOR, _PARAMS
    art = Path(artifact_dir_str)
    _MODEL = joblib.load(art / "model.pkl")
    _USES_INIT_SCORE = bool(getattr(_MODEL, "_uses_init_score", False))
    _FEATURE_ORDER = json.loads((art / "features.json").read_text())
    _IMPLIED_P_UP_IDX = _FEATURE_ORDER.index("implied_p_up")
    _USE_CALIBRATOR = use_calibrator
    if use_calibrator:
        cal_path = art / "calibrator.pkl"
        _CALIB = joblib.load(cal_path) if cal_path.exists() else getattr(_MODEL, "_calibrator", None)
    _PARAMS = params


def iter_frames(path: Path):
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = bytearray()
            while True:
                chunk = reader.read(1 << 16)
                if not chunk:
                    break
                buf.extend(chunk)
                start = 0
                while True:
                    nl = buf.find(b"\n", start)
                    if nl < 0:
                        break
                    line = bytes(buf[start:nl])
                    start = nl + 1
                    if line:
                        try:
                            yield orjson.loads(line)
                        except Exception:
                            continue
                del buf[:start]


def ttc_bucket(ttc_s: float) -> Optional[int]:
    for i, (lo, hi) in enumerate(BUCKET_EDGES):
        if lo <= ttc_s < hi:
            return i
    return None


def _predict_p_up(X: np.ndarray) -> float:
    """Returns final P(Up) for one row, handling init_score path for v2."""
    if _USES_INIT_SCORE:
        mp_val = float(X[0, _IMPLIED_P_UP_IDX])
        mp_clip = min(max(mp_val, LOGIT_CLIP[0]), LOGIT_CLIP[1])
        init_z = math.log(mp_clip / (1.0 - mp_clip))
        raw = float(_MODEL.predict(X, raw_score=True, num_iteration=_MODEL.best_iteration)[0])
        p = 1.0 / (1.0 + math.exp(-(init_z + raw)))
    else:
        p = float(_MODEL.predict(X, num_iteration=_MODEL.best_iteration)[0])
    if _USE_CALIBRATOR and _CALIB is not None:
        p = float(_CALIB.transform([p])[0])
    return p


def _worker_process_one_tagged(args: Tuple[str, str, str]) -> Tuple[str, List[dict]]:
    """Same as _worker_process_one but tags the result with the source date."""
    path_str, outcome, date_str = args
    entries = _worker_process_one((path_str, outcome))
    return (date_str, entries)


def _worker_process_one(args: Tuple[str, str]) -> List[dict]:
    path_str, outcome = args
    path = Path(path_str)
    min_edge = _PARAMS["min_edge"]
    min_ev_frac = _PARAMS["min_ev_frac"]
    fill_min = _PARAMS["fill_min"]
    fill_max = _PARAMS["fill_max"]

    state = MarketState()
    entries_by_bucket: Dict[int, dict] = {}
    buf = np.zeros((1, len(_FEATURE_ORDER)), dtype=np.float32)

    for frame in iter_frames(path):
        et = frame.get("event_type")
        if et not in EMIT_EVENT_TYPES:
            continue
        if not update_state(state, frame):
            continue
        ts_ns = int(frame.get("recv_ts_ns") or 0)
        ttc_s = (state.market_close_s * NS_PER_S - ts_ns) / NS_PER_S
        if ttc_s < TTC_MIN_S or ttc_s > TTC_MAX_S:
            continue
        b = ttc_bucket(ttc_s)
        if b is None:
            continue
        if b in entries_by_bucket:
            continue

        feats = state_to_features(state, ts_ns)
        for i, c in enumerate(_FEATURE_ORDER):
            buf[0, i] = feats.get(c, 0.0)
        p_up = _predict_p_up(buf)

        ba_up = feats["up_best_ask"]
        ba_dn = feats["down_best_ask"]
        ev_up = p_up - ba_up if (0.0 < ba_up < 1.0) else -1.0
        ev_dn = (1.0 - p_up) - ba_dn if (0.0 < ba_dn < 1.0) else -1.0

        side = None
        if ev_up >= min_edge and ev_up >= ev_dn and (ev_up / ba_up) >= min_ev_frac:
            side = "UP"; fill = ba_up; ev = ev_up
        elif ev_dn >= min_edge and ev_dn > ev_up and (ev_dn / ba_dn) >= min_ev_frac:
            side = "DOWN"; fill = ba_dn; ev = ev_dn
        if side is None:
            continue
        if not (fill_min <= fill <= fill_max):
            continue

        entries_by_bucket[b] = {
            "slug": state.event_slug,
            "ttc_s": round(ttc_s, 2),
            "bucket": b,
            "side": side,
            "fill_price": round(fill, 4),
            "p_model": round(p_up, 4),
            "ev_per_share": round(ev, 4),
            "outcome": outcome,
        }
        if len(entries_by_bucket) >= N_BUCKETS:
            break

    entries = []
    for b, c in sorted(entries_by_bucket.items()):
        shares = NOTIONAL_USD / c["fill_price"]
        fee = NOTIONAL_USD * COMMISSION_BPS / 1e4
        won = (c["side"] == "UP" and outcome == "Up") or (c["side"] == "DOWN" and outcome == "Down")
        payout = shares * 1.0 if won else 0.0
        pnl = payout - NOTIONAL_USD - fee
        c["shares"] = round(shares, 4)
        c["won"] = int(won)
        c["pnl_usd"] = round(pnl, 4)
        entries.append(c)
    return entries


def summarize(entries: List[dict], label: str) -> dict:
    if not entries:
        print(f"=== {label} === (no entries)")
        return {"label": label, "n_entries": 0}
    n = len(entries)
    n_wins = sum(e["won"] for e in entries)
    total_pnl = sum(e["pnl_usd"] for e in entries)
    nominal = NOTIONAL_USD * n
    print(f"=== {label} ===")
    print(f"  entries          : {n}")
    print(f"  notional         : ${nominal:,.2f}")
    print(f"  wins             : {n_wins}  ({100*n_wins/n:.2f}%)")
    print(f"  TOTAL PnL        : ${total_pnl:+,.2f}")
    print(f"  ROI on notional  : {100*total_pnl/nominal:+.2f}%")
    print(f"  avg PnL / trade  : ${total_pnl/n:+.4f}")
    by_bucket = defaultdict(list)
    for e in entries:
        by_bucket[e["bucket"]].append(e)
    print(f"  PnL by TTC bucket:")
    for b, (lo, hi) in enumerate(BUCKET_EDGES):
        bs = by_bucket.get(b, [])
        if not bs:
            print(f"    [{lo},{hi})s: n=  0")
            continue
        bn = len(bs); bp = sum(e["pnl_usd"] for e in bs); bw = sum(e["won"] for e in bs)
        af = float(np.mean([e["fill_price"] for e in bs]))
        ae = float(np.mean([e["ev_per_share"] for e in bs]))
        print(f"    [{lo},{hi})s: n={bn:>3}  pnl=${bp:+8.2f}  avg=${bp/bn:+.4f}  "
              f"wr={100*bw/bn:5.1f}%  fill={af:.3f}  ev/sh={ae:+.3f}")
    by_side = defaultdict(list)
    for e in entries:
        by_side[e["side"]].append(e)
    for sd, es in by_side.items():
        bp = sum(e["pnl_usd"] for e in es)
        bw = sum(e["won"] for e in es)
        print(f"  side={sd:4s}: n={len(es):>3}  pnl=${bp:+8.2f}  wr={100*bw/len(es):.1f}%")
    print()
    return {"label": label, "n_entries": n, "total_pnl": total_pnl,
            "roi": total_pnl / nominal, "win_rate": n_wins / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True,
                    help="e.g. artifacts_cleaned/poly_l2_only_v2")
    ap.add_argument("--date", default="2026-05-28",
                    help="Single date (ignored if --dates provided).")
    ap.add_argument("--dates", default="",
                    help="Comma-separated date list, e.g. "
                         "'2026-04-22,2026-04-30,2026-05-07'. Overrides --date.")
    ap.add_argument("--n-markets", type=int, default=200,
                    help="Per-day cap on markets queued.")
    ap.add_argument("--min-edge", type=float, default=MIN_EDGE_PER_SHARE)
    ap.add_argument("--min-ev-frac", type=float, default=MIN_EV_FRAC)
    ap.add_argument("--fill-min", type=float, default=FILL_PRICE_MIN)
    ap.add_argument("--fill-max", type=float, default=FILL_PRICE_MAX)
    ap.add_argument("--no-calibrator", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    art_dir = (REPO_ROOT / args.artifact_dir).resolve() if not Path(args.artifact_dir).is_absolute() \
        else Path(args.artifact_dir)
    print(f"[{ts()}] artifact_dir: {art_dir}", flush=True)
    print(f"[{ts()}] loading resolution ...", flush=True)
    res_map = load_resolution_map(RESOLUTION_ROOT, verbose=False)
    print(f"[{ts()}] resolution slugs = {len(res_map):,}", flush=True)

    dates = [d.strip() for d in args.dates.split(",") if d.strip()] if args.dates else [args.date]
    # Pre-filter for label across all dates. Each task carries its date for
    # per-day aggregation in the output.
    tasks: List[Tuple[str, str]] = []
    task_dates: List[str] = []
    n_no_label_total = 0
    n_files_total = 0
    per_day_queue_size: Dict[str, int] = {}
    for d in dates:
        day_dir = DATA_RAW / d
        files = sorted(day_dir.glob("*.l2.jsonl.zst"))[: args.n_markets]
        per_day_queue_size[d] = len(files)
        n_files_total += len(files)
        for fp in files:
            slug = fp.name.split("__")[1]
            outcome = res_map.get(slug)
            if outcome not in ("Up", "Down"):
                n_no_label_total += 1
                continue
            tasks.append((str(fp), outcome))
            task_dates.append(d)

    params = {
        "min_edge": args.min_edge,
        "min_ev_frac": args.min_ev_frac,
        "fill_min": args.fill_min,
        "fill_max": args.fill_max,
    }
    use_cal = not args.no_calibrator
    print(f"[{ts()}] dates: {dates}", flush=True)
    print(f"[{ts()}] per-day queue: " +
          ", ".join(f"{d}={n}" for d, n in per_day_queue_size.items()), flush=True)
    print(f"[{ts()}] {len(tasks)} markets to process (no_label_skipped={n_no_label_total}) | "
          f"min_edge=${args.min_edge} | min_ev_frac={args.min_ev_frac*100:.1f}% | "
          f"fills in [${args.fill_min:.2f},${args.fill_max:.2f}] | "
          f"calibrator={'on' if use_cal else 'OFF'} | workers={args.workers}", flush=True)

    t0 = time.time()
    all_entries: List[dict] = []
    entries_by_date: Dict[str, List[dict]] = {d: [] for d in dates}
    n_no_entry = 0

    with mp.Pool(processes=args.workers,
                 initializer=_worker_init,
                 initargs=(str(art_dir), use_cal, params),
                 maxtasksperchild=50) as pool:
        done = 0
        # Pair (task, date) and dispatch as a single-arg call. imap_unordered
        # doesn't preserve order, so we encode the date inside the task tuple.
        date_tagged_tasks = [(t[0], t[1], task_dates[i]) for i, t in enumerate(tasks)]
        for date_str, entries in pool.imap_unordered(
                _worker_process_one_tagged, date_tagged_tasks, chunksize=2):
            done += 1
            if not entries:
                n_no_entry += 1
            else:
                for e in entries:
                    e["date"] = date_str
                all_entries.extend(entries)
                entries_by_date[date_str].extend(entries)
            if done % 100 == 0:
                print(f"[{ts()}]   {done}/{len(tasks)} done, entries={len(all_entries)}",
                      flush=True)

    dt = time.time() - t0
    print(f"[{ts()}] backtest pass: {dt:.1f}s for {len(tasks)} markets "
          f"({dt/max(1,len(tasks)):.2f}s/market wall)", flush=True)
    print(f"[{ts()}] no_entry markets = {n_no_entry}, total entries = {len(all_entries)}")
    print()

    s_all = summarize(all_entries, "ALL SIDES (UP+DOWN) — AGGREGATE")
    s_up = summarize([e for e in all_entries if e["side"] == "UP"], "UP-ONLY — AGGREGATE")
    s_dn = summarize([e for e in all_entries if e["side"] == "DOWN"], "DOWN-ONLY — AGGREGATE")

    # Per-day summary table — the bit that tells us if the asymmetry persists.
    print("=== PER-DAY BREAKDOWN ===")
    print(f"  {'date':<12} {'n_all':>6} {'wr_all':>7} {'pnl_all':>10} "
          f"{'n_up':>5} {'wr_up':>7} {'pnl_up':>10} "
          f"{'n_dn':>5} {'wr_dn':>7} {'pnl_dn':>10}")
    per_day_rows = []
    for d in dates:
        es = entries_by_date[d]
        ups = [e for e in es if e["side"] == "UP"]
        dns = [e for e in es if e["side"] == "DOWN"]
        def stat(rows: List[dict]) -> Tuple[int, float, float]:
            if not rows:
                return (0, float("nan"), 0.0)
            n = len(rows)
            wr = sum(e["won"] for e in rows) / n
            pnl = sum(e["pnl_usd"] for e in rows)
            return (n, wr, pnl)
        a_n, a_wr, a_pnl = stat(es)
        u_n, u_wr, u_pnl = stat(ups)
        d_n, d_wr, d_pnl = stat(dns)
        print(f"  {d:<12} {a_n:>6} {a_wr*100:>6.1f}% ${a_pnl:>+8.2f} "
              f"{u_n:>5} {u_wr*100:>6.1f}% ${u_pnl:>+8.2f} "
              f"{d_n:>5} {d_wr*100:>6.1f}% ${d_pnl:>+8.2f}")
        per_day_rows.append({
            "date": d, "all_n": a_n, "all_wr": a_wr, "all_pnl": a_pnl,
            "up_n": u_n, "up_wr": u_wr, "up_pnl": u_pnl,
            "down_n": d_n, "down_wr": d_wr, "down_pnl": d_pnl,
        })
    print()

    # Save.
    out_dir = art_dir / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.dates.replace(",", "_") if args.dates else args.date
    out_path = out_dir / f"bucketed_mp_{tag}.json"
    out_path.write_text(json.dumps({
        "artifact_dir": str(art_dir),
        "dates": dates,
        "per_day_queue_size": per_day_queue_size,
        "n_markets_queued_total": n_files_total,
        "n_no_label": n_no_label_total,
        "n_no_entry": n_no_entry,
        "n_entries": len(all_entries),
        "elapsed_s": round(dt, 1),
        "per_day_summary": per_day_rows,
        "params": {
            "notional": NOTIONAL_USD,
            "min_edge_per_share": args.min_edge,
            "min_ev_frac": args.min_ev_frac,
            "ttc_range": [TTC_MIN_S, TTC_MAX_S],
            "buckets": BUCKET_EDGES,
            "fill_price_range": [args.fill_min, args.fill_max],
            "calibrated": use_cal,
            "workers": args.workers,
        },
        "summary_all": s_all,
        "summary_up_only": s_up,
        "summary_down_only": s_dn,
        "entries": all_entries,
    }, indent=2))
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
