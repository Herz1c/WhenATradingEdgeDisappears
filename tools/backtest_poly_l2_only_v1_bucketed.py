"""Bucketed backtest of poly_l2_only_v1.

Differences vs the previous quick backtest:
  1. Per market we partition the [10s, 60s] window into 5 TTC buckets of 10s
     each. Within each bucket we scan ALL qualifying ticks and take the single
     one with the largest EV-per-share. So each market contributes at most one
     entry per bucket (i.e. up to 5 entries per market). This produces a
     roughly balanced sample across TTC regimes.
  2. Stricter default thresholds so we only act on genuinely strong signals
     (the bot was forcing entries before — almost every market triggered).
  3. Reports headline PnL twice: all entries, and UP-only (banning DOWN buys).

Decision per tick:
  ev_up   = p_model      - best_ask_up       (USD profit / share if we buy UP)
  ev_down = (1 - p_model) - best_ask_down    (USD profit / share if we buy DOWN)
  Qualify if max(ev) >= MIN_EDGE_PER_SHARE AND ev/fill >= MIN_EV_FRAC.

Within each bucket, retain the tick with the highest qualifying EV-per-share.
"""
from __future__ import annotations

import argparse
import json
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
ARTIFACT = REPO_ROOT / "artifacts_cleaned" / "poly_l2_only_v1"

# Trading params (stricter than the quick run).
NOTIONAL_USD = 5.0
MIN_EDGE_PER_SHARE = 0.04   # 4 cents per share minimum
MIN_EV_FRAC = 0.06          # 6% expected return on cost
TTC_MIN_S = 10.0
TTC_MAX_S = 60.0
COMMISSION_BPS = 0.0
BUCKET_EDGES = [(50, 60), (40, 50), (30, 40), (20, 30), (10, 20)]
# Fill-price window to avoid long-shot calibration bias.
FILL_PRICE_MIN = 0.30
FILL_PRICE_MAX = 0.70
# Use isotonic-calibrated probabilities (model._calibrator from training).
USE_CALIBRATED = True


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


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


def backtest_one_market(path: Path, model, calibrator, feature_order: List[str],
                        outcome: str, buf: np.ndarray,
                        min_edge: float, min_ev_frac: float,
                        fill_min: float, fill_max: float) -> List[dict]:
    """Walk events; first qualifying tick in each TTC bucket triggers the
    entry. Once a bucket is filled, skip predictions for further ticks in
    that bucket. Once all 5 buckets are filled, stop reading the file.

    A tick "qualifies" if BOTH:
      - EV-per-share >= min_edge AND EV/cost >= min_ev_frac, AND
      - fill price is in [fill_min, fill_max] (avoid long-shot bias)
    """
    state = MarketState()
    entries_by_bucket: Dict[int, dict] = {}
    n_buckets = len(BUCKET_EDGES)
    best_iter = model.best_iteration

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
        # Skip predictions in buckets already filled — the big speedup.
        if b in entries_by_bucket:
            continue

        feats = state_to_features(state, ts_ns)
        for i, c in enumerate(feature_order):
            buf[0, i] = feats.get(c, 0.0)
        p_up_raw = float(model.predict(buf, num_iteration=best_iter)[0])
        if calibrator is not None:
            p_up = float(calibrator.transform([p_up_raw])[0])
        else:
            p_up = p_up_raw

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

        # Fill-price gate: skip long-shot and near-certain bets (calibration
        # bias is worst there).
        if not (fill_min <= fill <= fill_max):
            continue

        entries_by_bucket[b] = {
            "slug": state.event_slug,
            "ttc_s": round(ttc_s, 2),
            "bucket": b,
            "side": side,
            "fill_price": round(fill, 4),
            "p_model_raw": round(p_up_raw, 4),
            "p_model": round(p_up, 4),
            "ev_per_share": round(ev, 4),
            "outcome": outcome,
        }
        if len(entries_by_bucket) >= n_buckets:
            break

    # Settle.
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


def summarize(entries: List[dict], label: str, notional: float) -> dict:
    if not entries:
        print(f"=== {label} === (no entries)")
        return {"label": label, "n_entries": 0}
    n = len(entries)
    n_wins = sum(e["won"] for e in entries)
    total_pnl = sum(e["pnl_usd"] for e in entries)
    nominal = notional * n
    avg_pnl = total_pnl / n
    win_rate = n_wins / n
    roi = total_pnl / nominal
    print(f"=== {label} ===")
    print(f"  entries          : {n}")
    print(f"  notional         : ${nominal:,.2f}")
    print(f"  wins             : {n_wins}  ({100*win_rate:.2f}%)")
    print(f"  TOTAL PnL        : ${total_pnl:+,.2f}")
    print(f"  ROI on notional  : {100*roi:+.2f}%")
    print(f"  avg PnL / trade  : ${avg_pnl:+.4f}")
    # Per-bucket
    by_bucket = defaultdict(list)
    for e in entries:
        by_bucket[e["bucket"]].append(e)
    print(f"  PnL by TTC bucket:")
    for b, (lo, hi) in enumerate(BUCKET_EDGES):
        bs = by_bucket.get(b, [])
        if not bs:
            print(f"    [{lo},{hi})s: n=  0 (no entries)")
            continue
        bn = len(bs)
        bp = sum(e["pnl_usd"] for e in bs)
        bw = sum(e["won"] for e in bs)
        avg_fill = float(np.mean([e["fill_price"] for e in bs]))
        avg_ev = float(np.mean([e["ev_per_share"] for e in bs]))
        print(f"    [{lo},{hi})s: n={bn:>3}  pnl=${bp:+8.2f}  avg=${bp/bn:+.4f}  "
              f"wr={100*bw/bn:5.1f}%  fill={avg_fill:.3f}  ev/sh={avg_ev:+.3f}")
    # Side breakdown
    by_side = defaultdict(list)
    for e in entries:
        by_side[e["side"]].append(e)
    for sd, es in by_side.items():
        bp = sum(e["pnl_usd"] for e in es)
        bw = sum(e["won"] for e in es)
        print(f"  side={sd:4s}: n={len(es):>3}  pnl=${bp:+8.2f}  wr={100*bw/len(es):.1f}%")
    print()
    return {
        "label": label, "n_entries": n, "total_pnl": total_pnl, "roi": roi,
        "win_rate": win_rate, "avg_pnl": avg_pnl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-28")
    ap.add_argument("--n-markets", type=int, default=200)
    ap.add_argument("--min-edge", type=float, default=MIN_EDGE_PER_SHARE)
    ap.add_argument("--min-ev-frac", type=float, default=MIN_EV_FRAC)
    ap.add_argument("--fill-min", type=float, default=FILL_PRICE_MIN)
    ap.add_argument("--fill-max", type=float, default=FILL_PRICE_MAX)
    ap.add_argument("--no-calibrator", action="store_true",
                    help="Use raw model output instead of isotonic-calibrated.")
    args = ap.parse_args()

    print(f"[{ts()}] loading model + resolution ...", flush=True)
    model = joblib.load(ARTIFACT / "model.pkl")
    feature_order = json.loads((ARTIFACT / "features.json").read_text())
    res_map = load_resolution_map(RESOLUTION_ROOT, verbose=False)

    if args.no_calibrator:
        calibrator = None
        cal_msg = "RAW predictions"
    else:
        # Load standalone calibrator (more reliable than relying on the
        # _calibrator attribute surviving joblib roundtrip).
        cal_path = ARTIFACT / "calibrator.pkl"
        calibrator = joblib.load(cal_path) if cal_path.exists() else getattr(model, "_calibrator", None)
        cal_msg = "ISOTONIC-CALIBRATED predictions" if calibrator is not None else "RAW (no calibrator found)"

    day_dir = DATA_RAW / args.date
    files = sorted(day_dir.glob("*.l2.jsonl.zst"))[: args.n_markets]
    print(f"[{ts()}] {args.date}: {len(files)} files queued",
          f"| min_edge=${args.min_edge}", f"| min_ev_frac={args.min_ev_frac*100:.1f}%",
          f"| fills in [${args.fill_min:.2f}, ${args.fill_max:.2f}]",
          f"| using {cal_msg}",
          flush=True)
    print(f"[{ts()}] up to one entry per market per TTC bucket "
          f"({len(BUCKET_EDGES)} buckets, max {len(files)*len(BUCKET_EDGES)} trades)",
          flush=True)

    buf = np.zeros((1, len(feature_order)), dtype=np.float32)
    all_entries: List[dict] = []
    n_no_label = 0
    n_no_entry = 0
    t0 = time.time()

    for i, fp in enumerate(files):
        slug = fp.name.split("__")[1]
        outcome = res_map.get(slug)
        if outcome not in ("Up", "Down"):
            n_no_label += 1
            continue
        entries = backtest_one_market(fp, model, calibrator, feature_order,
                                      outcome, buf, args.min_edge, args.min_ev_frac,
                                      args.fill_min, args.fill_max)
        if not entries:
            n_no_entry += 1
        else:
            all_entries.extend(entries)
        if (i + 1) % 50 == 0:
            print(f"[{ts()}]   {i+1}/{len(files)} processed, total_entries={len(all_entries)}",
                  flush=True)
    dt = time.time() - t0
    print(f"[{ts()}] backtest done in {dt:.1f}s", flush=True)
    print(f"[{ts()}] {len(files)} markets queued | no_label={n_no_label} | "
          f"no_entry={n_no_entry} | total_entries={len(all_entries)}", flush=True)
    print()

    # --- summary 1: all entries (both sides) ---
    s_all = summarize(all_entries, "ALL SIDES (UP+DOWN)", NOTIONAL_USD)

    # --- summary 2: UP-only (ban DOWN buys) ---
    up_only = [e for e in all_entries if e["side"] == "UP"]
    s_up = summarize(up_only, "UP-ONLY (DOWN buying banned)", NOTIONAL_USD)

    # --- summary 3: DOWN-only for completeness ---
    dn_only = [e for e in all_entries if e["side"] == "DOWN"]
    if dn_only:
        s_dn = summarize(dn_only, "DOWN-ONLY (for diagnosis)", NOTIONAL_USD)
    else:
        s_dn = {"label": "DOWN-ONLY (for diagnosis)", "n_entries": 0}

    # Save
    out_dir = ARTIFACT / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bucketed_{args.date}.json"
    out_path.write_text(json.dumps({
        "date": args.date,
        "n_markets_queued": len(files),
        "n_no_label": n_no_label,
        "n_no_entry": n_no_entry,
        "n_entries": len(all_entries),
        "params": {
            "notional": NOTIONAL_USD,
            "min_edge_per_share": args.min_edge,
            "min_ev_frac": args.min_ev_frac,
            "ttc_range": [TTC_MIN_S, TTC_MAX_S],
            "buckets": BUCKET_EDGES,
            "fill_price_range": [args.fill_min, args.fill_max],
            "calibrated": not args.no_calibrator,
        },
        "summary_all": s_all,
        "summary_up_only": s_up,
        "summary_down_only": s_dn,
        "entries": all_entries,
    }, indent=2))
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
