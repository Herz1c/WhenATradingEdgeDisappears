"""Quick backtest of poly_l2_only_v1: $5 per market, one entry per market,
first qualifying tick wins, move to next market on entry.

Decision rule per tick:
  ev_up    = p_model - best_ask_up              (USD profit per share buying UP)
  ev_down  = (1 - p_model) - best_ask_down      (USD profit per share buying DOWN)
  - require ev > MIN_EDGE (cents per share) AND > MIN_EV_FRAC * cost
  - pick whichever side has the larger ev > 0
  - skip if best_ask <= 0 or best_ask >= 1 (broken book) or ttc not in [10, 60]

Fill: cross the spread, pay best_ask. Shares = NOTIONAL / best_ask.
Settle at market resolution: payout = shares * $1 if we picked the winner, else 0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import orjson
import zstandard as zstd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from poly_l2_only.extractor import (  # noqa: E402
    EMIT_EVENT_TYPES, FEATURE_COLUMNS, MarketState, NS_PER_S,
    state_to_features, update_state,
)
from poly_l2_only.resolution import load_resolution_map  # noqa: E402

DATA_RAW = REPO_ROOT / "data" / "raw" / "polymarket" / "btc_updown_5m"
RESOLUTION_ROOT = REPO_ROOT / "data" / "raw" / "polymarket" / "resolution" / "btc_updown_5m"
ARTIFACT = REPO_ROOT / "artifacts_cleaned" / "poly_l2_only_v1"

# Trading params.
NOTIONAL_USD = 5.0
MIN_EDGE_PER_SHARE = 0.02   # require >=2 cents EV per share (~3-4% on $0.5 token)
MIN_EV_FRAC = 0.03          # require expected return >= 3% on cost
TTC_MIN_S = 10.0
TTC_MAX_S = 60.0
COMMISSION_BPS = 0.0        # Polymarket: nominally 0


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


def predict_one(model, feats_dict: Dict[str, float], feature_order: List[str],
                buf: np.ndarray) -> float:
    for i, c in enumerate(feature_order):
        buf[0, i] = feats_dict.get(c, 0.0)
    return float(model.predict(buf, num_iteration=model.best_iteration)[0])


def backtest_one_market(path: Path, model, feature_order: List[str],
                        outcome: str, buf: np.ndarray) -> Optional[dict]:
    """Walk events in order; first qualifying signal triggers entry; settle."""
    state = MarketState()
    entry = None
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
        feats = state_to_features(state, ts_ns)
        p_up = predict_one(model, feats, feature_order, buf)

        ba_up = feats["up_best_ask"]
        ba_dn = feats["down_best_ask"]

        ev_up = p_up - ba_up if (ba_up > 0.0 and ba_up < 1.0) else -1.0
        ev_dn = (1.0 - p_up) - ba_dn if (ba_dn > 0.0 and ba_dn < 1.0) else -1.0

        # Pick best qualifying side.
        side = None
        if ev_up >= MIN_EDGE_PER_SHARE and ev_up >= ev_dn and (ev_up / ba_up) >= MIN_EV_FRAC:
            side = "UP"
            fill = ba_up
            ev = ev_up
        elif ev_dn >= MIN_EDGE_PER_SHARE and ev_dn > ev_up and (ev_dn / ba_dn) >= MIN_EV_FRAC:
            side = "DOWN"
            fill = ba_dn
            ev = ev_dn
        if side is None:
            continue

        shares = NOTIONAL_USD / fill
        fee = NOTIONAL_USD * COMMISSION_BPS / 1e4
        won = (side == "UP" and outcome == "Up") or (side == "DOWN" and outcome == "Down")
        payout = shares * 1.0 if won else 0.0
        pnl = payout - NOTIONAL_USD - fee

        entry = {
            "slug": state.event_slug,
            "ttc_s": round(ttc_s, 2),
            "side": side,
            "fill_price": round(fill, 4),
            "p_model": round(p_up, 4),
            "ev_per_share": round(ev, 4),
            "shares": round(shares, 4),
            "outcome": outcome,
            "won": int(won),
            "pnl_usd": round(pnl, 4),
        }
        break  # one entry per market

    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-28",
                    help="Day to backtest (default = 2026-05-28, OOS).")
    ap.add_argument("--n-markets", type=int, default=200)
    ap.add_argument("--min-edge", type=float, default=None)
    ap.add_argument("--min-ev-frac", type=float, default=None)
    args = ap.parse_args()

    global MIN_EDGE_PER_SHARE, MIN_EV_FRAC
    if args.min_edge is not None:
        MIN_EDGE_PER_SHARE = args.min_edge
    if args.min_ev_frac is not None:
        MIN_EV_FRAC = args.min_ev_frac

    print(f"[{ts()}] loading model + resolution ...", flush=True)
    model = joblib.load(ARTIFACT / "model.pkl")
    feature_order = json.loads((ARTIFACT / "features.json").read_text())
    res_map = load_resolution_map(RESOLUTION_ROOT, verbose=False)

    day_dir = DATA_RAW / args.date
    files = sorted(day_dir.glob("*.l2.jsonl.zst"))[: args.n_markets]
    print(f"[{ts()}] {args.date}: {len(files)} files queued, model best_iter={model.best_iteration}",
          flush=True)
    print(f"[{ts()}] params: notional=${NOTIONAL_USD}, min_edge=${MIN_EDGE_PER_SHARE}, "
          f"min_ev_frac={MIN_EV_FRAC*100:.1f}%, ttc=[{TTC_MIN_S},{TTC_MAX_S}]s",
          flush=True)

    buf = np.zeros((1, len(feature_order)), dtype=np.float32)
    entries: List[dict] = []
    n_no_label = 0
    n_no_entry = 0
    t0 = time.time()

    for i, fp in enumerate(files):
        slug = fp.name.split("__")[1]
        outcome = res_map.get(slug)
        if outcome not in ("Up", "Down"):
            n_no_label += 1
            continue
        entry = backtest_one_market(fp, model, feature_order, outcome, buf)
        if entry is None:
            n_no_entry += 1
        else:
            entries.append(entry)
        if (i + 1) % 50 == 0:
            print(f"[{ts()}]   {i+1}/{len(files)} processed, entries={len(entries)}", flush=True)
    dt = time.time() - t0
    print(f"[{ts()}] backtest pass done in {dt:.1f}s", flush=True)
    print()

    # Aggregate.
    n_eligible = len(files) - n_no_label
    n_entries = len(entries)
    n_wins = sum(e["won"] for e in entries)
    total_pnl = sum(e["pnl_usd"] for e in entries)
    notional = NOTIONAL_USD * n_entries
    avg_pnl = total_pnl / n_entries if n_entries else 0.0
    win_rate = n_wins / n_entries if n_entries else 0.0
    roi = total_pnl / notional if notional else 0.0
    avg_ttc = float(np.mean([e["ttc_s"] for e in entries])) if entries else 0.0
    avg_fill = float(np.mean([e["fill_price"] for e in entries])) if entries else 0.0
    avg_ev = float(np.mean([e["ev_per_share"] for e in entries])) if entries else 0.0
    pnl_by_side = Counter()
    n_by_side = Counter()
    for e in entries:
        pnl_by_side[e["side"]] += e["pnl_usd"]
        n_by_side[e["side"]] += 1

    print("=" * 60)
    print(f"BACKTEST RESULT — {args.date} — {len(files)} markets queued")
    print("=" * 60)
    print(f"  files queued        : {len(files)}")
    print(f"  no resolution label : {n_no_label}")
    print(f"  no entry triggered  : {n_no_entry}")
    print(f"  ENTRIES TAKEN       : {n_entries}  ({100*n_entries/n_eligible:.1f}% of eligible)")
    print(f"  notional deployed   : ${notional:,.2f}")
    print(f"  winning trades      : {n_wins}  ({100*win_rate:.2f}%)")
    print(f"  TOTAL PnL           : ${total_pnl:+,.2f}")
    print(f"  ROI on notional     : {100*roi:+.2f}%")
    print(f"  avg PnL / trade     : ${avg_pnl:+.4f}")
    print(f"  avg TTC at entry    : {avg_ttc:.1f}s")
    print(f"  avg fill price      : {avg_fill:.4f}")
    print(f"  avg EV per share    : ${avg_ev:+.4f}")
    print()
    for sd in ("UP", "DOWN"):
        n = n_by_side[sd]
        if not n:
            continue
        print(f"  side={sd}: n={n}, pnl=${pnl_by_side[sd]:+.2f}, "
              f"avg=${pnl_by_side[sd]/n:+.4f}")

    # Distribution of TTC at entry (5s buckets).
    if entries:
        ttcs = np.array([e["ttc_s"] for e in entries])
        pnls = np.array([e["pnl_usd"] for e in entries])
        print()
        print("  PnL by TTC bucket at entry:")
        for lo, hi in [(50, 60), (40, 50), (30, 40), (20, 30), (10, 20)]:
            m = (ttcs >= lo) & (ttcs < hi)
            if not m.any():
                continue
            print(f"    [{lo},{hi})s: n={int(m.sum()):>3}  "
                  f"pnl=${pnls[m].sum():+7.2f}  avg=${pnls[m].mean():+.4f}  "
                  f"winrate={float((pnls[m] > 0).mean())*100:.1f}%")

    # Save run.
    out_dir = REPO_ROOT / "artifacts_cleaned" / "poly_l2_only_v1" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"quick_{args.date}.json"
    out_path.write_text(json.dumps({
        "date": args.date,
        "n_markets_queued": len(files),
        "n_no_label": n_no_label,
        "n_no_entry": n_no_entry,
        "n_entries": n_entries,
        "n_wins": n_wins,
        "total_pnl_usd": total_pnl,
        "roi": roi,
        "win_rate": win_rate,
        "params": {
            "notional": NOTIONAL_USD,
            "min_edge_per_share": MIN_EDGE_PER_SHARE,
            "min_ev_frac": MIN_EV_FRAC,
            "ttc_range": [TTC_MIN_S, TTC_MAX_S],
        },
        "entries": entries,
    }, indent=2))
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
