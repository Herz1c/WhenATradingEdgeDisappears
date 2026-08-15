"""Phase 1: walk every market across N OOS days, sample the model on a
fixed grid (every SAMPLE_MS ms) inside the [10s, 60s] TTC window, and emit
one row per QUALIFYING signal (passes EV + fill-price filters).

Output: data/datasets/poly_l2_only_v2_signals.parquet

Schema per row:
  date, market_slug, outcome ("Up"|"Down"),
  ts_ns, ttc_s, side ("UP"|"DOWN"), fill, p_model, ev_per_share

The grid sampling (default 250 ms) makes this tractable: ~200 prediction
checkpoints per market in the 50s window, vs ~25k if we evaluated at every
WS event. Cooldown is 10 s so 250 ms granularity is far finer than the
strategy actually needs.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
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

# Defaults — same filters we've been using in the bucketed backtest.
MIN_EDGE = 0.04
MIN_EV_FRAC = 0.06
TTC_MIN_S = 10.0
TTC_MAX_S = 60.0
FILL_MIN = 0.30
FILL_MAX = 0.70
SAMPLE_MS = 250
LOGIT_CLIP = (1e-4, 1 - 1e-4)


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# Worker globals.
_MODEL = None
_CALIB = None
_FEATURE_ORDER: List[str] = []
_USES_INIT = False
_MP_IDX = -1


def _worker_init(artifact_dir_str: str) -> None:
    global _MODEL, _CALIB, _FEATURE_ORDER, _USES_INIT, _MP_IDX
    art = Path(artifact_dir_str)
    _MODEL = joblib.load(art / "model.pkl")
    _USES_INIT = bool(getattr(_MODEL, "_uses_init_score", False))
    _FEATURE_ORDER = orjson.loads((art / "features.json").read_bytes())
    _MP_IDX = _FEATURE_ORDER.index("implied_p_up")
    cal_path = art / "calibrator.pkl"
    _CALIB = joblib.load(cal_path) if cal_path.exists() else getattr(_MODEL, "_calibrator", None)


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


def _predict_p_up(X: np.ndarray) -> float:
    if _USES_INIT:
        mp_val = float(X[0, _MP_IDX])
        mp_clip = min(max(mp_val, LOGIT_CLIP[0]), LOGIT_CLIP[1])
        init_z = math.log(mp_clip / (1.0 - mp_clip))
        raw = float(_MODEL.predict(X, raw_score=True,
                                   num_iteration=_MODEL.best_iteration)[0])
        p = 1.0 / (1.0 + math.exp(-(init_z + raw)))
    else:
        p = float(_MODEL.predict(X, num_iteration=_MODEL.best_iteration)[0])
    if _CALIB is not None:
        p = float(_CALIB.transform([p])[0])
    return p


def _collect_one(args: Tuple[str, str, str]) -> List[dict]:
    """Per market: walk events, sample every SAMPLE_MS in [10,60]s ttc,
    emit qualifying signals (EV + fill-band filters). Returns list of rows."""
    path_str, outcome, date_str = args
    path = Path(path_str)
    state = MarketState()
    out: List[dict] = []
    last_predict_ts_ns = 0
    sample_ns = SAMPLE_MS * 1_000_000
    buf = np.zeros((1, len(_FEATURE_ORDER)), dtype=np.float32)
    win_start_ns = -1
    win_end_ns = -1
    slug = ""

    for frame in iter_frames(path):
        et = frame.get("event_type")
        if et not in EMIT_EVENT_TYPES:
            continue
        if not update_state(state, frame):
            continue

        # Initialize window once we have market metadata.
        if win_start_ns < 0 and state.market_close_s:
            win_end_ns = (state.market_close_s - int(TTC_MIN_S)) * NS_PER_S
            win_start_ns = (state.market_close_s - int(TTC_MAX_S)) * NS_PER_S
            slug = state.event_slug

        ts_ns = int(frame.get("recv_ts_ns") or 0)
        if win_start_ns < 0 or ts_ns < win_start_ns or ts_ns >= win_end_ns:
            continue
        # Sample at fixed cadence.
        if ts_ns - last_predict_ts_ns < sample_ns:
            continue
        last_predict_ts_ns = ts_ns

        feats = state_to_features(state, ts_ns)
        for i, c in enumerate(_FEATURE_ORDER):
            buf[0, i] = feats.get(c, 0.0)
        p_up = _predict_p_up(buf)

        ba_up = feats["up_best_ask"]
        ba_dn = feats["down_best_ask"]
        ttc_s = (state.market_close_s * NS_PER_S - ts_ns) / NS_PER_S

        # Check each side independently — strategy chooses which to take.
        if 0.0 < ba_up < 1.0:
            ev_up = p_up - ba_up
            if ev_up >= MIN_EDGE and (ev_up / ba_up) >= MIN_EV_FRAC \
                    and FILL_MIN <= ba_up <= FILL_MAX:
                out.append({
                    "date": date_str, "market_slug": slug, "outcome": outcome,
                    "ts_ns": ts_ns, "ttc_s": round(ttc_s, 3),
                    "side": "UP", "fill": round(ba_up, 4),
                    "p_model": round(p_up, 4), "ev": round(ev_up, 4),
                })
        if 0.0 < ba_dn < 1.0:
            ev_dn = (1.0 - p_up) - ba_dn
            if ev_dn >= MIN_EDGE and (ev_dn / ba_dn) >= MIN_EV_FRAC \
                    and FILL_MIN <= ba_dn <= FILL_MAX:
                out.append({
                    "date": date_str, "market_slug": slug, "outcome": outcome,
                    "ts_ns": ts_ns, "ttc_s": round(ttc_s, 3),
                    "side": "DOWN", "fill": round(ba_dn, 4),
                    "p_model": round(p_up, 4), "ev": round(ev_dn, 4),
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default="artifacts_cleaned/poly_l2_only_v2")
    ap.add_argument("--dates", default="2026-04-22,2026-04-30,2026-05-07,"
                                       "2026-05-15,2026-05-21,2026-05-28,2026-06-01")
    ap.add_argument("--n-markets", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default="data/datasets/poly_l2_only_v2_signals.parquet")
    args = ap.parse_args()

    art_dir = (REPO_ROOT / args.artifact_dir).resolve()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    print(f"[{ts()}] artifact: {art_dir}", flush=True)
    print(f"[{ts()}] dates: {dates}", flush=True)
    res_map = load_resolution_map(RESOLUTION_ROOT, verbose=False)
    print(f"[{ts()}] resolution slugs = {len(res_map):,}", flush=True)

    tasks: List[Tuple[str, str, str]] = []
    for d in dates:
        day_dir = DATA_RAW / d
        files = sorted(day_dir.glob("*.l2.jsonl.zst"))[: args.n_markets]
        for fp in files:
            slug = fp.name.split("__")[1]
            outcome = res_map.get(slug)
            if outcome not in ("Up", "Down"):
                continue
            tasks.append((str(fp), outcome, d))
    print(f"[{ts()}] {len(tasks)} markets queued, workers={args.workers}", flush=True)
    print(f"[{ts()}] sample={SAMPLE_MS}ms, filters min_edge=${MIN_EDGE}, "
          f"min_ev_frac={MIN_EV_FRAC*100:.1f}%, fill=[${FILL_MIN},${FILL_MAX}]",
          flush=True)

    t0 = time.time()
    all_signals: List[dict] = []
    with mp.Pool(processes=args.workers, initializer=_worker_init,
                 initargs=(str(art_dir),), maxtasksperchild=50) as pool:
        done = 0
        for sigs in pool.imap_unordered(_collect_one, tasks, chunksize=2):
            done += 1
            all_signals.extend(sigs)
            if done % 100 == 0:
                print(f"[{ts()}]   {done}/{len(tasks)} markets, "
                      f"signals so far = {len(all_signals):,}", flush=True)
    dt = time.time() - t0
    print(f"[{ts()}] collected {len(all_signals):,} signals in {dt:.1f}s "
          f"({dt/max(1,len(tasks)):.2f}s/market)", flush=True)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not all_signals:
        print("[ERROR] no signals collected. Aborting save.")
        return
    schema = pa.schema([
        pa.field("date", pa.string()),
        pa.field("market_slug", pa.string()),
        pa.field("outcome", pa.string()),
        pa.field("ts_ns", pa.int64()),
        pa.field("ttc_s", pa.float32()),
        pa.field("side", pa.string()),
        pa.field("fill", pa.float32()),
        pa.field("p_model", pa.float32()),
        pa.field("ev", pa.float32()),
    ])
    table = pa.Table.from_pylist(all_signals, schema=schema)
    pq.write_table(table, out_path, compression="zstd", compression_level=3)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[{ts()}] wrote {out_path} ({size_mb:.1f} MB, {table.num_rows:,} rows)", flush=True)


if __name__ == "__main__":
    main()
