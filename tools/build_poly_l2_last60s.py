"""Build the polymarket_l2_last60s_v1 training dataset.

Streams every recorded .l2.jsonl.zst, replays each frame through
poly_l2_only.extractor, and emits one row per L2 event whose recv_ts
falls in [market_close - 60s, market_close - 10s].

Inputs (Polymarket-only, by design):
  data/raw/polymarket/btc_updown_5m/<YYYY-MM-DD>/<HH-MM-SS>__<slug>__<id>.l2.jsonl.zst
  data/raw/polymarket/resolution/btc_updown_5m/<YYYY-MM-DD>/<HH>.resolution.jsonl.zst

Output:
  data/datasets/polymarket_l2_last60s_v1/<YYYY-MM-DD>.parquet
  data/datasets/polymarket_l2_last60s_v1/_manifest.json

Memory plan for 16 GB host:
  - Decompress zstd streamed in 64 KB chunks (constant memory per file).
  - One file = one market = ~25k events in window. Buffer per file, write to
    parquet immediately. Never hold more than one market's rows.
  - Use pyarrow ParquetWriter to append row groups per market.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

# Make src importable.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from poly_l2_only.extractor import (  # noqa: E402
    FEATURE_COLUMNS,
    MarketState,
    NS_PER_S,
    EMIT_EVENT_TYPES,
    state_to_features,
    update_state,
)
from poly_l2_only.resolution import load_resolution_map  # noqa: E402


# ----- config ---------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_L2_ROOT = REPO_ROOT / "data" / "raw" / "polymarket" / "btc_updown_5m"
RESOLUTION_ROOT = REPO_ROOT / "data" / "raw" / "polymarket" / "resolution" / "btc_updown_5m"
OUT_ROOT = REPO_ROOT / "data" / "datasets" / "polymarket_l2_last60s_v1"

WINDOW_START_OFFSET_S = 60   # include events with ttc <= 60
WINDOW_END_OFFSET_S = 10     # include events with ttc >  10  (stop at ttc=10)

# OOS test dates: well-populated days spread across the corpus.
DEFAULT_OOS_DATES = frozenset({
    "2026-04-22",
    "2026-04-30",
    "2026-05-07",
    "2026-05-15",
    "2026-05-21",
    "2026-05-28",
    "2026-06-01",
})

EVENT_TYPE_IDX: Dict[str, int] = {et: i for i, et in enumerate(EMIT_EVENT_TYPES)}


# ----- file parsing ---------------------------------------------------------

FILENAME_RE = re.compile(
    r"^(?P<hms>\d{2}-\d{2}-\d{2})__(?P<slug>btc-updown-5m-(?P<unix>\d+))__(?P<id>[0-9a-f]+)\.l2\.jsonl\.zst$"
)


def parse_l2_filename(name: str) -> Optional[dict]:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    return {"slug": g["slug"], "market_open_s": int(g["unix"]), "id": g["id"]}


def iter_frames(path: Path) -> Iterator[dict]:
    """Stream JSONL records out of a .zst file with bounded memory."""
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
                # find newlines
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
            if buf.strip():
                try:
                    yield orjson.loads(bytes(buf))
                except Exception:
                    pass


# ----- per-market processing ------------------------------------------------

def process_market_file(
    path: Path,
    resolution_map: Dict[str, str],
    oos_dates: frozenset,
    date_str: str,
) -> Optional[Dict[str, list]]:
    """Returns a column-store dict (column_name -> list of values) for one
    market, or None if the market should be skipped (no resolution / no events
    in window / no metadata)."""
    meta = parse_l2_filename(path.name)
    if not meta:
        return None
    slug = meta["slug"]
    outcome = resolution_map.get(slug)
    if outcome not in ("Up", "Down"):
        return None  # no label -> drop

    # window cutoffs from filename (no need to wait for first frame metadata)
    market_open_s = meta["market_open_s"]
    market_close_s = market_open_s + 300
    win_start_ns = (market_close_s - WINDOW_START_OFFSET_S) * NS_PER_S
    win_end_ns = (market_close_s - WINDOW_END_OFFSET_S) * NS_PER_S

    state = MarketState()

    label_up = 1 if outcome == "Up" else 0
    split = "test" if date_str in oos_dates else "train"

    # Pre-allocate column buffers.
    cols: Dict[str, list] = {c: [] for c in FEATURE_COLUMNS}
    cols["recv_ts_ns"] = []
    cols["event_type"] = []
    cols["market_slug"] = []
    cols["market_close_s"] = []
    cols["date"] = []
    cols["split"] = []
    cols["label_up"] = []

    gen = iter_frames(path)
    while True:
        try:
            frame = next(gen)
        except StopIteration:
            break
        except Exception:
            # Corrupted zst tail or bad line — stop reading this market.
            break
        et = frame.get("event_type")
        if et not in EMIT_EVENT_TYPES:
            continue
        # Update state on every emit-capable frame, always (so features at
        # window entry reflect the full pre-window history we recorded).
        ok = update_state(state, frame)
        if not ok:
            continue
        ts_ns = int(frame.get("recv_ts_ns") or 0)
        if ts_ns < win_start_ns or ts_ns >= win_end_ns:
            continue
        feats = state_to_features(state, ts_ns)
        for c in FEATURE_COLUMNS:
            cols[c].append(feats.get(c, 0.0))
        cols["recv_ts_ns"].append(ts_ns)
        cols["event_type"].append(EVENT_TYPE_IDX[et])
        cols["market_slug"].append(slug)
        cols["market_close_s"].append(market_close_s)
        cols["date"].append(date_str)
        cols["split"].append(split)
        cols["label_up"].append(label_up)

    if not cols["recv_ts_ns"]:
        return None
    return cols


# ----- parquet schema -------------------------------------------------------

def build_schema() -> pa.Schema:
    fields: List[pa.Field] = [pa.field(c, pa.float32()) for c in FEATURE_COLUMNS]
    fields += [
        pa.field("recv_ts_ns", pa.int64()),
        pa.field("event_type", pa.int8()),
        pa.field("market_slug", pa.string()),
        pa.field("market_close_s", pa.int64()),
        pa.field("date", pa.string()),
        pa.field("split", pa.string()),
        pa.field("label_up", pa.int8()),
    ]
    return pa.schema(fields)


# Worker globals (initialized in _worker_init via pool initializer).
_WORKER_RES_MAP: Dict[str, str] = {}
_WORKER_OOS: frozenset = frozenset()
_WORKER_SCHEMA: Optional[pa.Schema] = None


def _worker_init(res_map: Dict[str, str], oos_dates: frozenset) -> None:
    global _WORKER_RES_MAP, _WORKER_OOS, _WORKER_SCHEMA
    _WORKER_RES_MAP = res_map
    _WORKER_OOS = oos_dates
    _WORKER_SCHEMA = build_schema()


def _worker_process_one(args: Tuple[str, str]) -> Tuple[str, Optional[bytes], int, int, int]:
    """Process one market file; return (slug, serialized_table_or_None,
    n_rows, label_up, is_oos_flag).
    Tables are serialized to Arrow IPC stream bytes for fast cross-process
    transport without expensive pickling."""
    path_str, date_str = args
    path = Path(path_str)
    cols = process_market_file(path, _WORKER_RES_MAP, _WORKER_OOS, date_str)
    if cols is None:
        return (path.name, None, 0, 0, 0)
    table = cols_to_table(cols, _WORKER_SCHEMA)
    # Serialize to IPC stream (zero-copy on receive, no pickling).
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    payload = sink.getvalue().to_pybytes()
    label_up = int(cols["label_up"][0])
    is_oos = 1 if date_str in _WORKER_OOS else 0
    return (path.name, payload, table.num_rows, label_up, is_oos)


def cols_to_table(cols: Dict[str, list], schema: pa.Schema) -> pa.Table:
    arrays = []
    for f in schema:
        if f.type == pa.float32():
            arrays.append(pa.array(cols[f.name], type=pa.float32()))
        elif f.type == pa.int64():
            arrays.append(pa.array(cols[f.name], type=pa.int64()))
        elif f.type == pa.int8():
            arrays.append(pa.array(cols[f.name], type=pa.int8()))
        else:
            arrays.append(pa.array(cols[f.name], type=f.type))
    return pa.Table.from_arrays(arrays, schema=schema)


# ----- main runner ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-days", type=int, default=0,
                    help="If > 0, only process the first N days (debug).")
    ap.add_argument("--limit-markets-per-day", type=int, default=0,
                    help="If > 0, only process the first N markets per day (debug).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute one market end-to-end and print summary.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="Worker processes (default: CPU-1).")
    ap.add_argument("--serial", action="store_true",
                    help="Disable multiprocessing (debug).")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[{ts()}] loading resolution map from {RESOLUTION_ROOT}", flush=True)
    res_map = load_resolution_map(RESOLUTION_ROOT, verbose=True)
    print(f"[{ts()}] resolution_map size = {len(res_map)} slugs", flush=True)

    schema = build_schema()
    day_dirs = sorted(d for d in RAW_L2_ROOT.iterdir() if d.is_dir())
    if args.limit_days:
        day_dirs = day_dirs[: args.limit_days]

    summary = {
        "days": [],
        "totals": {"markets_processed": 0, "markets_skipped_no_label": 0,
                   "markets_skipped_empty": 0, "rows": 0,
                   "train_rows": 0, "test_rows": 0,
                   "train_markets": 0, "test_markets": 0,
                   "label_up_rows": 0},
        "oos_dates": sorted(DEFAULT_OOS_DATES),
        "window_offset_s": [WINDOW_START_OFFSET_S, WINDOW_END_OFFSET_S],
        "feature_columns": list(FEATURE_COLUMNS),
        "started_at": ts(),
    }

    # Build pool once, reuse across all days. maxtasksperchild caps memory drift.
    pool: Optional[mp.pool.Pool] = None
    if not args.serial and not args.dry_run:
        pool = mp.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(res_map, DEFAULT_OOS_DATES),
            maxtasksperchild=200,
        )
        print(f"[{ts()}] worker pool: {args.workers} processes", flush=True)

    for ddir in day_dirs:
        date_str = ddir.name
        files = sorted(ddir.glob("*.l2.jsonl.zst"))
        if args.limit_markets_per_day:
            files = files[: args.limit_markets_per_day]
        if not files:
            continue

        out_path = OUT_ROOT / f"{date_str}.parquet"
        if out_path.exists() and not args.dry_run:
            print(f"[{ts()}] {date_str}: parquet exists, skipping", flush=True)
            continue

        t0 = time.time()
        markets_ok = 0
        markets_no_label = 0
        markets_empty = 0
        rows_day = 0
        train_rows_day = 0
        test_rows_day = 0
        label_up_rows_day = 0
        is_oos = date_str in DEFAULT_OOS_DATES

        writer: Optional[pq.ParquetWriter] = None
        try:
            if args.dry_run or args.serial:
                # Sequential path (debugging).
                for i, fp in enumerate(files):
                    cols = process_market_file(fp, res_map, DEFAULT_OOS_DATES, date_str)
                    if cols is None:
                        parsed = parse_l2_filename(fp.name)
                        if not parsed or parsed["slug"] not in res_map:
                            markets_no_label += 1
                        else:
                            markets_empty += 1
                        continue
                    table = cols_to_table(cols, schema)
                    if writer is None and not args.dry_run:
                        writer = pq.ParquetWriter(out_path, schema,
                                                  compression="zstd", compression_level=3,
                                                  use_dictionary=True)
                    if writer is not None:
                        writer.write_table(table)
                    markets_ok += 1
                    n = table.num_rows
                    rows_day += n
                    if is_oos:
                        test_rows_day += n
                    else:
                        train_rows_day += n
                    label_up_rows_day += int(cols["label_up"][0]) * n
                    if args.dry_run:
                        print(f"  dry-run {fp.name}: {n} rows, label_up={cols['label_up'][0]}")
                        return
                    if (i + 1) % 50 == 0:
                        print(f"[{ts()}] {date_str}: {i+1}/{len(files)} files, "
                              f"{rows_day:,} rows, RSS={rss_mb():.0f} MB", flush=True)
            else:
                # Parallel path.
                assert pool is not None
                tasks = [(str(fp), date_str) for fp in files]
                # imap_unordered for streaming results; chunksize 1 keeps memory
                # tight (large per-market payloads, ~1-3 MB each).
                done = 0
                for fname, payload, n, label_up, _oos in pool.imap_unordered(
                        _worker_process_one, tasks, chunksize=1):
                    done += 1
                    if payload is None:
                        parsed = parse_l2_filename(fname)
                        if not parsed or parsed["slug"] not in res_map:
                            markets_no_label += 1
                        else:
                            markets_empty += 1
                    else:
                        # Deserialize IPC bytes to Arrow Table (zero-copy).
                        reader = pa.ipc.open_stream(pa.BufferReader(payload))
                        table = reader.read_all()
                        if writer is None:
                            writer = pq.ParquetWriter(out_path, schema,
                                                      compression="zstd", compression_level=3,
                                                      use_dictionary=True)
                        writer.write_table(table)
                        markets_ok += 1
                        rows_day += n
                        if is_oos:
                            test_rows_day += n
                        else:
                            train_rows_day += n
                        label_up_rows_day += label_up * n
                        del table, payload
                    if done % 50 == 0:
                        print(f"[{ts()}] {date_str}: {done}/{len(files)} files, "
                              f"{rows_day:,} rows, RSS={rss_mb():.0f} MB", flush=True)
        finally:
            if writer is not None:
                writer.close()

        dt = time.time() - t0
        size_mb = (out_path.stat().st_size / 1e6) if out_path.exists() else 0.0
        day_rec = {
            "date": date_str,
            "files": len(files),
            "markets_written": markets_ok,
            "markets_skipped_no_label": markets_no_label,
            "markets_skipped_empty": markets_empty,
            "rows": rows_day,
            "train_rows": train_rows_day,
            "test_rows": test_rows_day,
            "is_oos_day": is_oos,
            "parquet_mb": round(size_mb, 1),
            "elapsed_s": round(dt, 1),
        }
        summary["days"].append(day_rec)
        summary["totals"]["markets_processed"] += markets_ok
        summary["totals"]["markets_skipped_no_label"] += markets_no_label
        summary["totals"]["markets_skipped_empty"] += markets_empty
        summary["totals"]["rows"] += rows_day
        summary["totals"]["train_rows"] += train_rows_day
        summary["totals"]["test_rows"] += test_rows_day
        if is_oos:
            summary["totals"]["test_markets"] += markets_ok
        else:
            summary["totals"]["train_markets"] += markets_ok
        summary["totals"]["label_up_rows"] += label_up_rows_day

        print(f"[{ts()}] {date_str} done: {markets_ok} markets, {rows_day:,} rows, "
              f"{size_mb:.1f} MB, {dt:.1f}s, RSS={rss_mb():.0f} MB",
              flush=True)

        # Flush manifest after each day so we always have a checkpoint.
        summary["finished_at"] = ts()
        (OUT_ROOT / "_manifest.json").write_text(json.dumps(summary, indent=2))

    if pool is not None:
        pool.close()
        pool.join()
    summary["finished_at"] = ts()
    (OUT_ROOT / "_manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"[{ts()}] ALL DONE", flush=True)
    print(json.dumps(summary["totals"], indent=2))


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def rss_mb() -> float:
    try:
        import psutil  # type: ignore
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return -1.0


if __name__ == "__main__":
    main()
