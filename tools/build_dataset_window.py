#!/usr/bin/env python3
"""Build a dense_close parquet for an arbitrary [start, end] UTC window.

Lets us produce a "1-hour ground truth" dataset on demand instead of
waiting for a full day to complete. Uses the existing dataset_factory
code path — only difference is we filter MarketLabels to those whose
market_close_ts_ns falls inside the requested window.

Usage:
    py -3 tools/build_dataset_window.py \\
        --start 2026-05-22T08:00:00Z \\
        --end   2026-05-22T09:00:00Z \\
        --output data/datasets/_shadow_windows/2026-05-22_08-09.parquet

Notes:
- A market is included if its `market_close_ts_ns` is in [start, end).
  (We don't trim sub-window snapshots — the whole last-60s tail of any
   included market is built.)
- Requires the market to have RESOLVED before we run (the dataset
  factory needs `resolved_side` to fill the label-dependent columns).
  If you point this at a window that's so recent that resolution
  records haven't been written yet, those markets will silently
  drop out — re-run a few minutes later.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from polymarket_recorder.dataset_factory import (  # noqa: E402
    DEFAULT_DATASET_OUTPUT_DIR,
    DENSE_DATASET_NAME,
    _build_day_datasets,
    _build_market_history_features,
    _load_market_labels,
)
from market_recorders.dataset_policy import DEFAULT_DATASET_POLICY_PATH  # noqa: E402


def _parse_iso(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help='UTC ISO, e.g. "2026-05-22T08:00:00Z"')
    ap.add_argument("--end",   required=True, help='UTC ISO, e.g. "2026-05-22T09:00:00Z"')
    ap.add_argument("--raw-root", default="data", type=Path,
                    help="Root dir holding the recorder shards. UnifiedRawReader looks "
                         "under <root>/raw/<source>/... so pass the dir that *contains* `raw/` "
                         "(default: data)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output parquet path. The dataset_factory writes one parquet per UTC date; "
                         "this script then concatenates / copies the slice you asked for to this path.")
    ap.add_argument("--policy", default=DEFAULT_DATASET_POLICY_PATH, type=Path,
                    help="Dataset policy YAML (defaults to repo default).")
    ap.add_argument("--workdir", default=Path("data/datasets/_shadow_build"), type=Path,
                    help="Scratch dir for the per-day parquet the factory writes before we slice.")
    args = ap.parse_args()

    start = _parse_iso(args.start)
    end   = _parse_iso(args.end)
    if end <= start:
        print(f"FATAL: --end ({end}) must be > --start ({start})", file=sys.stderr)
        return 2
    start_ns = int(start.timestamp() * 1e9)
    end_ns   = int(end.timestamp()   * 1e9)

    days = []
    cur = start.date()
    while cur <= end.date():
        days.append(cur)
        cur += timedelta(days=1)
    print(f"window: {start.isoformat()} -> {end.isoformat()}  ({(end-start).total_seconds()/3600:.2f}h)")
    print(f"scanning days: {[d.isoformat() for d in days]}")

    # 1) Load labels for the date range the window touches.
    labels_by_day, label_summary = _load_market_labels(
        root=args.raw_root,
        policy_path=args.policy,
        date_from=days[0],
        date_to=days[-1],
    )
    all_labels = [lab for day_labels in labels_by_day.values() for lab in day_labels]
    print(f"loaded {len(all_labels)} resolved+labelled markets across {len(days)} day(s)")

    # 2) Filter to markets whose market_close_ts_ns is inside the window.
    in_window = [lab for lab in all_labels if start_ns <= lab.market_close_ts_ns < end_ns]
    print(f"  -> {len(in_window)} markets close inside the window")
    if not in_window:
        print("no markets in window — exiting (nothing to build).")
        return 1

    # 3) Group filtered labels by UTC date for the per-day builder.
    by_day: dict[date, list] = {}
    for lab in in_window:
        d = datetime.fromtimestamp(lab.market_close_ts_ns / 1e9, UTC).date()
        by_day.setdefault(d, []).append(lab)

    history_features = _build_market_history_features({d: labs for d, labs in by_day.items()})

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 4) Build one parquet per day touched, into the scratch workdir.
    import polars as pl
    pieces = []
    for d, day_labels in sorted(by_day.items()):
        print(f"  building {d.isoformat()} ({len(day_labels)} markets)...")
        task = {
            "root": str(args.raw_root),
            "target_date": d.isoformat(),
            "date_from": days[0].isoformat(),
            "policy_path": str(args.policy),
            "dataset_output_dir": str(args.workdir),
            "labels": [asdict(lab) for lab in day_labels],
            "history_features_by_market_slug": {
                lab.market_slug: history_features.get(lab.market_slug, {}) for lab in day_labels
            },
            "skip_existing": False,  # always rebuild (window scope may differ)
        }
        result = _build_day_datasets(task)
        dense_path = Path(result["paths"].get(DENSE_DATASET_NAME, ""))
        if not dense_path.exists():
            print(f"    !! {DENSE_DATASET_NAME} not produced for {d}: {result.get('exclusions')}")
            continue
        df = pl.read_parquet(dense_path)
        # Sub-window trim (paranoia: only keep rows whose snapshot_ts_ns is < end_ns)
        df = df.filter(pl.col("snapshot_ts_ns") < end_ns)
        pieces.append(df)
        print(f"    -> {len(df):,} dense rows from {d.isoformat()}")

    if not pieces:
        print("no dense rows produced — exiting.")
        return 1

    out = pl.concat(pieces, how="diagonal").sort(["market_slug", "snapshot_ts_ns"])
    out.write_parquet(args.output)

    print(f"\nWROTE  {args.output}   ({len(out):,} rows, {out['market_slug'].n_unique()} markets)")
    print(f"  ttc range: {out['t_to_close_s'].min():.1f}s -> {out['t_to_close_s'].max():.1f}s")
    print(f"  bid range: ${out['up_token_best_bid'].min():.3f} -> ${out['up_token_best_bid'].max():.3f}")

    # Summary JSON next to the parquet
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "raw_root": str(args.raw_root),
        "policy_path": str(args.policy),
        "n_markets_in_window": len(in_window),
        "n_rows": len(out),
        "days_touched": [d.isoformat() for d in sorted(by_day.keys())],
        "output_parquet": str(args.output),
    }, indent=2))
    print(f"WROTE  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
