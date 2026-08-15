#!/usr/bin/env python3
"""Build ONLY the dense_close dataset for one or more UTC days.

Mirrors src/polymarket_recorder/dataset_factory._build_day_datasets but
skips the coarse + market_interval outputs to save time and disk.
Writes to data/datasets/resolution_snapshot_dataset_v1_dense_close/.

Usage:
    py -3 tools/build_dense_close_only.py --dates 2026-05-21,2026-05-22
"""
from __future__ import annotations

import argparse
import io
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from polymarket_recorder.dataset_factory import (
    DENSE_DATASET_NAME, SECOND_NS,
    _apply_training_contract, _assert_no_forbidden_columns,
    _build_market_history_features, _build_market_live_window,
    _build_snapshot_row, _load_l2_indices_for_day, _load_live_reference_index,
    _load_market_labels, _load_rest_indices_for_day,
    _sample_times_for_dense_close, _write_parquet,
)
from market_recorders.dataset_policy import DEFAULT_DATASET_POLICY_PATH


def _parse_date(s: str) -> date:
    return date.fromisoformat(s.strip())


def build_dense_close_for_day(*, root: Path, policy_path: Path, target_date: date,
                              date_from_for_labels: date,
                              labels: list, history_features: dict,
                              out_dir: Path) -> int:
    """Returns number of rows written; mirrors _build_day_datasets but
    only fires the dense_close path."""
    shard_dates = [target_date - timedelta(days=1), target_date]
    market_slugs = {l.market_slug for l in labels}
    print(f"  [{target_date}] loading live_reference / L2 / REST indices...", flush=True)
    live_reference = _load_live_reference_index(root=root, shard_dates=shard_dates)
    l2_indices, _ = _load_l2_indices_for_day(
        root=root, policy_path=policy_path,
        shard_dates=shard_dates, market_slugs=market_slugs,
    )
    rest_indices, _ = _load_rest_indices_for_day(
        root=root, policy_path=policy_path,
        shard_dates=shard_dates, market_slugs=market_slugs,
    )

    dense_rows: list[dict] = []
    for label in sorted(labels, key=lambda x: (x.market_open_ts_ns, x.market_slug)):
        market_live = _build_market_live_window(label=label, live_reference=live_reference)
        up_index = l2_indices.get(label.market_slug, {}).get("up")
        down_index = l2_indices.get(label.market_slug, {}).get("down")
        rest_index = rest_indices.get(label.market_slug)
        for snapshot_ts_ns in _sample_times_for_dense_close(label):
            bucket_ms = 250 if snapshot_ts_ns < label.market_close_ts_ns - (10 * SECOND_NS) else 100
            dense_rows.append(_build_snapshot_row(
                label=label, snapshot_ts_ns=snapshot_ts_ns,
                sample_family="dense_close", sample_bucket_ms=bucket_ms,
                live_reference=live_reference, market_live=market_live,
                up_index=up_index, down_index=down_index, rest_index=rest_index,
            ))

    if not dense_rows:
        print(f"  [{target_date}] no rows produced.")
        return 0

    frame = pd.DataFrame(dense_rows).sort_values(["snapshot_ts_ns", "market_slug"], kind="stable")
    frame, _qrows = _apply_training_contract(frame, dataset_name=DENSE_DATASET_NAME)
    _assert_no_forbidden_columns(frame)
    path = out_dir / f"{target_date.isoformat()}.parquet"
    _write_parquet(frame, path)
    print(f"  [{target_date}] wrote {len(frame):,} rows -> {path}", flush=True)
    return len(frame)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", required=True, help="Comma-separated UTC dates, e.g. '2026-05-21,2026-05-22'")
    ap.add_argument("--raw-root", default=Path("data"), type=Path)
    ap.add_argument("--policy", default=DEFAULT_DATASET_POLICY_PATH, type=Path)
    ap.add_argument("--out-dir", default=Path("data/datasets/resolution_snapshot_dataset_v1_dense_close"),
                    type=Path)
    args = ap.parse_args()

    dates = sorted({_parse_date(d) for d in args.dates.split(",") if d.strip()})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"target dates: {[d.isoformat() for d in dates]}")
    print(f"output dir:   {args.out_dir}")

    print(f"\nloading market labels for {dates[0]} ... {dates[-1]}", flush=True)
    labels_by_day, summary = _load_market_labels(
        root=args.raw_root, policy_path=args.policy,
        date_from=dates[0], date_to=dates[-1],
    )
    matched = summary.get("matched_markets", "?")
    missing = summary.get("missing_resolution_market_count", "?")
    print(f"  matched_markets: {matched}, missing_resolution: {missing}")

    history_features = _build_market_history_features(labels_by_day)
    total = 0
    for d in dates:
        day_labels = labels_by_day.get(d, [])
        print(f"\n=== {d.isoformat()} : {len(day_labels)} markets ===", flush=True)
        if not day_labels:
            print(f"  no labelled markets for {d} — skipping")
            continue
        n = build_dense_close_for_day(
            root=args.raw_root, policy_path=args.policy,
            target_date=d, date_from_for_labels=dates[0],
            labels=day_labels, history_features=history_features,
            out_dir=args.out_dir,
        )
        total += n

    print(f"\nDONE — total dense_close rows written: {total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
