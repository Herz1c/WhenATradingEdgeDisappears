#!/usr/bin/env python3
"""Live-pipeline feature parity test.

For a given [start, end] window:

  1. Build "reference" rows by running the existing offline
     dataset_factory pipeline (slow, but already known-correct — it
     produced all our training data).
  2. Build "live" rows by replaying the same raw events in arrival
     order through src/live/streaming_engine.py.
  3. Join on (market_slug, snapshot_ts_ns) and diff every numeric
     feature column.

A successful run means: the streaming engine produces the same 72
features as the offline path, given identical raw inputs. That makes
it safe to swap the file-replay source for a true WebSocket source
without changing the strategy behavior.

Usage:
    py -3 tools/live_feature_parity.py \\
        --start 2026-05-20T14:00:00Z \\
        --end   2026-05-20T15:00:00Z \\
        --output docs/live_feature_parity/2026-05-20_14-15.md
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

import numpy as np
import pandas as pd
import polars as pl

from polymarket_recorder.dataset_factory import MarketLabel
from live.streaming_engine import load_live_reference_from_canonical, replay_window_to_rows


def _labels_from_reference_parquet(ref: pl.DataFrame) -> list[MarketLabel]:
    """Reconstruct MarketLabels from a reference parquet — cheaper
    than re-scanning the strike+resolution archive."""
    by_slug: dict[str, dict] = {}
    cols_needed = ["market_slug", "market_id", "condition_id",
                   "market_open_ts_ns", "market_close_ts_ns",
                   "price_to_beat", "resolved_side"]
    for c in cols_needed:
        if c not in ref.columns:
            raise RuntimeError(f"reference parquet missing column: {c}")
    sub = ref.select(cols_needed).unique(subset=["market_slug"]).sort("market_open_ts_ns")
    labels: list[MarketLabel] = []
    for row in sub.iter_rows(named=True):
        slug = str(row["market_slug"])
        if slug in by_slug:
            continue
        by_slug[slug] = row
        labels.append(MarketLabel(
            market_slug=slug,
            market_id=str(row["market_id"] or ""),
            condition_id=str(row["condition_id"] or ""),
            strike_recv_ts_ns=int(row["market_open_ts_ns"]),     # only used for sort stability
            price_to_beat=float(row["price_to_beat"]),
            market_open_ts_ns=int(row["market_open_ts_ns"]),
            market_close_ts_ns=int(row["market_close_ts_ns"]),
            market_open_s=int(row["market_open_ts_ns"]) // 1_000_000_000,
            market_close_s=int(row["market_close_ts_ns"]) // 1_000_000_000,
            up_asset_id=None,                                    # streaming engine routes by slug, not asset id
            down_asset_id=None,
            resolved_side=str(row["resolved_side"] or "unknown"),
            resolution_recv_ts_ns=int(row["market_close_ts_ns"]),
            resolution_resolved_ts_ns=int(row["market_close_ts_ns"]),
        ))
    return labels


def _parse_iso(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"): s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end",   required=True)
    ap.add_argument("--raw-root", default=Path("data"), type=Path)
    ap.add_argument("--output",   required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path,
                    help="Existing dense_close parquet covering [start, end] — used as the "
                         "ground-truth reference. Build with tools/build_dataset_window.py "
                         "or use the historical per-day parquet under data/datasets/.")
    args = ap.parse_args()

    start = _parse_iso(args.start); end = _parse_iso(args.end)
    start_ns = int(start.timestamp() * 1e9)
    end_ns   = int(end.timestamp() * 1e9)
    print(f"window: {start.isoformat()} -> {end.isoformat()}")

    days: list[date] = []
    d = start.date()
    while d <= end.date():
        days.append(d); d += timedelta(days=1)

    # ── 1) reference parquet + labels ────────────────────────────────────────
    ref = pl.read_parquet(args.reference)
    ref = ref.filter((pl.col("snapshot_ts_ns") >= start_ns) & (pl.col("snapshot_ts_ns") < end_ns))
    if len(ref) == 0:
        print(f"reference parquet has 0 rows in window — pick a parquet that covers {start}..{end}")
        return 1
    print(f"reference rows in window: {len(ref):,}  columns: {len(ref.columns)}")

    in_window = _labels_from_reference_parquet(ref)
    print(f"markets in window: {len(in_window)}")

    # ── 3) build "live" rows via streaming engine ────────────────────────────
    print("loading live reference index...")
    live_ref = load_live_reference_from_canonical(root=args.raw_root, shard_dates=days)
    print(f"live_reference rows: {len(live_ref.ts_seconds):,}")

    print("replaying raw events through streaming engine...")
    rows = replay_window_to_rows(
        raw_root=args.raw_root, labels=in_window,
        start_ns=start_ns, end_ns=end_ns, live_reference=live_ref,
    )
    print(f"live rows emitted: {len(rows):,}")
    if not rows:
        print("live engine produced no rows"); return 3
    live = pl.from_pandas(pd.DataFrame(rows))
    if "snapshot_ts_ns" not in live.columns:
        print("live rows missing snapshot_ts_ns — emit schema mismatch"); return 4

    # Restrict both to the window
    ref  = ref.filter((pl.col("snapshot_ts_ns") >= start_ns) & (pl.col("snapshot_ts_ns") < end_ns))
    live = live.filter((pl.col("snapshot_ts_ns") >= start_ns) & (pl.col("snapshot_ts_ns") < end_ns))

    # ── 4) join + diff ───────────────────────────────────────────────────────
    join_keys = ["market_slug", "snapshot_ts_ns"]
    common_cols = [c for c in ref.columns if c in live.columns and c not in join_keys]
    print(f"diffable columns: {len(common_cols)}")

    # only diff numeric cols (everything else is metadata / leakage)
    numeric_cols = [c for c in common_cols
                    if ref.schema[c].is_numeric() and live.schema[c].is_numeric()]
    print(f"numeric columns: {len(numeric_cols)}")

    j = ref.select(join_keys + numeric_cols).rename({c: f"{c}__ref" for c in numeric_cols}).join(
        live.select(join_keys + numeric_cols).rename({c: f"{c}__live" for c in numeric_cols}),
        on=join_keys, how="inner",
    )
    n_common = len(j)
    print(f"common rows (intersection on join keys): {n_common:,}")

    n_ref_only  = len(ref)  - n_common
    n_live_only = len(live) - n_common

    # per-feature diff stats — convert each to float and compute absdiff
    summary_rows: list[dict] = []
    for c in numeric_cols:
        a = j[f"{c}__ref"].cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy()
        b = j[f"{c}__live"].cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy()
        # both-null counts agreement; either-null counts disagreement (handled as inf below)
        both_null = np.isnan(a) & np.isnan(b)
        either_null = np.isnan(a) ^ np.isnan(b)
        diff = np.where(both_null, 0.0, np.where(either_null, np.inf, np.abs(a - b)))
        finite = diff[np.isfinite(diff)]
        summary_rows.append({
            "feature": c,
            "n": int(n_common),
            "both_null_frac":   float(np.mean(both_null)),
            "either_null_frac": float(np.mean(either_null)),
            "max_finite_diff":  float(finite.max()) if finite.size else 0.0,
            "p99_finite_diff":  float(np.percentile(finite, 99)) if finite.size else 0.0,
            "p50_finite_diff":  float(np.percentile(finite, 50)) if finite.size else 0.0,
            "n_disagree_gt_1e_6": int(np.sum(diff > 1e-6)),
        })
    summary_rows.sort(key=lambda r: -r["max_finite_diff"])

    # ── 5) report ────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    add = lines.append
    add(f"# Live ↔ offline feature parity")
    add("")
    add(f"- window: {start.isoformat()} → {end.isoformat()}")
    add(f"- reference rows (offline dataset_factory): **{len(ref):,}**")
    add(f"- live rows (streaming engine):              **{len(live):,}**")
    add(f"- joined common rows: **{n_common:,}**  "
        f"(ref-only: {n_ref_only:,} | live-only: {n_live_only:,})")
    add(f"- numeric columns diffed: **{len(numeric_cols)}**")
    add("")

    worst = max(summary_rows, key=lambda r: r["max_finite_diff"]) if summary_rows else None
    overall_max = worst["max_finite_diff"] if worst else 0.0
    add(f"## Verdict")
    if overall_max <= 1e-6 and n_common >= 0.99 * len(ref):
        add("**PASS** — every feature matches within 1e-6, coverage ≥ 99%.\n")
    else:
        add(f"**FAIL** — overall max diff = {overall_max:.4e}, "
            f"coverage = {n_common/max(1,len(ref))*100:.2f}%.\n")
        if worst:
            add(f"Worst feature: `{worst['feature']}` (max diff {worst['max_finite_diff']:.4e}, "
                f"{worst['n_disagree_gt_1e_6']} rows > 1e-6)\n")

    add(f"## Top 30 features by max abs diff")
    add(f"| feature | max | p99 | p50 | rows > 1e-6 | either-null frac |")
    add(f"|---|---:|---:|---:|---:|---:|")
    for r in summary_rows[:30]:
        add(f"| `{r['feature']}` | {r['max_finite_diff']:.4e} | {r['p99_finite_diff']:.4e} | "
            f"{r['p50_finite_diff']:.4e} | {r['n_disagree_gt_1e_6']} | "
            f"{r['either_null_frac']:.4f} |")
    add("")
    add(f"## Perfect-match features (max diff = 0)")
    perfect = [r for r in summary_rows if r["max_finite_diff"] == 0.0 and r["either_null_frac"] == 0.0]
    add(f"{len(perfect)} of {len(summary_rows)} numeric features identical bit-for-bit.")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    # also dump the full per-feature JSON for forensics
    args.output.with_suffix(".json").write_text(json.dumps(summary_rows, indent=2))
    print(f"\nWROTE {args.output}")
    print(f"  overall max diff: {overall_max:.4e}")
    print(f"  perfect-match features: {len(perfect)} / {len(summary_rows)}")
    print(f"  coverage: {n_common/max(1,len(ref))*100:.2f}%")
    return 0 if (overall_max <= 1e-6 and n_common >= 0.99 * len(ref)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
