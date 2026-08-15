"""Sanity report over data/datasets/polymarket_l2_last60s_v1/*.parquet.

Reads the manifest plus a streaming scan of every shard. Outputs:
- Per-day row/market/label counts
- Train vs OOS totals
- ttc_s histogram
- Label balance overall and per split
- Spot checks: implied_p_up calibration vs label_up
- Feature null/zero rates (warn if a key feature is dead)
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "datasets" / "polymarket_l2_last60s_v1"


def main() -> None:
    manifest_path = DATA_DIR / "_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print("=== MANIFEST TOTALS ===")
        print(json.dumps(manifest["totals"], indent=2))
        print(f"OOS dates: {manifest['oos_dates']}")
        print(f"Window: -{manifest['window_offset_s'][0]}s .. -{manifest['window_offset_s'][1]}s")
        print(f"Features: {len(manifest['feature_columns'])}")
        print()

    parquets = sorted(DATA_DIR.glob("*.parquet"))
    print(f"=== SHARDS: {len(parquets)} ===")
    total_size_mb = sum(p.stat().st_size for p in parquets) / 1e6
    print(f"Total parquet size: {total_size_mb:.1f} MB")
    print()

    # Streaming scan via polars LazyFrame.
    lf = pl.scan_parquet(str(DATA_DIR / "*.parquet"))

    # Per-day summary
    per_day = (
        lf.group_by("date")
          .agg([
              pl.len().alias("rows"),
              pl.col("market_slug").n_unique().alias("markets"),
              pl.col("label_up").mean().alias("label_up_frac"),
              pl.col("split").first().alias("split"),
          ])
          .sort("date")
          .collect()
    )
    print("=== PER-DAY ===")
    print(per_day)
    print()

    # Train vs OOS totals
    by_split = (
        lf.group_by("split")
          .agg([
              pl.len().alias("rows"),
              pl.col("market_slug").n_unique().alias("markets"),
              pl.col("label_up").mean().alias("label_up_frac"),
          ])
          .collect()
    )
    print("=== TRAIN vs OOS ===")
    print(by_split)
    print()

    # ttc histogram (binned to 5s)
    ttc_hist = (
        lf.with_columns((pl.col("ttc_s") // 5 * 5).alias("ttc_bin"))
          .group_by("ttc_bin")
          .agg([pl.len().alias("rows"), pl.col("label_up").mean().alias("p_up")])
          .sort("ttc_bin", descending=True)
          .collect()
    )
    print("=== ROWS PER 5s TTC BIN ===")
    print(ttc_hist)
    print()

    # implied_p_up calibration check
    calib = (
        lf.with_columns(((pl.col("implied_p_up").clip(0.0, 1.0) * 10).floor() / 10).alias("p_bin"))
          .group_by("p_bin")
          .agg([pl.len().alias("rows"), pl.col("label_up").mean().alias("actual_p_up")])
          .sort("p_bin")
          .collect()
    )
    print("=== implied_p_up CALIBRATION (deciles) ===")
    print(calib)
    print()

    # Feature health: zero-rates per feature
    feat_cols = [c for c in lf.collect_schema().names()
                 if c not in ("recv_ts_ns", "event_type", "market_slug",
                              "market_close_s", "date", "split", "label_up")]
    zero_rates = (
        lf.select([(pl.col(c) == 0).mean().alias(c) for c in feat_cols]).collect()
    )
    zero_dict = {c: float(zero_rates[c][0]) for c in feat_cols}
    dead = {c: r for c, r in zero_dict.items() if r > 0.95}
    print(f"=== FEATURE HEALTH ===")
    print(f"Mostly-zero features (>95%): {len(dead)}")
    for c, r in sorted(dead.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {c}: {r:.3f}")


if __name__ == "__main__":
    main()
