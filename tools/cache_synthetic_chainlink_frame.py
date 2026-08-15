"""Materialise and cache the synthetic Chainlink reference frame for a day range."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chainlink_recorder.synthetic_chainlink import (  # noqa: E402
    SECOND_NS,
    STALE_NS,
    SOURCE_NAMES,
    load_chainlink_ground_truth,
    load_clean_source_updates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--cache-path", type=Path, default=Path("data/_cache/synthetic_chainlink_experiment.parquet"))
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("loading_sources", {"root": str(args.root), "max_workers": args.max_workers}, flush=True)

    source_updates = {
        name: load_clean_source_updates(root=args.root, source_name=name, max_workers=args.max_workers)
        for name in SOURCE_NAMES
    }
    ground_truth = load_chainlink_ground_truth(root=args.root, max_workers=args.max_workers)

    import numpy as np
    import pandas as pd

    start = ground_truth[0].chainlink_ts_ns
    end = ground_truth[-1].chainlink_ts_ns
    seconds = np.arange(start, end + SECOND_NS, SECOND_NS, dtype=np.int64)

    base = pd.DataFrame({"ts_ns": seconds})
    base["dt"] = pd.to_datetime(base["ts_ns"], unit="ns", utc=True)
    base["hour"] = base["dt"].dt.hour
    base["minute"] = base["dt"].dt.minute
    base["dow"] = base["dt"].dt.day_name()
    base["tod_frac"] = (base["hour"] * 3600 + base["minute"] * 60) / 86400.0
    for harmonic in [1, 2, 3, 4, 6, 8, 12]:
        base[f"sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * base["tod_frac"])
        base[f"cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * base["tod_frac"])

    for name in SOURCE_NAMES:
        updates = source_updates[name]
        df = pd.DataFrame(
            {
                "ts_ns": [u.second_ts_ns for u in updates],
                f"{name}_mid": [u.mid for u in updates],
                f"{name}_weight": [u.weight for u in updates],
                f"{name}_last_recv_ts_ns": [u.last_recv_ts_ns for u in updates],
            }
        )
        base = base.merge(df, on="ts_ns", how="left")
        base[f"{name}_mid"] = base[f"{name}_mid"].ffill()
        base[f"{name}_weight"] = base[f"{name}_weight"].ffill()
        base[f"{name}_last_recv_ts_ns"] = base[f"{name}_last_recv_ts_ns"].ffill()
        stale = base[f"{name}_last_recv_ts_ns"].isna() | (
            (base["ts_ns"] - base[f"{name}_last_recv_ts_ns"]) > STALE_NS
        )
        base[f"{name}_staleness_s"] = (
            (base["ts_ns"] - base[f"{name}_last_recv_ts_ns"]) / SECOND_NS
        ).where(~base[f"{name}_last_recv_ts_ns"].isna())
        base.loc[stale, f"{name}_mid"] = np.nan
        base.loc[stale, f"{name}_weight"] = np.nan

    base["spot"] = base["binance_spot_mid"]
    base["usdm"] = base["binance_usdm_mid"]
    base["hl"] = base["hyperliquid_mid"]
    base["source_count"] = base[["spot", "usdm", "hl"]].notna().sum(axis=1)
    base["avg_nonspot"] = base[["usdm", "hl"]].mean(axis=1)
    base["basis_usdm"] = base["spot"] - base["usdm"]
    base["basis_hl"] = base["spot"] - base["hl"]
    base["avg_basis"] = base[["basis_usdm", "basis_hl"]].mean(axis=1)
    base["basis_diff"] = base["basis_usdm"] - base["basis_hl"]
    base["median_all"] = base[["spot", "usdm", "hl"]].median(axis=1)
    base["max_cross_spread"] = (
        base[["spot", "usdm", "hl"]].max(axis=1) - base[["spot", "usdm", "hl"]].min(axis=1)
    )
    base["spot_minus_median"] = base["spot"] - base["median_all"]
    base["spot_nonspot_gap"] = base["spot"] - base["avg_nonspot"]

    for col in [
        "spot",
        "avg_basis",
        "basis_usdm",
        "basis_hl",
        "basis_diff",
        "max_cross_spread",
        "spot_nonspot_gap",
    ]:
        for win in [5, 15, 30, 60, 120, 300, 900]:
            base[f"{col}_mean_{win}"] = base[col].rolling(win, min_periods=1).mean()
            base[f"{col}_std_{win}"] = base[col].rolling(
                win, min_periods=max(2, min(win, 5))
            ).std()
            base[f"{col}_ema_{win}"] = base[col].ewm(
                span=win, adjust=False, min_periods=1
            ).mean()

    for col in ["spot", "usdm", "hl", "avg_nonspot"]:
        for win in [1, 2, 5, 10, 30, 60, 120, 300]:
            base[f"{col}_ret_{win}"] = base[col] / base[col].shift(win) - 1.0

    logret = np.log(base["spot"] / base["spot"].shift(1))
    for win in [5, 10, 30, 60, 120, 300, 900]:
        base[f"spot_rv_{win}"] = (
            logret.pow(2).rolling(win, min_periods=max(2, min(win, 5))).sum()
        ).pow(0.5)

    for name in SOURCE_NAMES:
        base[f"{name}_present"] = base[f"{name}_mid"].notna().astype(int)

    truth = pd.DataFrame(
        {
            "ts_ns": [e.chainlink_ts_ns for e in ground_truth],
            "chainlink_price": [e.chainlink_price for e in ground_truth],
        }
    )
    frame = truth.merge(base, on="ts_ns", how="left")
    frame = frame[frame["source_count"] >= 2].copy()
    frame["date"] = frame["dt"].dt.date.astype(str)
    frame["baseline"] = frame["spot"] * 1.00029 + 0.03 * frame["avg_basis"].fillna(0.0)
    frame["baseline_abs"] = (frame["baseline"] - frame["chainlink_price"]).abs()

    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.cache_path, index=False)
    print(
        "saved_cache",
        {
            "path": str(args.cache_path),
            "rows": int(len(frame)),
            "baseline_medae": float(frame["baseline_abs"].median()),
            "baseline_p90": float(frame["baseline_abs"].quantile(0.9)),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
