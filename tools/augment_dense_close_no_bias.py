"""Augment the dense_close parquet with RAW-binance-derived versions of
every feature the existing model used in a bias-corrected form.

The new columns all have a '_raw' suffix so they coexist with the originals
(and so we can A/B). They are computed exclusively from binance_spot_mid
(raw) + price_to_beat (the strike) — no synthetic_corrected, no rolling_bias
touched anywhere.

Computed columns:
  delta_to_strike_raw                = binance_spot_mid - price_to_beat
  abs_delta_to_strike_raw            = |delta_to_strike_raw|
  delta_sign_raw                     = sign(delta_to_strike_raw)
  delta_to_strike_over_vol_raw       = delta_to_strike_raw / max(btc_realized_vol_15s_raw, 1.0)
  btc_return_{1,3,5,10}s_raw         = (price_now - price_t_back) / price_t_back   [raw binance]
  btc_realized_vol_{5,15}s_raw       = std of 1-sec log returns over the window   [raw binance]
  time_spent_above_strike_recent_s_raw, time_spent_below_strike_recent_s_raw
                                     = count of last N seconds (default 60) where
                                       raw binance > / < strike

The lookback features need a per-second raw binance time series. That lives
in data/canonical/live_reference_events_v1/<date>.parquet under the
binance_spot_mid column (the column itself is RAW — separate from
synthetic_corrected which gets the bias). We join the dense_close
snapshots' second to that series and look back.

Usage:
  py -3 tools/augment_dense_close_no_bias.py --dates 2026-04-19 .. 2026-05-29
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
DC = REPO / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close"
CANON = REPO / "data" / "canonical" / "live_reference_events_v1"
OUT = REPO / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close_no_bias"
OUT.mkdir(parents=True, exist_ok=True)

LOOKBACK_FOR_PREFIX_S = 60   # matches existing time_spent_above/below_strike_recent_s


def build_btc_series(canon_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ts_seconds, binance_spot_mid) as parallel numpy arrays,
    using ONLY the raw binance column from canonical. None values are
    forward-filled with the last seen value."""
    df = pl.read_parquet(canon_path, columns=["ts_seconds", "binance_spot_mid"])
    df = df.sort("ts_seconds")
    ts = df["ts_seconds"].to_numpy().astype(np.int64)
    px = df["binance_spot_mid"].to_numpy().astype(float)
    # forward fill nans
    last = math.nan
    out_px = np.empty_like(px)
    for i in range(len(px)):
        v = px[i]
        if not math.isnan(v):
            last = v
        out_px[i] = last
    return ts, out_px


def lookup_at(ts_seconds: int, ts_arr: np.ndarray, px_arr: np.ndarray) -> float:
    """Returns the raw binance price at second ts_seconds (or the nearest
    earlier second). NaN if before the series start."""
    if len(ts_arr) == 0 or ts_seconds < ts_arr[0]:
        return math.nan
    idx = np.searchsorted(ts_arr, ts_seconds, side="right") - 1
    if idx < 0:
        return math.nan
    return float(px_arr[idx])


def compute_returns_and_vol(snap_seconds: np.ndarray,
                            ts_arr: np.ndarray, px_arr: np.ndarray):
    """Vectorised computation of per-row 1s/3s/5s/10s returns and
    5s/15s realised vol (std of log returns) directly from the per-second
    raw binance series."""
    # Map each snap_second to its index in ts_arr.
    snap_idx_in_ts = np.searchsorted(ts_arr, snap_seconds, side="right") - 1
    snap_idx_in_ts = np.where(snap_idx_in_ts >= 0, snap_idx_in_ts, -1)
    n = len(snap_seconds)

    def ret(horizon: int) -> np.ndarray:
        out = np.full(n, math.nan)
        for i in range(n):
            idx = snap_idx_in_ts[i]
            if idx < horizon: continue
            cur = px_arr[idx]; prev = px_arr[idx - horizon]
            if not (math.isfinite(cur) and math.isfinite(prev)) or prev == 0:
                continue
            out[i] = (cur - prev) / prev
        return out

    def vol(window: int) -> np.ndarray:
        # log returns std over the last `window` 1-sec intervals
        out = np.full(n, math.nan)
        # Precompute log returns once for the whole series
        with np.errstate(invalid="ignore", divide="ignore"):
            log_rets_series = np.log(px_arr[1:] / px_arr[:-1])
            log_rets_series = np.where(np.isfinite(log_rets_series), log_rets_series, math.nan)
        for i in range(n):
            idx = snap_idx_in_ts[i]
            if idx < window: continue
            # log returns ending at idx, length window
            seg = log_rets_series[idx - window: idx]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 2:
                out[i] = float(np.std(seg, ddof=1))
        return out

    return {
        "btc_return_1s_raw":  ret(1),
        "btc_return_3s_raw":  ret(3),
        "btc_return_5s_raw":  ret(5),
        "btc_return_10s_raw": ret(10),
        "btc_realized_vol_5s_raw":  vol(5),
        "btc_realized_vol_15s_raw": vol(15),
    }


def compute_prefix_counts(snap_seconds: np.ndarray,
                          strikes: np.ndarray,
                          ts_arr: np.ndarray, px_arr: np.ndarray,
                          lookback_s: int = LOOKBACK_FOR_PREFIX_S):
    """For each snapshot, count of last `lookback_s` 1-sec slots where
    raw binance was above / below the strike."""
    snap_idx = np.searchsorted(ts_arr, snap_seconds, side="right") - 1
    n = len(snap_seconds)
    above = np.full(n, math.nan)
    below = np.full(n, math.nan)
    for i in range(n):
        idx = snap_idx[i]
        if idx < 0: continue
        start = max(0, idx - lookback_s + 1)
        seg = px_arr[start: idx + 1]
        if len(seg) == 0: continue
        strike = strikes[i]
        a = np.sum(seg > strike)
        b = np.sum(seg < strike)
        above[i] = float(a)
        below[i] = float(b)
    return above, below


def augment_one_day(date_iso: str) -> Path:
    dc_path = DC / f"{date_iso}.parquet"
    canon_path = CANON / f"{date_iso}.parquet"
    if not dc_path.exists():
        print(f"  skip {date_iso}: no dense_close parquet")
        return None
    if not canon_path.exists():
        print(f"  skip {date_iso}: no canonical parquet")
        return None
    print(f"  loading {date_iso} ({dc_path.stat().st_size / 1e6:.1f} MB)")
    df = pl.read_parquet(dc_path)
    n = len(df)
    print(f"    {n} rows; building raw-binance series...")

    ts_arr, px_arr = build_btc_series(canon_path)
    finite = px_arr[np.isfinite(px_arr)]
    if len(finite) == 0:
        print(f"    skip {date_iso}: canonical has no finite binance_spot_mid")
        return None
    print(f"    canonical series: {len(ts_arr)} seconds, "
          f"px range ${finite.min():.2f} - ${finite.max():.2f}")

    # Snapshot seconds
    snap_sec = (df["snapshot_ts_ns"].to_numpy() // 1_000_000_000).astype(np.int64)
    strikes  = df["price_to_beat"].to_numpy().astype(float)

    # Direct algebra (no lookback)
    binance_now = df["binance_spot_mid"].to_numpy().astype(float)
    delta_raw   = binance_now - strikes
    abs_delta   = np.abs(delta_raw)
    delta_sign  = np.sign(delta_raw)

    # Returns + vol from per-second canonical raw binance
    print("    computing returns + vol from raw binance series...")
    rv = compute_returns_and_vol(snap_sec, ts_arr, px_arr)

    # delta / vol (use 15s vol)
    vol15 = rv["btc_realized_vol_15s_raw"]
    delta_over_vol = np.where(vol15 > 0, delta_raw / np.maximum(vol15, 1e-9), math.nan)

    # Prefix counts
    print(f"    computing prefix counts ({LOOKBACK_FOR_PREFIX_S}s window)...")
    above, below = compute_prefix_counts(snap_sec, strikes, ts_arr, px_arr)

    new_cols = {
        "delta_to_strike_raw":              delta_raw,
        "abs_delta_to_strike_raw":          abs_delta,
        "delta_sign_raw":                   delta_sign,
        "delta_to_strike_over_vol_raw":     delta_over_vol,
        "time_spent_above_strike_recent_s_raw": above,
        "time_spent_below_strike_recent_s_raw": below,
        **rv,
    }

    for k, v in new_cols.items():
        df = df.with_columns(pl.Series(name=k, values=v))

    out_path = OUT / f"{date_iso}.parquet"
    df.write_parquet(out_path)
    print(f"    -> {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    # Quick sanity print
    finite_delta = delta_raw[np.isfinite(delta_raw)]
    print(f"    delta_to_strike_raw   : p25={np.percentile(finite_delta,25):+.2f}  "
          f"p50={np.percentile(finite_delta,50):+.2f}  p75={np.percentile(finite_delta,75):+.2f}  "
          f"abs_mean=${np.abs(finite_delta).mean():.2f}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True,
                    help="ISO dates (YYYY-MM-DD), e.g. 2026-05-21 2026-05-22 ...")
    args = ap.parse_args()
    for d in args.dates:
        print(f"\n=== {d} ===")
        augment_one_day(d)


if __name__ == "__main__":
    main()
