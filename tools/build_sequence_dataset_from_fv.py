"""Build v0 market-sequence tensors from fair_value_v2 source-time snapshots.

This is intentionally a fast sanity-check dataset, not a raw replay rebuild.
Each 5-minute market becomes one causal sequence over the existing fair-value
decision grid.  Timesteps keep source-freshness masks so sequence models can be
trained/evaluated only where the same state would be live-replicable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.train_fair_value_v2_source_time import feature_cols  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "datasets" / "fair_value_v2_source_time"
DEFAULT_OUT = ROOT / "data" / "datasets" / "btc_5m_sequences_v0_from_fv"

NS = 1_000_000_000
QUOTE_COLS = [
    "up_best_bid",
    "up_best_ask",
    "down_best_bid",
    "down_best_ask",
    "implied_p_up",
]
AUDIT_COLS = [
    "pm_source_lag_s",
    "pm_recv_lag_s",
    "pm_delivery_lag_s",
    "coinbase_source_lag_s",
    "rtds_source_age_s",
    "rtds_age_s",
]


def _date_from_path(path: Path) -> str:
    return path.name[:10]


def _expand_date_filter(spec: str | None) -> set[str] | None:
    if not spec or spec.lower() == "all":
        return None
    out: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            d0 = pd.Timestamp(a).date()
            d1 = pd.Timestamp(b).date()
            cur = d0
            while cur <= d1:
                out.add(cur.isoformat())
                cur = (pd.Timestamp(cur) + pd.Timedelta(days=1)).date()
        else:
            out.add(part)
    return out


def _uses_cex(feature_set: str) -> bool:
    return feature_set not in {"pm_rtds_safe"}


def _finite_or_nan(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    values[~np.isfinite(values)] = np.nan
    return values


def _valid_rows(
    frame: pd.DataFrame,
    *,
    uses_cex: bool,
    max_pm_source_lag_s: float | None,
    max_cex_source_lag_s: float | None,
    max_rtds_source_age_s: float | None,
    max_rtds_age_s: float | None,
) -> np.ndarray:
    valid = np.ones(len(frame), dtype=bool)
    if max_pm_source_lag_s is not None and "pm_source_lag_s" in frame:
        valid &= np.isfinite(frame["pm_source_lag_s"].to_numpy(float))
        valid &= frame["pm_source_lag_s"].to_numpy(float) <= max_pm_source_lag_s
    if uses_cex and max_cex_source_lag_s is not None and "coinbase_source_lag_s" in frame:
        valid &= np.isfinite(frame["coinbase_source_lag_s"].to_numpy(float))
        valid &= frame["coinbase_source_lag_s"].to_numpy(float) <= max_cex_source_lag_s
    if max_rtds_source_age_s is not None and "rtds_source_age_s" in frame:
        valid &= np.isfinite(frame["rtds_source_age_s"].to_numpy(float))
        valid &= frame["rtds_source_age_s"].to_numpy(float) <= max_rtds_source_age_s
    if max_rtds_age_s is not None and "rtds_age_s" in frame:
        valid &= np.isfinite(frame["rtds_age_s"].to_numpy(float))
        valid &= frame["rtds_age_s"].to_numpy(float) <= max_rtds_age_s
    return valid


def _split_dates(dates: list[str], train_end: str, test_start: str, val_days: int) -> dict[str, list[str]]:
    train_all = [d for d in dates if d <= train_end]
    test = [d for d in dates if d >= test_start]
    if len(train_all) <= val_days:
        raise SystemExit(f"need more train dates than val_days={val_days}; have {len(train_all)}")
    val = train_all[-val_days:]
    train = train_all[:-val_days]
    return {"train": train, "val": val, "test": test}


def _array_stats(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_features = x.shape[-1]
    mean = np.zeros(n_features, dtype=np.float32)
    std = np.ones(n_features, dtype=np.float32)
    flat = x.reshape(-1, n_features)
    flat_mask = mask.reshape(-1)
    for j in range(n_features):
        vals = flat[flat_mask, j]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            m = float(vals.mean())
            s = float(vals.std())
            mean[j] = m if math.isfinite(m) else 0.0
            std[j] = s if math.isfinite(s) and s > 1e-8 else 1.0
    return mean, std


def _normalize(x: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    y = (x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    y[~np.isfinite(y)] = 0.0
    y[~mask] = 0.0
    return y.astype(np.float32, copy=False)


def _save_npz(path: Path, *, compress: bool, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    date_filter = _expand_date_filter(args.dates)
    files = sorted(dataset_dir.glob("*.parquet"))
    if date_filter is not None:
        files = [p for p in files if _date_from_path(p) in date_filter]
    if not files:
        raise SystemExit(f"no parquet files found in {dataset_dir}")

    sample = pd.read_parquet(files[0])
    features = feature_cols(sample, feature_set=args.feature_set)
    if "implied_p_up" not in features:
        raise SystemExit("implied_p_up must be present so residual sequence models can use market prior")
    uses_cex = _uses_cex(args.feature_set)
    ttc_grid = np.arange(float(args.seq_ttc_max), float(args.seq_ttc_min) - 0.5, -float(args.cadence_s),
                         dtype=np.float32)
    seq_len = int(ttc_grid.size)
    ttc_to_slot = {int(round(float(ttc))): i for i, ttc in enumerate(ttc_grid)}

    episodes: list[dict[str, Any]] = []
    per_day: dict[str, dict[str, Any]] = {}
    t0 = time.time()

    needed_cols = sorted(set(features + QUOTE_COLS + AUDIT_COLS + [
        "market_slug", "now_ns", "resolved_up", "ttc_s", "cex_source", "dataset_clock_mode",
    ]))
    for path in files:
        date_s = _date_from_path(path)
        frame = pd.read_parquet(path)
        cols = [c for c in needed_cols if c in frame.columns]
        frame = frame[cols].copy()
        frame = frame[(frame["ttc_s"] >= args.seq_ttc_min - 1e-6)
                      & (frame["ttc_s"] <= args.seq_ttc_max + 1e-6)]
        if frame.empty:
            per_day[date_s] = {"episodes": 0, "rows": 0, "kept": 0, "valid_timesteps": 0}
            continue
        valid_source = _valid_rows(
            frame,
            uses_cex=uses_cex,
            max_pm_source_lag_s=args.max_pm_source_lag_s,
            max_cex_source_lag_s=args.max_cex_source_lag_s,
            max_rtds_source_age_s=args.max_rtds_source_age_s,
            max_rtds_age_s=args.max_rtds_age_s,
        )
        frame["_source_valid"] = valid_source
        n_day_rows = len(frame)
        n_day_kept = 0
        n_day_valid = 0
        n_day_episodes = 0
        for slug, g in frame.groupby("market_slug", sort=False):
            g = g.sort_values("now_ns", kind="stable")
            x = np.full((seq_len, len(features)), np.nan, dtype=np.float32)
            quotes = np.full((seq_len, len(QUOTE_COLS)), np.nan, dtype=np.float32)
            audit = np.full((seq_len, len(AUDIT_COLS)), np.nan, dtype=np.float32)
            now_ns = np.zeros(seq_len, dtype=np.int64)
            source_valid = np.zeros(seq_len, dtype=bool)
            row_present = np.zeros(seq_len, dtype=bool)

            # If duplicate rows land in the same 1s bucket, keep the latest now_ns.
            for _, row in g.iterrows():
                slot = ttc_to_slot.get(int(round(float(row["ttc_s"]))))
                if slot is None:
                    continue
                ts = int(row["now_ns"])
                if now_ns[slot] and ts < now_ns[slot]:
                    continue
                row_present[slot] = True
                now_ns[slot] = ts
                x[slot, :] = _finite_or_nan(row.reindex(features).to_numpy(dtype=np.float32))
                quotes[slot, :] = _finite_or_nan(row.reindex(QUOTE_COLS).to_numpy(dtype=np.float32))
                audit[slot, :] = _finite_or_nan(row.reindex(AUDIT_COLS).to_numpy(dtype=np.float32))
                source_valid[slot] = bool(row["_source_valid"])

            finite_feature_mask = np.isfinite(x).all(axis=1)
            valid_mask = row_present & source_valid & finite_feature_mask
            valid_count = int(valid_mask.sum())
            if valid_count < args.min_valid_steps:
                continue
            if valid_count / seq_len < args.min_valid_frac:
                continue
            y_vals = g["resolved_up"].dropna().astype(int).unique()
            if len(y_vals) != 1:
                continue
            episodes.append({
                "date": date_s,
                "market_slug": str(slug),
                "x": x,
                "quotes": quotes,
                "audit": audit,
                "now_ns": now_ns,
                "row_present": row_present,
                "source_valid": source_valid,
                "valid_mask": valid_mask,
                "resolved_up": int(y_vals[0]),
                "valid_count": valid_count,
            })
            n_day_episodes += 1
            n_day_kept += int(row_present.sum())
            n_day_valid += valid_count
        per_day[date_s] = {
            "episodes": n_day_episodes,
            "rows": int(n_day_rows),
            "present_timesteps": n_day_kept,
            "valid_timesteps": n_day_valid,
            "valid_frac": round(n_day_valid / max(1, n_day_episodes * seq_len), 4) if n_day_episodes else 0.0,
        }
        print(f"{date_s}: episodes={n_day_episodes} valid_steps={n_day_valid} "
              f"valid_frac={per_day[date_s]['valid_frac']:.2%}", flush=True)

    if not episodes:
        raise SystemExit("no episodes survived sequence filters")

    dates = sorted({ep["date"] for ep in episodes})
    splits = _split_dates(dates, args.train_end, args.test_start, args.val_days)
    split_by_date = {d: name for name, ds in splits.items() for d in ds}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        split = split_by_date.get(ep["date"])
        if split:
            buckets[split].append(ep)
    if not buckets["train"] or not buckets["val"] or not buckets["test"]:
        raise SystemExit({k: len(v) for k, v in buckets.items()})

    def stack_eps(eps: list[dict[str, Any]], *, normalized: bool,
                  mean: np.ndarray | None = None, std: np.ndarray | None = None) -> dict[str, Any]:
        x_raw = np.stack([ep["x"] for ep in eps], axis=0).astype(np.float32)
        valid_mask = np.stack([ep["valid_mask"] for ep in eps], axis=0)
        if normalized:
            assert mean is not None and std is not None
            x = _normalize(x_raw, valid_mask, mean, std)
        else:
            x = x_raw
        return {
            "X": x,
            "valid_mask": valid_mask,
            "row_present_mask": np.stack([ep["row_present"] for ep in eps], axis=0),
            "source_valid_mask": np.stack([ep["source_valid"] for ep in eps], axis=0),
            "y": np.array([ep["resolved_up"] for ep in eps], dtype=np.int8),
            "p_market": np.stack([
                ep["quotes"][:, QUOTE_COLS.index("implied_p_up")] for ep in eps
            ], axis=0).astype(np.float32),
            "quotes": np.stack([ep["quotes"] for ep in eps], axis=0).astype(np.float32),
            "audit": np.stack([ep["audit"] for ep in eps], axis=0).astype(np.float32),
            "now_ns": np.stack([ep["now_ns"] for ep in eps], axis=0).astype(np.int64),
            "market_slug": np.array([ep["market_slug"] for ep in eps]),
            "date": np.array([ep["date"] for ep in eps]),
            "valid_count": np.array([ep["valid_count"] for ep in eps], dtype=np.int16),
            "ttc_grid": ttc_grid,
        }

    train_raw = stack_eps(buckets["train"], normalized=False)
    mean, std = _array_stats(train_raw["X"], train_raw["valid_mask"])
    norm = {
        "feature_names": features,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_split": "train",
        "fit_valid_timesteps": int(train_raw["valid_mask"].sum()),
    }

    for split, eps in buckets.items():
        arrays = stack_eps(eps, normalized=True, mean=mean, std=std)
        _save_npz(out_dir / f"{split}.npz", compress=args.compress, **arrays)

    split_manifest = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "train_markets": len(buckets["train"]),
        "val_markets": len(buckets["val"]),
        "test_markets": len(buckets["test"]),
    }
    policy = {
        "dataset_version": "btc_5m_sequences_v0_from_fv",
        "source_dataset": str(dataset_dir),
        "output_dir": str(out_dir),
        "feature_set": args.feature_set,
        "feature_names": features,
        "quote_names": QUOTE_COLS,
        "audit_names": AUDIT_COLS,
        "uses_cex": uses_cex,
        "causal_order": "early_to_late",
        "sequence_definition": "existing fair_value_v2 source-time decision rows, bucketed by integer TTC seconds",
        "target": "resolved_up per market; sequence losses must use valid_mask for live-replicable timesteps",
        "ttc_grid": ttc_grid.tolist(),
        "freshness_policy": {
            "max_pm_source_lag_s": args.max_pm_source_lag_s,
            "max_cex_source_lag_s": args.max_cex_source_lag_s if uses_cex else None,
            "max_rtds_source_age_s": args.max_rtds_source_age_s,
            "max_rtds_age_s": args.max_rtds_age_s,
            "min_valid_steps": args.min_valid_steps,
            "min_valid_frac": args.min_valid_frac,
        },
        "splits": split_manifest,
        "normalization": norm,
        "per_day": per_day,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (out_dir / "quote_names.json").write_text(json.dumps(QUOTE_COLS, indent=2), encoding="utf-8")
    (out_dir / "audit_names.json").write_text(json.dumps(AUDIT_COLS, indent=2), encoding="utf-8")
    (out_dir / "normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    (out_dir / "splits.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return policy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--feature-set", default="cex_ticker_min",
                    choices=("full", "pm_rtds_safe", "cex_oracle_gap", "cex_oracle_core", "cex_ticker_min"))
    ap.add_argument("--dates", default="all", help="all, comma list, or YYYY-MM-DD:YYYY-MM-DD ranges")
    ap.add_argument("--seq-ttc-min", type=float, default=3.0)
    ap.add_argument("--seq-ttc-max", type=float, default=90.0)
    ap.add_argument("--cadence-s", type=float, default=1.0)
    ap.add_argument("--max-pm-source-lag-s", type=float, default=2.0)
    ap.add_argument("--max-cex-source-lag-s", type=float, default=2.0,
                    help="applies only to feature sets that use CEX-derived features")
    ap.add_argument("--max-rtds-source-age-s", type=float, default=60.0)
    ap.add_argument("--max-rtds-age-s", type=float, default=60.0)
    ap.add_argument("--min-valid-steps", type=int, default=10)
    ap.add_argument("--min-valid-frac", type=float, default=0.10)
    ap.add_argument("--train-end", default="2026-05-13")
    ap.add_argument("--test-start", default="2026-05-14")
    ap.add_argument("--val-days", type=int, default=2)
    ap.add_argument("--compress", action="store_true", help="use compressed npz; slower but smaller")
    args = ap.parse_args()
    manifest = build(args)
    print(json.dumps({
        "output_dir": manifest["output_dir"],
        "features": len(manifest["feature_names"]),
        "splits": manifest["splits"],
        "freshness_policy": manifest["freshness_policy"],
        "elapsed_s": manifest["elapsed_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
