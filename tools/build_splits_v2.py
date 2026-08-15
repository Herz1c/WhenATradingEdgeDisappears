"""Assemble dataset v2 splits from existing daily episode shards (Phase 3).

v1's fatal flaw was a 2-day val (calibration/selection hazard, documented in
the episode audit). v2 fixes the split scheme without touching the shards:

  - train:    2026-04-19 .. 2026-06-16          (model fitting)
  - val:      2026-06-23 .. 2026-06-27          (early stopping + calibration;
              the 06-17..06-22 recorder outage is a natural embargo)
  - test:     2026-06-28 .. 2026-07-02          (report split for the trainer;
              still selection-era data, NOT the gate)
  - holdout:  2026-07-03 onward, GROWING        (never used for selection; the
              final gate + shadow comparison only; kept as daily shards)
  - walkforward folds over train+val days for policy selection (expanding
              train, 5-day OOS blocks, 1-day embargo)

Outputs to --out (default data/datasets/btc_5m_episodes_v2_200ms):
  train.npz, val.npz               (normalized with v2 train-only stats)
  normalization.json               (recomputed, train valid steps only)
  splits_v2.json                   (day lists + fold definitions + hashes)
  feature_names.json, quote_names.json, audit_names.json (copied from v1)
  daily/ is NOT copied; consumers read v1's daily/ shards directly (single
  source of truth), holdout consumers included.

Usage:
    py tools/build_splits_v2.py                # default split scheme
    py tools/build_splits_v2.py --dry-run      # print day assignment only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1_DIR = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
V2_DIR = ROOT / "data" / "datasets" / "btc_5m_episodes_v2_200ms"

TRAIN_END = "2026-06-16"
VAL_START = "2026-06-23"
VAL_END = "2026-06-27"
TEST_START = "2026-06-28"
TEST_END = "2026-07-02"
HOLDOUT_START = "2026-07-03"
FOLD_TEST_DAYS = 5
FOLD_EMBARGO_DAYS = 1
FOLD_MIN_TRAIN_DAYS = 20

# daily shards store unnormalized X_raw; split npz stores normalized X (v1 format)
CONCAT_KEYS = ["X_raw", "valid_mask", "row_present_mask", "source_valid_mask", "y",
               "p_market", "quotes", "audit", "now_ns", "market_slug", "date",
               "open_s", "close_s", "strike", "valid_count"]


def shard_days(daily_dir: Path) -> list[str]:
    return sorted(p.stem for p in daily_dir.glob("*.npz"))


def make_folds(days: list[str]) -> list[dict]:
    folds = []
    i = FOLD_MIN_TRAIN_DAYS
    while i + FOLD_EMBARGO_DAYS + 1 < len(days):
        test_start = i + FOLD_EMBARGO_DAYS
        test_days = days[test_start: test_start + FOLD_TEST_DAYS]
        if not test_days:
            break
        folds.append({
            "fold": len(folds),
            "train_days": days[:i],
            "embargo_days": days[i: test_start],
            "test_days": test_days,
        })
        i = test_start + len(test_days)
    return folds


def assemble(days: list[str], daily_dir: Path) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {k: [] for k in CONCAT_KEYS}
    for d in days:
        z = np.load(daily_dir / f"{d}.npz", allow_pickle=False)
        for k in CONCAT_KEYS:
            parts[k].append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1-dir", type=Path, default=V1_DIR)
    ap.add_argument("--out", type=Path, default=V2_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    daily_dir = args.v1_dir / "daily"
    days = shard_days(daily_dir)
    train_days = [d for d in days if d <= TRAIN_END]
    val_days = [d for d in days if VAL_START <= d <= VAL_END]
    test_days = [d for d in days if TEST_START <= d <= TEST_END]
    holdout_days = [d for d in days if d >= HOLDOUT_START]
    selection_days = train_days + val_days + test_days
    folds = make_folds(selection_days)

    print(f"train:   {len(train_days)}d  {train_days[0]} .. {train_days[-1]}")
    print(f"val:     {len(val_days)}d  {val_days[0]} .. {val_days[-1]}")
    print(f"test:    {len(test_days)}d  {test_days[0]} .. {test_days[-1]}")
    print(f"holdout: {len(holdout_days)}d  {holdout_days[0]} .. {holdout_days[-1]} (growing)")
    print(f"walkforward folds: {len(folds)} "
          f"(test blocks of {FOLD_TEST_DAYS} available days, embargo {FOLD_EMBARGO_DAYS})")
    if args.dry_run:
        for f in folds:
            print(f"  fold {f['fold']}: train {len(f['train_days'])}d "
                  f"-> test {f['test_days'][0]}..{f['test_days'][-1]}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # pass 1: normalization stats from train valid steps, streamed per shard
    # (mirrors v1 builder: fit on train valid steps only)
    n_feat = None
    s0 = 0
    s1 = None
    s2 = None
    for d in train_days:
        z = np.load(daily_dir / f"{d}.npz", allow_pickle=False)
        flat = z["X_raw"][z["valid_mask"]].astype(np.float64)
        if s1 is None:
            n_feat = flat.shape[1]
            s1 = np.zeros(n_feat)
            s2 = np.zeros(n_feat)
        s0 += flat.shape[0]
        s1 += flat.sum(axis=0)
        s2 += (flat ** 2).sum(axis=0)
    mean = s1 / s0
    std = np.sqrt(np.maximum(s2 / s0 - mean ** 2, 0.0))
    std[std <= 1e-12] = 1.0
    feature_names = json.loads((args.v1_dir / "feature_names.json").read_text(encoding="utf-8"))
    norm = {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_valid_timesteps": int(s0),
        "fit_split": f"train {train_days[0]}..{train_days[-1]}",
    }
    (args.out / "normalization.json").write_text(json.dumps(norm), encoding="utf-8")
    print(f"normalization fit on {s0} valid steps ({time.time() - t0:.0f}s)", flush=True)

    mean32 = mean.astype(np.float32)
    std32 = std.astype(np.float32)
    counts = {}
    for name, day_list in (("train", train_days), ("val", val_days), ("test", test_days)):
        print(f"assembling {name}...", flush=True)
        blob = assemble(day_list, daily_dir)
        Xn = blob.pop("X_raw")            # normalize in place to cap peak RAM
        Xn -= mean32
        Xn /= std32
        Xn[~blob["valid_mask"]] = 0.0
        np.nan_to_num(Xn, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        blob["X"] = Xn
        np.savez(args.out / f"{name}.npz", **blob)
        counts[name] = int(blob["y"].shape[0])
        print(f"{name}.npz written ({counts[name]} markets, "
              f"{time.time() - t0:.0f}s)", flush=True)
        del blob, Xn

    for fn in ("feature_names.json", "quote_names.json", "audit_names.json"):
        src = args.v1_dir / fn
        if src.exists():
            shutil.copy2(src, args.out / fn)

    def _hash(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()

    splits = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scheme": {
            "train_end": TRAIN_END, "val_start": VAL_START, "val_end": VAL_END,
            "test_start": TEST_START, "test_end": TEST_END,
            "holdout_start": HOLDOUT_START,
            "fold_test_days": FOLD_TEST_DAYS, "fold_embargo_days": FOLD_EMBARGO_DAYS,
            "fold_min_train_days": FOLD_MIN_TRAIN_DAYS,
        },
        "train_days": train_days,
        "val_days": val_days,
        "test_days": test_days,
        "holdout_days_at_creation": holdout_days,
        "holdout_note": "holdout grows daily; consumers must glob v1 daily/ for days >= holdout_start",
        "daily_shards_dir": str(daily_dir),
        "walkforward_folds": folds,
        "counts": counts,
        "hashes": {f"{n}.npz": _hash(args.out / f"{n}.npz") for n in counts},
    }
    (args.out / "splits_v2.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    print(f"-> {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
