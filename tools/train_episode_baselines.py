"""Train baseline models for the 5Hz BTC 5m episode dataset.

The main baseline is market-implied probability (`p_market`).  The tabular
baseline is a LightGBM residual classifier trained with `init_score=logit(p)`;
at inference we explicitly add the tree raw score to the market logit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_OUT = ROOT / "artifacts" / "btc_5m_episode_baselines_v1"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64, copy=False), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64, copy=False), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    p = p.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    if not y.size:
        return float("nan")
    p = np.clip(p.astype(np.float64, copy=False), 1e-6, 1.0 - 1e-6)
    y = y.astype(np.float64, copy=False)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    if y.size < 2:
        return float("nan")
    y = y.astype(np.int8, copy=False)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p.astype(np.float64, copy=False), kind="mergesort")
    ranks = np.arange(1, y.size + 1, dtype=np.float64)
    ys = y[order]
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    if not y.size:
        return float("nan")
    p = np.clip(p.astype(np.float64, copy=False), 0.0, 1.0)
    y = y.astype(np.float64, copy=False)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(y.size)
    acc = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if m.any():
            acc += float(m.sum()) / total * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(acc)


def _metrics(p: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    y = y.astype(np.int8, copy=False)
    return {
        "n": int(y.size),
        "p_mean": float(np.mean(p)) if y.size else float("nan"),
        "y_mean": float(np.mean(y)) if y.size else float("nan"),
        "brier": _brier(p, y),
        "logloss": _logloss(p, y),
        "auc": _auc(p, y),
        "ece_20": _ece(p, y, bins=20),
    }


def _parse_window(text: str) -> tuple[str, float, float]:
    label, span = text.split("=", 1) if "=" in text else (text, text)
    lo_s, hi_s = span.replace(":", "-").split("-", 1)
    lo, hi = float(lo_s), float(hi_s)
    return label, lo, hi


def _ttc_grid(seq_len: int, cadence_s: float) -> np.ndarray:
    return (seq_len * cadence_s - np.arange(seq_len, dtype=np.float32) * cadence_s).astype(np.float32)


def _load_split(dataset: Path, split: str) -> dict[str, Any]:
    z = np.load(dataset / f"{split}.npz", allow_pickle=False)
    return {
        "X": z["X"],
        "valid": z["valid_mask"].astype(bool, copy=False),
        "y": z["y"].astype(np.int8, copy=False),
        "p_market": z["p_market"].astype(np.float32, copy=False),
        "date": z["date"],
        "market_slug": z["market_slug"],
        "valid_count": z["valid_count"],
    }


def _rows_for_window(split: dict[str, Any], ttc: np.ndarray, lo: float, hi: float) -> dict[str, Any]:
    band = (ttc > lo) & (ttc <= hi)
    mask = split["valid"] & band[None, :]
    ep_idx, step_idx = np.nonzero(mask)
    x = split["X"][ep_idx, step_idx, :].astype(np.float32, copy=False)
    p = split["p_market"][ep_idx, step_idx].astype(np.float32, copy=False)
    y = split["y"][ep_idx].astype(np.int8, copy=False)
    return {
        "X": x,
        "p": np.clip(p, 1e-5, 1.0 - 1e-5),
        "base_logit": _logit(p),
        "y": y,
        "ep_idx": ep_idx.astype(np.int32, copy=False),
        "step_idx": step_idx.astype(np.int16, copy=False),
        "date": split["date"][ep_idx],
    }


def _market_level_metrics(split: dict[str, Any], p_step: np.ndarray, ep_idx: np.ndarray) -> dict[str, Any]:
    n_ep = int(split["y"].shape[0])
    sums = np.zeros(n_ep, dtype=np.float64)
    counts = np.zeros(n_ep, dtype=np.int32)
    np.add.at(sums, ep_idx, p_step.astype(np.float64, copy=False))
    np.add.at(counts, ep_idx, 1)
    ok = counts > 0
    p_avg = sums[ok] / counts[ok]
    y = split["y"][ok]
    return _metrics(p_avg.astype(np.float32), y)


def _per_day_metrics(p: np.ndarray, y: np.ndarray, dates: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in sorted(set(str(x) for x in dates.tolist())):
        m = dates == d
        if not np.any(m):
            continue
        row = {"date": d}
        row.update(_metrics(p[m], y[m]))
        out.append(row)
    return out


def evaluate_predictions(
    split_name: str,
    split: dict[str, Any],
    ttc: np.ndarray,
    windows: list[tuple[str, float, float]],
    predictor,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, lo, hi in windows:
        rows = _rows_for_window(split, ttc, lo, hi)
        if rows["y"].size == 0:
            out[label] = {"step": _metrics(np.array([], dtype=np.float32), np.array([], dtype=np.int8))}
            continue
        p = predictor(rows)
        out[label] = {
            "step": _metrics(p, rows["y"]),
            "market_mean": _market_level_metrics(split, p, rows["ep_idx"]),
            "per_day": _per_day_metrics(p, rows["y"], rows["date"]),
        }
    return out


def _maybe_subsample(rows: dict[str, Any], max_rows: int, seed: int) -> dict[str, Any]:
    n = int(rows["y"].size)
    if max_rows <= 0 or n <= max_rows:
        return rows
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in rows.items()}


def train_lgbm_residual(
    train_rows: dict[str, Any],
    val_rows: dict[str, Any],
    *,
    rounds: int,
    early_stopping: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    seed: int,
    num_threads: int,
) -> Any:
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 255,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": num_threads,
    }
    dtrain = lgb.Dataset(
        train_rows["X"],
        label=train_rows["y"],
        init_score=train_rows["base_logit"],
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        val_rows["X"],
        label=val_rows["y"],
        init_score=val_rows["base_logit"],
        reference=dtrain,
        free_raw_data=False,
    )
    callbacks = [
        lgb.early_stopping(early_stopping, verbose=True),
        lgb.log_evaluation(period=50),
    ]
    return lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )


def _write_summary(out_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# BTC 5m Episode Baselines",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Train window: `{report['train_window']['label']}` ({report['train_window']['lo']}, {report['train_window']['hi']}] TTC",
        "",
        "## Headline TTC 15-90",
        "",
        "| model | split | n steps | brier | logloss | auc | ece_20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in ("market", "lgbm_residual"):
        if model_name not in report:
            continue
        for split in ("train", "val", "test"):
            m = report[model_name][split].get("ttc_15_90", {}).get("step")
            if not m:
                continue
            lines.append(
                f"| {model_name} | {split} | {m['n']} | {m['brier']:.6f} | "
                f"{m['logloss']:.6f} | {m['auc']:.6f} | {m['ece_20']:.6f} |"
            )
    lines.extend(["", "## Files", "", "- `baseline_report.json`", "- `comparison_table.md`"])
    (out_dir / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--train-window", default="ttc_15_90=15-90")
    ap.add_argument("--eval-window", action="append",
                    default=["all_valid=0-300", "ttc_10_45=10-45", "ttc_15_75=15-75",
                             "ttc_15_90=15-90", "ttc_50_75=50-75"])
    ap.add_argument("--cadence-s", type=float, default=0.2)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--early-stopping", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--min-data-in-leaf", type=int, default=300)
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="0 means use every valid row in train-window")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--skip-lgbm", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = json.loads((dataset / "feature_names.json").read_text(encoding="utf-8"))
    train = _load_split(dataset, "train")
    val = _load_split(dataset, "val")
    test = _load_split(dataset, "test")
    seq_len = int(train["X"].shape[1])
    ttc = _ttc_grid(seq_len, args.cadence_s)
    windows = [_parse_window(x) for x in args.eval_window]
    train_label, train_lo, train_hi = _parse_window(args.train_window)

    print(f"dataset={dataset}")
    print(f"shapes train={train['X'].shape} val={val['X'].shape} test={test['X'].shape}")
    print(f"train_window={train_label} ({train_lo},{train_hi}] eval_windows={[w[0] for w in windows]}")

    report: dict[str, Any] = {
        "dataset": str(dataset),
        "out_dir": str(out_dir),
        "feature_names": feature_names,
        "train_window": {"label": train_label, "lo": train_lo, "hi": train_hi},
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "split_shapes": {
            "train": list(train["X"].shape),
            "val": list(val["X"].shape),
            "test": list(test["X"].shape),
        },
    }

    print("evaluating market baseline ...")
    market_predictor = lambda rows: rows["p"]
    report["market"] = {
        split_name: evaluate_predictions(split_name, split, ttc, windows, market_predictor)
        for split_name, split in (("train", train), ("val", val), ("test", test))
    }

    if not args.skip_lgbm:
        print("materializing LGBM train/val rows ...")
        train_rows = _maybe_subsample(_rows_for_window(train, ttc, train_lo, train_hi),
                                      args.max_train_rows, args.seed)
        val_rows = _rows_for_window(val, ttc, train_lo, train_hi)
        print(f"lgbm rows train={train_rows['X'].shape} val={val_rows['X'].shape}")
        booster = train_lgbm_residual(
            train_rows,
            val_rows,
            rounds=args.rounds,
            early_stopping=args.early_stopping,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            seed=args.seed,
            num_threads=args.num_threads,
        )

        def lgbm_predictor(rows: dict[str, Any]) -> np.ndarray:
            raw_delta = booster.predict(rows["X"], raw_score=True, num_iteration=booster.best_iteration)
            return _sigmoid(rows["base_logit"] + raw_delta.astype(np.float32, copy=False))

        print("evaluating LGBM residual baseline ...")
        report["lgbm_residual"] = {
            split_name: evaluate_predictions(split_name, split, ttc, windows, lgbm_predictor)
            for split_name, split in (("train", train), ("val", val), ("test", test))
        }
        report["lgbm_residual"]["best_iteration"] = int(booster.best_iteration or booster.current_iteration())
        booster.save_model(str(out_dir / "lgbm_residual_model.txt"))
        importance = {
            name: float(score)
            for name, score in sorted(
                zip(feature_names, booster.feature_importance(importance_type="gain")),
                key=lambda x: x[1],
                reverse=True,
            )
        }
        (out_dir / "lgbm_feature_importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")

    report["elapsed_s"] = round(time.time() - t0, 2)
    (out_dir / "baseline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary(out_dir, report)

    headline = {
        model: {
            split: report[model][split]["ttc_15_90"]["step"]
            for split in ("train", "val", "test")
        }
        for model in ("market", "lgbm_residual")
        if model in report
    }
    print("\nHEADLINE ttc_15_90 step metrics")
    print(json.dumps(headline, indent=2))
    print(f"report -> {out_dir / 'baseline_report.json'}")
    print(f"summary -> {out_dir / 'comparison_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
