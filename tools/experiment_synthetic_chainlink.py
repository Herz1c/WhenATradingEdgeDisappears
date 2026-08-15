"""Sweep synthetic-Chainlink construction parameters and score each against recorded resolution prices."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chainlink_recorder.synthetic_chainlink import (  # noqa: E402
    SECOND_NS,
    SOURCE_NAMES,
    STALE_NS,
    BINANCE_USDM_QUARANTINE_HOURS,
    _flatten_ground_truth,
    _flatten_updates,
    _parse_binance_ws_file,
    _parse_hyperliquid_ws_file,
    _parse_public_delayed_file,
    _valid_clean_files,
)


def _thread_map(loader, paths: list[str], *, max_workers: int):
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(loader, paths))


def _load_updates_threaded(root: Path, source_name: str, *, max_workers: int):
    if source_name == "binance_spot":
        prefix = Path("binance") / "spot" / "BTCUSDT"
        files = _valid_clean_files(root=root, relative_prefix=prefix, kind="ws")
        chunks = _thread_map(_parse_binance_ws_file, [str(path) for _, _, path in files], max_workers=max_workers)
        return _flatten_updates(chunks)
    if source_name == "binance_usdm":
        prefix = Path("binance") / "usdm" / "BTCUSDT"
        files = _valid_clean_files(
            root=root,
            relative_prefix=prefix,
            kind="ws",
            excluded_hours=BINANCE_USDM_QUARANTINE_HOURS,
        )
        chunks = _thread_map(_parse_binance_ws_file, [str(path) for _, _, path in files], max_workers=max_workers)
        return _flatten_updates(chunks)
    if source_name == "hyperliquid":
        prefix = Path("hyperliquid") / "linear" / "BTC"
        files = _valid_clean_files(root=root, relative_prefix=prefix, kind="ws")
        chunks = _thread_map(_parse_hyperliquid_ws_file, [str(path) for _, _, path in files], max_workers=max_workers)
        return _flatten_updates(chunks)
    raise ValueError(f"Unsupported source {source_name}")


def _load_ground_truth_threaded(root: Path, *, max_workers: int):
    prefix = Path("chainlink_public_delayed") / "public_stream_page" / "BTCUSD"
    files = _valid_clean_files(root=root, relative_prefix=prefix, kind="page")
    chunks = _thread_map(_parse_public_delayed_file, [str(path) for _, _, path in files], max_workers=max_workers)
    return _flatten_ground_truth(chunks)


def build_frame(root: Path, *, max_workers: int) -> pd.DataFrame:
    source_updates = {
        name: _load_updates_threaded(root=root, source_name=name, max_workers=max_workers)
        for name in SOURCE_NAMES
    }
    ground_truth = _load_ground_truth_threaded(root=root, max_workers=max_workers)

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
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--cache-path", type=Path, default=Path("data/_cache/synthetic_chainlink_experiment.parquet"))
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path: Path = args.cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.rebuild_cache:
        frame = pd.read_parquet(cache_path)
        print("loaded_cache", {"path": str(cache_path), "rows": int(len(frame))}, flush=True)
    else:
        print("building_frame", {"root": str(args.root), "max_workers": args.max_workers}, flush=True)
        frame = build_frame(args.root, max_workers=args.max_workers)
        frame.to_parquet(cache_path, index=False)
        print("saved_cache", {"path": str(cache_path), "rows": int(len(frame))}, flush=True)

    print(
        "baseline",
        {
            "medae": float(frame["baseline_abs"].median()),
            "p90": float(frame["baseline_abs"].quantile(0.9)),
            "within_3": float((frame["baseline_abs"] <= 3.0).mean()),
        },
        flush=True,
    )

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [
        c
        for c in frame.columns
        if c
        not in {"ts_ns", "dt", "dow", "date", "chainlink_price", "baseline", "baseline_abs"}
        and pd.api.types.is_numeric_dtype(frame[c])
    ]
    cat = ["dow"]
    x = frame[numeric + cat]
    y = frame["chainlink_price"]

    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ]
    )

    models = {
        "linear": Pipeline([("pre", pre), ("model", LinearRegression())]),
        "ridge": Pipeline(
            [("pre", pre), ("model", RidgeCV(alphas=np.logspace(-3, 5, 24)))]
        ),
        "extra_trees": Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    ExtraTreesRegressor(
                        random_state=0,
                        n_estimators=600,
                        max_depth=14,
                        min_samples_leaf=3,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    RandomForestRegressor(
                        random_state=0,
                        n_estimators=500,
                        max_depth=16,
                        min_samples_leaf=3,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gbrt": Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=0,
                        max_depth=6,
                        max_iter=500,
                        learning_rate=0.03,
                    ),
                ),
            ]
        ),
    }

    for name, model in models.items():
        model.fit(x, y)
        pred = model.predict(x)
        abs_err = np.abs(pred - y)
        print(
            "IN",
            name,
            {
                "medae": float(np.median(abs_err)),
                "p90": float(np.quantile(abs_err, 0.9)),
                "within_3": float(np.mean(abs_err <= 3.0)),
            },
            flush=True,
        )

    for name, model in models.items():
        preds = np.full(len(frame), np.nan)
        for held in sorted(frame["date"].unique()):
            tr = frame["date"] != held
            te = frame["date"] == held
            model.fit(x.loc[tr], y.loc[tr])
            preds[te] = model.predict(x.loc[te])
        abs_err = np.abs(preds - y)
        print(
            "LODO",
            name,
            {
                "medae": float(np.nanmedian(abs_err)),
                "p90": float(np.nanquantile(abs_err, 0.9)),
                "within_3": float(np.nanmean(abs_err <= 3.0)),
            },
            flush=True,
        )
        for held in sorted(frame["date"].unique()):
            mask = frame["date"] == held
            held_abs = abs_err[mask]
            print(
                " ",
                held,
                {
                    "medae": float(np.nanmedian(held_abs)),
                    "p90": float(np.nanquantile(held_abs, 0.9)),
                    "within_3": float(np.nanmean(held_abs <= 3.0)),
                    "count": int(mask.sum()),
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
