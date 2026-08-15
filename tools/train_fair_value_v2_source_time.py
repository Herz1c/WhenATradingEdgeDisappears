"""Train and execution-backtest fair_value_v2_source_time."""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "datasets" / "fair_value_v2_source_time"
ART = ROOT / "artifacts" / "fair_value_v2_source_time"
PM_RTDS_DS = ROOT / "data" / "datasets" / "fair_value_v2_pm_rtds"
PM_RTDS_ART = ROOT / "artifacts" / "fair_value_v2_pm_rtds"

META = {
    "market_slug",
    "now_ns",
    "resolved_up",
    "chainlink_close_price",
    "cex_source",
    "dataset_clock_mode",
    "ttc_s",
}
LEVEL_LEAK = {"cex_mid", "strike", "synthetic_chainlink_nowcast", "binance_mid"}
AUDIT_ONLY = {
    "pm_source_lag_s",
    "pm_recv_lag_s",
    "pm_delivery_lag_s",
    "coinbase_source_lag_s",
    "rtds_source_age_s",
    "rtds_latency_s",
}
EXCLUDE = META | LEVEL_LEAK | AUDIT_ONLY
CEX_DERIVED = {
    "synthetic_chainlink_nowcast",
    "delta_to_strike_nowcast",
    "basis_binance_chainlink",
    "btc_realized_vol_15s",
    "delta_over_vol",
    "p_bs",
    "cex_mid",
    "cex_microprice_tilt",
    "cex_spread",
    "btc_ret_1s",
    "btc_ret_3s",
    "btc_ret_5s",
    "btc_ret_10s",
    "btc_depth_imbalance",
    "btc_ofi_1s",
    "btc_ofi_5s",
    "time_frac_above_strike_30s",
    "strike_crossings_30s",
    "book_vs_oracle_gap",
}
CEX_FEATURE_SET_ALLOW = {
    "pm_rtds_safe": set(),
    "cex_oracle_gap": {
        "book_vs_oracle_gap",
    },
    "cex_oracle_core": {
        "book_vs_oracle_gap",
        "delta_to_strike_nowcast",
        "p_bs",
    },
    "cex_ticker_min": {
        "book_vs_oracle_gap",
        "delta_to_strike_nowcast",
        "p_bs",
        "basis_binance_chainlink",
        "btc_realized_vol_15s",
        "btc_ret_10s",
        "delta_over_vol",
    },
}

NS = 1_000_000_000
DEFAULT_TTC_WINDOWS = [(10.0, 45.0), (50.0, 75.0), (15.0, 90.0)]
DEFAULT_DELAYS = [1.0, 2.0, 3.0]
DEFAULT_SLIPPAGE = [0.01, 0.03, 0.05]
STRATEGY_WINDOWS = [(3.0, 15.0), (10.0, 45.0), (15.0, 30.0), (30.0, 60.0),
                    (50.0, 75.0), (60.0, 90.0), (15.0, 90.0)]
STRATEGY_EV_GRID = [0.02, 0.05, 0.075, 0.10, 0.125, 0.15,
                    0.175, 0.20, 0.25, 0.30, 0.40]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p)
    y = y[order].astype(int)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def taker_fee(price: float | np.ndarray) -> float | np.ndarray:
    return 0.072 * price * (1.0 - price)


def load_days(dataset_dir: Path, d0: str, d1: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(dataset_dir / "*.parquet"))):
        d = os.path.basename(f)[:10]
        if d0 <= d <= d1:
            frame = pd.read_parquet(f)
            frame["date"] = d
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def feature_cols(df: pd.DataFrame, *, feature_set: str = "full") -> list[str]:
    if feature_set in CEX_FEATURE_SET_ALLOW:
        extra_exclude = CEX_DERIVED - CEX_FEATURE_SET_ALLOW[feature_set]
    else:
        extra_exclude = set()
    return [
        c for c in df.columns
        if c not in EXCLUDE
        and c not in extra_exclude
        and c != "date"
        and df[c].dtype != object
        and not c.endswith("_delay")
    ]


def probability_report(df: pd.DataFrame, p_model: np.ndarray) -> dict[str, Any]:
    y = df["resolved_up"].astype(int).to_numpy()
    p_bs = (df["p_bs"].astype(float).to_numpy()
            if "p_bs" in df.columns else df["implied_p_up"].astype(float).to_numpy())
    baselines = {
        "model_cal": p_model,
        "book": df["implied_p_up"].astype(float).to_numpy(),
        "p_bs": p_bs,
    }
    return {
        name: {
            "brier": _brier(p, y),
            "logloss": _logloss(p, y),
            "auc": _auc(p, y),
        }
        for name, p in baselines.items()
    }


def _book_quality_mask(df: pd.DataFrame, tol: float) -> np.ndarray:
    mid_sum = df["up_mid"].astype(float).to_numpy() + df["down_mid"].astype(float).to_numpy()
    return (
        (np.abs(1.0 - mid_sum) <= tol)
        & (df["up_book_evts_5s"].astype(float).to_numpy() > 0.0)
        & (df["down_book_evts_5s"].astype(float).to_numpy() > 0.0)
    )


def execution_backtest(
    df: pd.DataFrame,
    p: np.ndarray,
    *,
    label: str,
    ttc_min: float,
    ttc_max: float,
    delay_s: float,
    slippage_cap: float,
    ev_thr: float,
    fixed_shares: float,
    price_lo: float,
    price_hi: float,
    book_quality: bool = True,
    book_tol: float = 0.03,
) -> dict[str, Any]:
    work = df.copy()
    work["_p"] = p.astype(float)
    work.sort_values(["market_slug", "now_ns"], inplace=True, kind="stable")
    delay_ns = int(round(delay_s * NS))
    trades: list[dict[str, Any]] = []
    for slug, g in work.groupby("market_slug", sort=False):
        g = g.reset_index(drop=True)
        now = g["now_ns"].astype(np.int64).to_numpy()
        p_up = g["_p"].to_numpy(dtype=float)
        up_ask = g["up_best_ask"].to_numpy(dtype=float)
        dn_ask = g["down_best_ask"].to_numpy(dtype=float)
        ttc = g["ttc_s"].to_numpy(dtype=float)
        y_up = g["resolved_up"].to_numpy(dtype=int)
        ev_up = p_up - up_ask
        ev_dn = (1.0 - p_up) - dn_ask
        take_up = (
            (ttc > ttc_min) & (ttc <= ttc_max)
            & (ev_up >= ev_thr) & (ev_up >= ev_dn)
            & (up_ask > price_lo) & (up_ask < price_hi)
        )
        take_dn = (
            (ttc > ttc_min) & (ttc <= ttc_max)
            & (ev_dn >= ev_thr) & (ev_dn > ev_up)
            & (dn_ask > price_lo) & (dn_ask < price_hi)
        )
        if book_quality:
            bq = _book_quality_mask(g, book_tol)
            take_up &= bq
            take_dn &= bq
        for i in range(len(g)):
            side = None
            quote = 0.0
            if take_up[i]:
                side = "UP"
                quote = float(up_ask[i])
            elif take_dn[i]:
                side = "DOWN"
                quote = float(dn_ask[i])
            if side is None:
                continue
            j = int(np.searchsorted(now, int(now[i] + delay_ns), side="left"))
            if j >= len(g):
                continue
            delayed_ask = float(up_ask[j] if side == "UP" else dn_ask[j])
            if not (0.0 < delayed_ask < 1.0):
                continue
            if delayed_ask > quote + slippage_cap:
                continue
            win = bool(y_up[i] == 1) if side == "UP" else bool(y_up[i] == 0)
            payoff = 1.0 if win else 0.0
            fee = float(taker_fee(delayed_ask))
            pnl = fixed_shares * (payoff - delayed_ask - fee)
            trades.append({
                "market_slug": slug,
                "now_ns": int(now[i]),
                "fill_ns": int(now[j]),
                "side": side,
                "quote": quote,
                "fill": delayed_ask,
                "ttc_s": float(ttc[i]),
                "win": win,
                "pnl": pnl,
                "p_model": float(p_up[i]),
            })
            break
    pnl = np.asarray([t["pnl"] for t in trades], dtype=float)
    wins = int(sum(1 for t in trades if t["win"]))
    total_cost = float(sum(fixed_shares * (t["fill"] + float(taker_fee(t["fill"]))) for t in trades))
    return {
        "label": label,
        "ttc_min": ttc_min,
        "ttc_max": ttc_max,
        "delay_s": delay_s,
        "slippage_cap": slippage_cap,
        "ev_thr": ev_thr,
        "fixed_shares": fixed_shares,
        "trades": len(trades),
        "wins": wins,
        "win_rate": float(wins / len(trades)) if trades else 0.0,
        "total_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "roi_on_cost": float(pnl.sum() / total_cost) if total_cost > 0 else 0.0,
        "trade_sample": trades[:20],
    }


def strategy_selection_sweep(test: pd.DataFrame, p_test: np.ndarray, *,
                             ev_thr_default: float,
                             fixed_shares: float,
                             price_lo: float,
                             price_hi: float,
                             book_quality: bool) -> dict[str, Any]:
    frame = test.copy()
    frame["_p"] = p_test
    may = frame[frame["date"] <= "2026-05-31"].copy()
    june = frame[frame["date"] >= "2026-06-01"].copy()
    rows = []
    for lo, hi in STRATEGY_WINDOWS:
        for ev_thr in STRATEGY_EV_GRID:
            for split_name, split in (("may_oos_select", may), ("june_holdout", june), ("all_oos", frame)):
                if split.empty:
                    continue
                p = split["_p"].to_numpy(dtype=float)
                data = split.drop(columns=["_p"])
                r = execution_backtest(
                    data, p, label="model_cal",
                    ttc_min=lo, ttc_max=hi,
                    delay_s=2.0, slippage_cap=0.05,
                    ev_thr=ev_thr,
                    fixed_shares=fixed_shares,
                    price_lo=price_lo,
                    price_hi=price_hi,
                    book_quality=book_quality,
                )
                rows.append({
                    "split": split_name,
                    "ttc_min": lo,
                    "ttc_max": hi,
                    "ev_thr": ev_thr,
                    "trades": r["trades"],
                    "win_rate": r["win_rate"],
                    "total_pnl": r["total_pnl"],
                    "avg_pnl": r["avg_pnl"],
                    "roi_on_cost": r["roi_on_cost"],
                })

    def _lookup(split: str, lo: float, hi: float, ev_thr: float) -> dict[str, Any]:
        return next(
            r for r in rows
            if r["split"] == split and r["ttc_min"] == lo and r["ttc_max"] == hi
            and abs(r["ev_thr"] - ev_thr) < 1e-12
        )

    configs = []
    for lo, hi in STRATEGY_WINDOWS:
        for ev_thr in STRATEGY_EV_GRID:
            item = {
                "ttc_min": lo,
                "ttc_max": hi,
                "ev_thr": ev_thr,
                "may_oos_select": _lookup("may_oos_select", lo, hi, ev_thr) if not may.empty else None,
                "june_holdout": _lookup("june_holdout", lo, hi, ev_thr) if not june.empty else None,
                "all_oos": _lookup("all_oos", lo, hi, ev_thr),
            }
            configs.append(item)

    may_ranked = sorted(
        [c for c in configs if c["may_oos_select"] and c["may_oos_select"]["trades"] >= 20],
        key=lambda c: c["may_oos_select"]["total_pnl"],
        reverse=True,
    )
    both_positive = [
        c for c in configs
        if c["may_oos_select"] and c["june_holdout"]
        and c["may_oos_select"]["trades"] >= 10
        and c["june_holdout"]["trades"] >= 10
        and c["may_oos_select"]["total_pnl"] > 0
        and c["june_holdout"]["total_pnl"] > 0
    ]
    both_positive.sort(
        key=lambda c: c["may_oos_select"]["total_pnl"] + c["june_holdout"]["total_pnl"],
        reverse=True,
    )
    default_key = next(
        c for c in configs
        if c["ttc_min"] == 15.0 and c["ttc_max"] == 90.0
        and abs(c["ev_thr"] - ev_thr_default) < 1e-12
    )
    return {
        "policy_delay_s": 2.0,
        "policy_slippage_cap": 0.05,
        "selection_split": "may_oos_select",
        "holdout_split": "june_holdout",
        "default_config": default_key,
        "top_may_configs_min20": may_ranked[:15],
        "both_positive_min10_each": both_positive[:20],
        "configs": configs,
    }


def main() -> int:
    import joblib
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--artifacts-dir", type=Path, default=None)
    ap.add_argument("--feature-set", choices=("full", "pm_rtds_safe", "cex_oracle_gap",
                                             "cex_oracle_core", "cex_ticker_min"),
                    default="full",
                    help="CEX subset. cex_* variants use only ticker-derived oracle features, no CEX trades.")
    ap.add_argument("--train-end", default="2026-05-13")
    ap.add_argument("--test-start", default="2026-05-14")
    ap.add_argument("--test-end", default="2026-12-31")
    ap.add_argument("--ev-thr", type=float, default=0.10)
    ap.add_argument("--fixed-shares", type=float, default=5.1)
    ap.add_argument("--price-lo", type=float, default=0.30)
    ap.add_argument("--price-hi", type=float, default=0.70)
    ap.add_argument("--book-quality", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-pm-source-lag-s", type=float, default=None,
                    help="quarantine rows whose newest PM frame (source time) is older than this "
                         "at emit time. The pre-2026-07 recorder ran minutes behind the PM stream, "
                         "so ~90%% of historical rows carry a stale book; pass e.g. 2.0 to train/"
                         "evaluate on genuinely fresh rows only. Default: off (legacy behaviour).")
    args = ap.parse_args()
    if args.dataset_dir is None:
        args.dataset_dir = DS
    if args.artifacts_dir is None:
        args.artifacts_dir = ART if args.feature_set == "full" else PM_RTDS_ART / args.feature_set

    t0 = time.time()
    train = load_days(args.dataset_dir, "2000-01-01", args.train_end)
    test = load_days(args.dataset_dir, args.test_start, args.test_end)
    if train.empty or test.empty:
        raise SystemExit(f"train rows={len(train)} test rows={len(test)}; need both")
    if args.max_pm_source_lag_s is not None:
        thr = float(args.max_pm_source_lag_s)
        for name, frame in (("train", train), ("test", test)):
            lag = frame["pm_source_lag_s"].astype(float).to_numpy()
            keep = np.isfinite(lag) & (lag <= thr)
            kept_by_day = frame.loc[keep].groupby("date").size()
            total_by_day = frame.groupby("date").size()
            retention = (kept_by_day / total_by_day).fillna(0.0)
            print(f"pm_source_lag_s<={thr}s quarantine [{name}]: keep {int(keep.sum()):,}/{len(frame):,} "
                  f"rows ({100.0 * keep.mean():.1f}%); per-day retention "
                  f"min={retention.min():.2%} median={retention.median():.2%}", flush=True)
            if name == "train":
                train = frame.loc[keep].copy()
            else:
                test = frame.loc[keep].copy()
        if train.empty or test.empty:
            raise SystemExit("quarantine removed all rows; lower --max-pm-source-lag-s or rebuild data")
    feats = feature_cols(train, feature_set=args.feature_set)
    if "implied_p_up" not in feats:
        raise SystemExit("implied_p_up missing from features")
    print(f"train={len(train):,} test={len(test):,} features={len(feats)}", flush=True)

    dates = sorted(train["date"].unique())
    if len(dates) < 4:
        raise SystemExit("need at least 4 train dates for train/calibration split")
    val_dates = set(dates[-2:])
    tr = train[~train["date"].isin(val_dates)].copy()
    va = train[train["date"].isin(val_dates)].copy()

    def xy(frame: pd.DataFrame):
        x = frame[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        y = frame["resolved_up"].astype(int).to_numpy()
        init = _logit(frame["implied_p_up"].astype(float).to_numpy())
        return x, y, init

    xtr, ytr, itr = xy(tr)
    xva, yva, iva = xy(va)
    dtr = lgb.Dataset(xtr, label=ytr, init_score=itr, feature_name=feats)
    dva = lgb.Dataset(xva, label=yva, init_score=iva, reference=dtr)
    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "metric": "binary_logloss",
    }
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=2000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(60), lgb.log_evaluation(50)],
    )

    def predict(frame: pd.DataFrame) -> np.ndarray:
        x = frame[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        init = _logit(frame["implied_p_up"].astype(float).to_numpy())
        raw = booster.predict(x, raw_score=True)
        return _sigmoid(init + raw)

    p_va_raw = predict(va)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_va_raw, yva)

    p_test_raw = predict(test)
    p_test = iso.transform(p_test_raw)
    p_book = test["implied_p_up"].astype(float).to_numpy()
    p_bs = (test["p_bs"].astype(float).to_numpy()
            if "p_bs" in test.columns else test["implied_p_up"].astype(float).to_numpy())
    prob = probability_report(test, p_test)

    stress = []
    for lo, hi in DEFAULT_TTC_WINDOWS:
        for delay in DEFAULT_DELAYS:
            for slip in DEFAULT_SLIPPAGE:
                stress.append(execution_backtest(
                    test, p_test,
                    label="model_cal",
                    ttc_min=lo, ttc_max=hi,
                    delay_s=delay, slippage_cap=slip,
                    ev_thr=args.ev_thr,
                    fixed_shares=args.fixed_shares,
                    price_lo=args.price_lo,
                    price_hi=args.price_hi,
                    book_quality=args.book_quality,
                ))
    primary = next(
        r for r in stress
        if r["ttc_min"] == 15.0 and r["ttc_max"] == 90.0
        and r["delay_s"] == 2.0 and r["slippage_cap"] == 0.05
    )
    baselines = [
        execution_backtest(
            test, p_book, label="book", ttc_min=15.0, ttc_max=90.0,
            delay_s=2.0, slippage_cap=0.05, ev_thr=args.ev_thr,
            fixed_shares=args.fixed_shares, price_lo=args.price_lo, price_hi=args.price_hi,
            book_quality=args.book_quality,
        ),
        execution_backtest(
            test, p_bs, label="p_bs", ttc_min=15.0, ttc_max=90.0,
            delay_s=2.0, slippage_cap=0.05, ev_thr=args.ev_thr,
            fixed_shares=args.fixed_shares, price_lo=args.price_lo, price_hi=args.price_hi,
            book_quality=args.book_quality,
        ),
    ]
    strategy_sweep = strategy_selection_sweep(
        test,
        p_test,
        ev_thr_default=args.ev_thr,
        fixed_shares=args.fixed_shares,
        price_lo=args.price_lo,
        price_hi=args.price_hi,
        book_quality=args.book_quality,
    )
    positive_stress = sum(1 for r in stress if r["total_pnl"] > 0 and r["trades"] > 0)
    broad_policy_alive = primary["total_pnl"] > 0 and primary["trades"] > 0
    holdout_candidates = strategy_sweep["both_positive_min10_each"]
    if not broad_policy_alive and holdout_candidates:
        verdict = "broad_policy_absent_but_narrow_edge_candidate"
    elif not broad_policy_alive:
        verdict = "edge_absent"
    elif positive_stress < max(1, len(stress) // 3):
        verdict = "edge_fragile_not_live_ready"
    else:
        verdict = "edge_present_candidate"

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.artifacts_dir / "model.txt"))
    joblib.dump(iso, args.artifacts_dir / "calibrator.pkl")
    (args.artifacts_dir / "features.json").write_text(json.dumps(feats, indent=2), encoding="utf-8")
    feature_policy = {
        "feature_set": args.feature_set,
        "require_cex": args.feature_set != "pm_rtds_safe",
        "allowed_cex_features": sorted(CEX_FEATURE_SET_ALLOW.get(args.feature_set, CEX_DERIVED)),
        "excluded_features": sorted(
            (CEX_DERIVED - CEX_FEATURE_SET_ALLOW[args.feature_set])
            if args.feature_set in CEX_FEATURE_SET_ALLOW else []
        ),
    }
    (args.artifacts_dir / "feature_policy.json").write_text(json.dumps(feature_policy, indent=2), encoding="utf-8")
    importance = sorted(zip(feats, booster.feature_importance(importance_type="gain")),
                        key=lambda x: -float(x[1]))
    (args.artifacts_dir / "feature_importance.json").write_text(json.dumps(importance, indent=2), encoding="utf-8")
    report = {
        "dataset": str(args.dataset_dir),
        "feature_set": args.feature_set,
        "require_cex": bool(feature_policy["require_cex"]),
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": feats,
        "calibration_dates": sorted(str(d) for d in val_dates),
        "probability": prob,
        "primary_backtest": primary,
        "baseline_backtests": baselines,
        "stress_grid": stress,
        "strategy_selection_sweep": strategy_sweep,
        "positive_stress_cells": positive_stress,
        "stress_cells": len(stress),
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (args.artifacts_dir / "edge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== probability ===")
    for name, row in prob.items():
        print(f"{name:10s} brier={row['brier']:.5f} logloss={row['logloss']:.5f} auc={row['auc']:.4f}")
    print("\n=== primary execution backtest ===")
    print(json.dumps(primary, indent=2)[:4000])
    print(f"\nVERDICT: {verdict}")
    print(f"Artifacts -> {args.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
