"""Search source-time fair-value policies without retraining the model.

The search is intentionally split into a May OOS selection slice and a June
holdout slice so the output is useful as a robustness check rather than just a
leaderboard fitted to all available outcomes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_fair_value_v2_source_time import (  # noqa: E402
    ART,
    DS,
    NS,
    _logit,
    _sigmoid,
    load_days,
    taker_fee,
)


EV_GRID = [
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.175,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
]
TTC_EDGES = [3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]
BUCKETS = [(3.0, 15.0), (15.0, 30.0), (30.0, 50.0), (50.0, 75.0), (75.0, 90.0)]


@dataclass(frozen=True)
class HourFilter:
    label: str
    hours_utc: tuple[int, ...] | None


@dataclass(frozen=True)
class Policy:
    label: str
    rules: tuple[tuple[float, float, float], ...]
    hour_filter: HourFilter = HourFilter("all", None)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rules": [
                {"ttc_min": lo, "ttc_max": hi, "ev_thr": ev}
                for lo, hi, ev in self.rules
            ],
            "hour_filter": {
                "label": self.hour_filter.label,
                "hours_utc": list(self.hour_filter.hours_utc) if self.hour_filter.hours_utc is not None else None,
            },
        }


@dataclass
class Group:
    slug: str
    date: np.ndarray
    now: np.ndarray
    p: np.ndarray
    up_ask: np.ndarray
    dn_ask: np.ndarray
    ttc: np.ndarray
    y_up: np.ndarray
    hour_utc: np.ndarray
    book_quality: np.ndarray


def predict_with_artifacts(frame: pd.DataFrame, artifacts_dir: Path) -> np.ndarray:
    import joblib
    import lightgbm as lgb

    feats = json.loads((artifacts_dir / "features.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(artifacts_dir / "model.txt"))
    calibrator = joblib.load(artifacts_dir / "calibrator.pkl")
    x = frame[feats].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    init = _logit(frame["implied_p_up"].astype(float).to_numpy())
    raw = booster.predict(x, raw_score=True)
    return calibrator.transform(_sigmoid(init + raw))


def add_time_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["now_ns"].astype(np.int64), unit="ns", utc=True)
    out["_hour_utc"] = ts.dt.hour.astype(np.int16)
    return out


def prepare_groups(frame: pd.DataFrame, p: np.ndarray, *, book_tol: float) -> list[Group]:
    work = frame.copy()
    work["_p"] = p.astype(float)
    work.sort_values(["market_slug", "now_ns"], inplace=True, kind="stable")
    mid_sum = work["up_mid"].astype(float).to_numpy() + work["down_mid"].astype(float).to_numpy()
    work["_book_quality"] = (
        (np.abs(1.0 - mid_sum) <= book_tol)
        & (work["up_book_evts_5s"].astype(float).to_numpy() > 0.0)
        & (work["down_book_evts_5s"].astype(float).to_numpy() > 0.0)
    )
    groups: list[Group] = []
    for slug, g in work.groupby("market_slug", sort=False):
        groups.append(Group(
            slug=str(slug),
            date=g["date"].astype(str).to_numpy(),
            now=g["now_ns"].astype(np.int64).to_numpy(),
            p=g["_p"].astype(float).to_numpy(),
            up_ask=g["up_best_ask"].astype(float).to_numpy(),
            dn_ask=g["down_best_ask"].astype(float).to_numpy(),
            ttc=g["ttc_s"].astype(float).to_numpy(),
            y_up=g["resolved_up"].astype(np.int8).to_numpy(),
            hour_utc=g["_hour_utc"].astype(np.int16).to_numpy(),
            book_quality=g["_book_quality"].astype(bool).to_numpy(),
        ))
    return groups


def _rule_threshold(ttc: np.ndarray, rules: tuple[tuple[float, float, float], ...]) -> np.ndarray:
    thr = np.full(len(ttc), np.nan, dtype=float)
    for lo, hi, ev in rules:
        m = (ttc > lo) & (ttc <= hi)
        thr[m] = ev
    return thr


def backtest_policy(
    groups: list[Group],
    policy: Policy,
    *,
    delay_s: float,
    slippage_cap: float,
    fixed_shares: float,
    price_lo: float,
    price_hi: float,
) -> dict[str, Any]:
    delay_ns = int(round(delay_s * NS))
    hours = None if policy.hour_filter.hours_utc is None else np.asarray(policy.hour_filter.hours_utc, dtype=np.int16)
    trades: list[dict[str, Any]] = []
    for g in groups:
        if len(g.now) == 0:
            continue
        thr = _rule_threshold(g.ttc, policy.rules)
        eligible = np.isfinite(thr) & g.book_quality
        if hours is not None:
            eligible &= np.isin(g.hour_utc, hours)
        ev_up = g.p - g.up_ask
        ev_dn = (1.0 - g.p) - g.dn_ask
        take_up = (
            eligible
            & (ev_up >= thr)
            & (ev_up >= ev_dn)
            & (g.up_ask > price_lo)
            & (g.up_ask < price_hi)
        )
        take_dn = (
            eligible
            & (ev_dn >= thr)
            & (ev_dn > ev_up)
            & (g.dn_ask > price_lo)
            & (g.dn_ask < price_hi)
        )
        for i in np.flatnonzero(take_up | take_dn):
            side = "UP" if take_up[i] else "DOWN"
            quote = float(g.up_ask[i] if side == "UP" else g.dn_ask[i])
            j = int(np.searchsorted(g.now, int(g.now[i] + delay_ns), side="left"))
            if j >= len(g.now):
                continue
            fill = float(g.up_ask[j] if side == "UP" else g.dn_ask[j])
            if not (0.0 < fill < 1.0):
                continue
            if fill > quote + slippage_cap:
                continue
            win = bool(g.y_up[i] == 1) if side == "UP" else bool(g.y_up[i] == 0)
            fee = float(taker_fee(fill))
            pnl = fixed_shares * ((1.0 if win else 0.0) - fill - fee)
            trades.append({
                "market_slug": g.slug,
                "date": str(g.date[i]),
                "now_ns": int(g.now[i]),
                "side": side,
                "quote": quote,
                "fill": fill,
                "ttc_s": float(g.ttc[i]),
                "hour_utc": int(g.hour_utc[i]),
                "win": win,
                "pnl": float(pnl),
                "p_model": float(g.p[i]),
            })
            break
    pnl = np.asarray([t["pnl"] for t in trades], dtype=float)
    wins = int(sum(1 for t in trades if t["win"]))
    total_cost = float(sum(fixed_shares * (t["fill"] + float(taker_fee(t["fill"]))) for t in trades))
    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + float(t["pnl"])
    return {
        "policy": policy.to_json(),
        "delay_s": delay_s,
        "slippage_cap": slippage_cap,
        "trades": len(trades),
        "wins": wins,
        "win_rate": float(wins / len(trades)) if trades else 0.0,
        "total_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "avg_pnl": float(pnl.mean()) if len(pnl) else 0.0,
        "roi_on_cost": float(pnl.sum() / total_cost) if total_cost > 0 else 0.0,
        "positive_days": int(sum(1 for v in by_day.values() if v > 0.0)),
        "active_days": int(len(by_day)),
        "by_day": dict(sorted(by_day.items())),
        "trade_sample": trades[:15],
    }


def make_windows() -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, hi in itertools.combinations(TTC_EDGES, 2):
        if hi - lo >= 10.0:
            out.append((lo, hi))
    return out


def make_hour_filters() -> list[HourFilter]:
    filters = [HourFilter("all", None)]
    for length in (4, 8, 12):
        for start in range(0, 24, 4):
            hours = tuple((start + i) % 24 for i in range(length))
            filters.append(HourFilter(f"utc_{start:02d}_{length}h", hours))
    return filters


def compact_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "policy": result["policy"],
        "trades": result["trades"],
        "wins": result["wins"],
        "win_rate": result["win_rate"],
        "total_pnl": result["total_pnl"],
        "avg_pnl": result["avg_pnl"],
        "roi_on_cost": result["roi_on_cost"],
        "active_days": result["active_days"],
        "positive_days": result["positive_days"],
    }


def evaluate_on_splits(
    policy: Policy,
    groups_by_split: dict[str, list[Group]],
    *,
    delay_s: float,
    slippage_cap: float,
    fixed_shares: float,
    price_lo: float,
    price_hi: float,
) -> dict[str, Any]:
    out = {"policy": policy.to_json()}
    for split, groups in groups_by_split.items():
        out[split] = compact_result(
            split,
            backtest_policy(
                groups,
                policy,
                delay_s=delay_s,
                slippage_cap=slippage_cap,
                fixed_shares=fixed_shares,
                price_lo=price_lo,
                price_hi=price_hi,
            ),
        )
    return out


def search_single_window_time(
    groups_by_split: dict[str, list[Group]],
    *,
    delay_s: float,
    slippage_cap: float,
    fixed_shares: float,
    price_lo: float,
    price_hi: float,
    min_select_trades: int,
    min_holdout_trades: int,
) -> dict[str, Any]:
    windows = make_windows()
    hour_filters = make_hour_filters()
    base_ranked: list[dict[str, Any]] = []
    may_groups = groups_by_split["may_select"]
    for lo, hi in windows:
        for ev in EV_GRID:
            policy = Policy(f"ttc_{lo:g}_{hi:g}_ev_{ev:g}", ((lo, hi, ev),))
            r = backtest_policy(
                may_groups,
                policy,
                delay_s=delay_s,
                slippage_cap=slippage_cap,
                fixed_shares=fixed_shares,
                price_lo=price_lo,
                price_hi=price_hi,
            )
            if r["trades"] >= min_select_trades:
                base_ranked.append({"policy": policy, "may": compact_result("may_select", r)})
    base_ranked.sort(key=lambda x: (x["may"]["total_pnl"], x["may"]["avg_pnl"]), reverse=True)
    print(f"single-window base candidates={len(base_ranked)}", flush=True)

    expanded: list[dict[str, Any]] = []
    for base in base_ranked[:40]:
        base_policy: Policy = base["policy"]
        lo, hi, ev = base_policy.rules[0]
        for hf in hour_filters:
            policy = Policy(f"ttc_{lo:g}_{hi:g}_ev_{ev:g}_{hf.label}", ((lo, hi, ev),), hf)
            may = backtest_policy(
                may_groups,
                policy,
                delay_s=delay_s,
                slippage_cap=slippage_cap,
                fixed_shares=fixed_shares,
                price_lo=price_lo,
                price_hi=price_hi,
            )
            if may["trades"] >= min_select_trades:
                expanded.append({
                    "policy": policy,
                    "may_select": compact_result("may_select", may),
                })
    expanded.sort(key=lambda x: (x["may_select"]["total_pnl"], x["may_select"]["avg_pnl"]), reverse=True)
    print(f"single-window expanded candidates={len(expanded)}", flush=True)

    evaluated = []
    for item in expanded[:120]:
        p = item["policy"]
        row = evaluate_on_splits(
            p,
            groups_by_split,
            delay_s=delay_s,
            slippage_cap=slippage_cap,
            fixed_shares=fixed_shares,
            price_lo=price_lo,
            price_hi=price_hi,
        )
        evaluated.append(row)

    survivors = [
        r for r in evaluated
        if r["may_select"]["trades"] >= min_select_trades
        and r["june_holdout"]["trades"] >= min_holdout_trades
        and r["may_select"]["total_pnl"] > 0.0
        and r["june_holdout"]["total_pnl"] > 0.0
    ]
    survivors.sort(key=lambda r: (r["june_holdout"]["total_pnl"], r["all_oos"]["total_pnl"]), reverse=True)
    return {
        "base_top_may": [
            {"policy": x["policy"].to_json(), "may_select": x["may"]}
            for x in base_ranked[:40]
        ],
        "expanded_top_may": [
            evaluate_on_splits(
                x["policy"],
                groups_by_split,
                delay_s=delay_s,
                slippage_cap=slippage_cap,
                fixed_shares=fixed_shares,
                price_lo=price_lo,
                price_hi=price_hi,
            )
            for x in expanded[:40]
        ],
        "survivors_may_positive_june_positive": survivors[:40],
    }


def search_bucket_matrix(
    groups_by_split: dict[str, list[Group]],
    *,
    delay_s: float,
    slippage_cap: float,
    fixed_shares: float,
    price_lo: float,
    price_hi: float,
    min_select_trades: int,
    min_holdout_trades: int,
) -> dict[str, Any]:
    may_groups = groups_by_split["may_select"]
    choices_by_bucket: list[list[tuple[float, float, float] | None]] = []
    single_bucket_top: dict[str, list[dict[str, Any]]] = {}
    for lo, hi in BUCKETS:
        rows = []
        for ev in EV_GRID:
            policy = Policy(f"bucket_{lo:g}_{hi:g}_ev_{ev:g}", ((lo, hi, ev),))
            may = backtest_policy(
                may_groups,
                policy,
                delay_s=delay_s,
                slippage_cap=slippage_cap,
                fixed_shares=fixed_shares,
                price_lo=price_lo,
                price_hi=price_hi,
            )
            if may["trades"] >= max(5, min_select_trades // 3):
                rows.append({"rule": (lo, hi, ev), "may": compact_result("may_select", may)})
        rows.sort(key=lambda x: (x["may"]["total_pnl"], x["may"]["avg_pnl"]), reverse=True)
        single_bucket_top[f"{lo:g}-{hi:g}"] = rows[:10]
        choices = [None] + [tuple(x["rule"]) for x in rows[:2]]
        choices_by_bucket.append(choices)
    print("bucket single choices prepared", flush=True)

    evaluated = []
    for combo in itertools.product(*choices_by_bucket):
        rules = tuple(rule for rule in combo if rule is not None)
        if not rules:
            continue
        label = "matrix_" + "__".join(f"{lo:g}_{hi:g}_{ev:g}" for lo, hi, ev in rules)
        policy = Policy(label, rules)
        may = backtest_policy(
            may_groups,
            policy,
            delay_s=delay_s,
            slippage_cap=slippage_cap,
            fixed_shares=fixed_shares,
            price_lo=price_lo,
            price_hi=price_hi,
        )
        if may["trades"] >= min_select_trades:
            evaluated.append({"policy": policy, "may_select": compact_result("may_select", may)})
    evaluated.sort(key=lambda x: (x["may_select"]["total_pnl"], x["may_select"]["avg_pnl"]), reverse=True)
    print(f"bucket matrix candidates={len(evaluated)}", flush=True)

    full = [
        evaluate_on_splits(
            x["policy"],
            groups_by_split,
            delay_s=delay_s,
            slippage_cap=slippage_cap,
            fixed_shares=fixed_shares,
            price_lo=price_lo,
            price_hi=price_hi,
        )
        for x in evaluated[:250]
    ]
    survivors = [
        r for r in full
        if r["may_select"]["trades"] >= min_select_trades
        and r["june_holdout"]["trades"] >= min_holdout_trades
        and r["may_select"]["total_pnl"] > 0.0
        and r["june_holdout"]["total_pnl"] > 0.0
    ]
    survivors.sort(key=lambda r: (r["june_holdout"]["total_pnl"], r["all_oos"]["total_pnl"]), reverse=True)
    return {
        "buckets": [{"ttc_min": lo, "ttc_max": hi} for lo, hi in BUCKETS],
        "single_bucket_top_may": single_bucket_top,
        "matrix_top_may": full[:40],
        "survivors_may_positive_june_positive": survivors[:40],
    }


def print_rows(title: str, rows: list[dict[str, Any]], *, limit: int = 8) -> None:
    print(f"\n=== {title} ===")
    for row in rows[:limit]:
        pol = row["policy"]
        rules = ",".join(f"{r['ttc_min']:g}-{r['ttc_max']:g}@{r['ev_thr']:g}" for r in pol["rules"])
        hf = pol["hour_filter"]["label"]
        may = row["may_select"]
        june = row["june_holdout"]
        all_oos = row["all_oos"]
        print(
            f"{rules:45s} {hf:12s} | "
            f"May {may['trades']:4d} pnl={may['total_pnl']:8.2f} avg={may['avg_pnl']:6.3f} | "
            f"June {june['trades']:4d} pnl={june['total_pnl']:8.2f} avg={june['avg_pnl']:6.3f} | "
            f"All {all_oos['trades']:4d} pnl={all_oos['total_pnl']:8.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=DS)
    ap.add_argument("--artifacts-dir", type=Path, default=ART)
    ap.add_argument("--test-start", default="2026-05-14")
    ap.add_argument("--test-end", default="2026-06-29")
    ap.add_argument("--delay-s", type=float, default=2.0)
    ap.add_argument("--slippage-cap", type=float, default=0.05)
    ap.add_argument("--fixed-shares", type=float, default=5.1)
    ap.add_argument("--price-lo", type=float, default=0.30)
    ap.add_argument("--price-hi", type=float, default=0.70)
    ap.add_argument("--book-tol", type=float, default=0.03)
    ap.add_argument("--min-select-trades", type=int, default=20)
    ap.add_argument("--min-holdout-trades", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    test = load_days(args.dataset_dir, args.test_start, args.test_end)
    if test.empty:
        raise SystemExit("no test rows loaded")
    test = add_time_columns(test)
    p = predict_with_artifacts(test, args.artifacts_dir)
    may_mask = test["date"] <= "2026-05-31"
    june_mask = test["date"] >= "2026-06-01"
    groups_by_split = {
        "may_select": prepare_groups(test[may_mask].copy(), p[may_mask.to_numpy()], book_tol=args.book_tol),
        "june_holdout": prepare_groups(test[june_mask].copy(), p[june_mask.to_numpy()], book_tol=args.book_tol),
        "all_oos": prepare_groups(test.copy(), p, book_tol=args.book_tol),
    }
    print(
        f"loaded rows={len(test):,} may_groups={len(groups_by_split['may_select']):,} "
        f"june_groups={len(groups_by_split['june_holdout']):,}",
        flush=True,
    )

    single = search_single_window_time(
        groups_by_split,
        delay_s=args.delay_s,
        slippage_cap=args.slippage_cap,
        fixed_shares=args.fixed_shares,
        price_lo=args.price_lo,
        price_hi=args.price_hi,
        min_select_trades=args.min_select_trades,
        min_holdout_trades=args.min_holdout_trades,
    )
    matrix = search_bucket_matrix(
        groups_by_split,
        delay_s=args.delay_s,
        slippage_cap=args.slippage_cap,
        fixed_shares=args.fixed_shares,
        price_lo=args.price_lo,
        price_hi=args.price_hi,
        min_select_trades=args.min_select_trades,
        min_holdout_trades=args.min_holdout_trades,
    )
    report = {
        "dataset": str(args.dataset_dir),
        "artifacts_dir": str(args.artifacts_dir),
        "test_start": args.test_start,
        "test_end": args.test_end,
        "selection_split": "may_select",
        "holdout_split": "june_holdout",
        "execution": {
            "delay_s": args.delay_s,
            "slippage_cap": args.slippage_cap,
            "fixed_shares": args.fixed_shares,
            "price_lo": args.price_lo,
            "price_hi": args.price_hi,
            "book_tol": args.book_tol,
        },
        "single_window_time_search": single,
        "ttc_bucket_matrix_search": matrix,
        "elapsed_s": round(time.time() - t0, 2),
    }
    out = args.artifacts_dir / "policy_search_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_rows(
        "single-window/time survivors (May>0, June>0)",
        single["survivors_may_positive_june_positive"],
    )
    print_rows(
        "TTC bucket matrix survivors (May>0, June>0)",
        matrix["survivors_may_positive_june_positive"],
    )
    print(f"\nreport -> {out}")
    print(f"elapsed_s={report['elapsed_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
