"""Deep policy search for fair_value_v2_source_time model outputs.

This script evaluates many execution policies using the same source-time model
artifacts and a vectorized version of the execution simulator:

* one entry per market
* decision at T, fill quote at T + delay
* fill only when delayed ask is no worse than decision quote + slippage cap
* taker fee included

Selection is kept separate from holdout: May OOS is used to rank candidates,
June OOS is reported as holdout. The script is a research tool, not a live-bot
policy generator.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from search_fair_value_v2_policies import (  # noqa: E402
    ART,
    DS,
    add_time_columns,
    predict_with_artifacts,
)
from train_fair_value_v2_source_time import NS, load_days, taker_fee  # noqa: E402


EV_GRID = [
    0.0,
    0.005,
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
    0.60,
]
EV_GRID_MATRIX = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TTC_EDGES = [3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]
BUCKETS = [(3.0, 15.0), (15.0, 30.0), (30.0, 50.0), (50.0, 75.0), (75.0, 90.0)]
DEFAULT_DELAYS = [1.0, 2.0, 3.0, 4.0]
DEFAULT_SLIPPAGE = [0.01, 0.03, 0.05]


@dataclass(frozen=True)
class Rule:
    lo: float
    hi: float
    ev: float
    hours: tuple[int, ...] | None = None

    def label(self) -> str:
        h = "all" if self.hours is None else "utc" + "".join(f"{x:02d}" for x in self.hours[:2]) + f"_{len(self.hours)}h"
        return f"{self.lo:g}-{self.hi:g}@{self.ev:g}:{h}"

    def to_json(self) -> dict[str, Any]:
        return {
            "ttc_min": self.lo,
            "ttc_max": self.hi,
            "ev_thr": self.ev,
            "hours_utc": list(self.hours) if self.hours is not None else None,
        }


@dataclass(frozen=True)
class Policy:
    label: str
    rules: tuple[Rule, ...]

    def to_json(self) -> dict[str, Any]:
        return {"label": self.label, "rules": [r.to_json() for r in self.rules]}


@dataclass
class SearchData:
    name: str
    market_slug: np.ndarray
    group_id: np.ndarray
    date: np.ndarray
    hour: np.ndarray
    ttc: np.ndarray
    best_ev: np.ndarray
    quote: np.ndarray
    best_is_up: np.ndarray
    y_up: np.ndarray
    book_quality: np.ndarray
    fill_by_delay: dict[float, np.ndarray]
    fill_valid_by_delay: dict[float, np.ndarray]


def _book_quality(frame: pd.DataFrame, book_tol: float) -> np.ndarray:
    mid_sum = frame["up_mid"].astype(float).to_numpy() + frame["down_mid"].astype(float).to_numpy()
    return (
        (np.abs(1.0 - mid_sum) <= book_tol)
        & (frame["up_book_evts_5s"].astype(float).to_numpy() > 0.0)
        & (frame["down_book_evts_5s"].astype(float).to_numpy() > 0.0)
    )


def prepare_search_data(frame: pd.DataFrame, p: np.ndarray, *,
                        name: str, book_tol: float, delays: Iterable[float]) -> SearchData:
    work = frame.copy()
    work["_p"] = p.astype(float)
    work.sort_values(["market_slug", "now_ns"], inplace=True, kind="stable")
    p_sorted = work["_p"].to_numpy(dtype=float)
    up_ask = work["up_best_ask"].astype(float).to_numpy()
    dn_ask = work["down_best_ask"].astype(float).to_numpy()
    ev_up = p_sorted - up_ask
    ev_dn = (1.0 - p_sorted) - dn_ask
    best_is_up = ev_up >= ev_dn
    quote = np.where(best_is_up, up_ask, dn_ask)
    best_ev = np.where(best_is_up, ev_up, ev_dn)

    market = work["market_slug"].astype(str).to_numpy()
    _, group_id = np.unique(market, return_inverse=True)
    now = work["now_ns"].astype(np.int64).to_numpy()
    y_up = work["resolved_up"].astype(np.int8).to_numpy()
    bq = _book_quality(work, book_tol)

    fill_by_delay: dict[float, np.ndarray] = {}
    fill_valid_by_delay: dict[float, np.ndarray] = {}
    for delay in delays:
        fill = np.full(len(work), np.nan, dtype=float)
        valid = np.zeros(len(work), dtype=bool)
        delay_ns = int(round(delay * NS))
        start = 0
        while start < len(work):
            gid = group_id[start]
            end = start + 1
            while end < len(work) and group_id[end] == gid:
                end += 1
            nows = now[start:end]
            js = np.searchsorted(nows, nows + delay_ns, side="left")
            inside = js < len(nows)
            idx = np.arange(start, end)[inside]
            dst = start + js[inside]
            fill[idx] = np.where(best_is_up[idx], up_ask[dst], dn_ask[dst])
            valid[idx] = np.isfinite(fill[idx]) & (fill[idx] > 0.0) & (fill[idx] < 1.0)
            start = end
        fill_by_delay[delay] = fill
        fill_valid_by_delay[delay] = valid

    return SearchData(
        name=name,
        market_slug=market,
        group_id=group_id.astype(np.int32),
        date=work["date"].astype(str).to_numpy(),
        hour=work["_hour_utc"].astype(np.int16).to_numpy(),
        ttc=work["ttc_s"].astype(float).to_numpy(),
        best_ev=best_ev,
        quote=quote,
        best_is_up=best_is_up,
        y_up=y_up,
        book_quality=bq,
        fill_by_delay=fill_by_delay,
        fill_valid_by_delay=fill_valid_by_delay,
    )


def _policy_mask(data: SearchData, policy: Policy) -> np.ndarray:
    mask = np.zeros(len(data.ttc), dtype=bool)
    for rule in policy.rules:
        m = (data.ttc > rule.lo) & (data.ttc <= rule.hi) & (data.best_ev >= rule.ev)
        if rule.hours is not None:
            m &= np.isin(data.hour, np.asarray(rule.hours, dtype=np.int16))
        mask |= m
    return mask


def evaluate_policy(data: SearchData, policy: Policy, *,
                    delay_s: float, slippage_cap: float,
                    fixed_shares: float, price_lo: float, price_hi: float) -> dict[str, Any]:
    fill = data.fill_by_delay[delay_s]
    valid = (
        data.book_quality
        & data.fill_valid_by_delay[delay_s]
        & (data.quote > price_lo)
        & (data.quote < price_hi)
        & (fill <= data.quote + slippage_cap)
        & _policy_mask(data, policy)
    )
    idx = np.flatnonzero(valid)
    if len(idx):
        gids = data.group_id[idx]
        first = idx[np.r_[True, gids[1:] != gids[:-1]]]
    else:
        first = idx
    wins = np.where(data.best_is_up[first], data.y_up[first] == 1, data.y_up[first] == 0)
    fills = fill[first]
    pnl = fixed_shares * (wins.astype(float) - fills - taker_fee(fills))
    cost = fixed_shares * (fills + taker_fee(fills))
    by_day: dict[str, float] = {}
    for d, v in zip(data.date[first], pnl):
        by_day[str(d)] = by_day.get(str(d), 0.0) + float(v)
    by_hour: dict[str, int] = {}
    for h in data.hour[first]:
        key = str(int(h))
        by_hour[key] = by_hour.get(key, 0) + 1
    day_vals = np.asarray(list(by_day.values()), dtype=float)
    return {
        "split": data.name,
        "trades": int(len(first)),
        "wins": int(wins.sum()) if len(first) else 0,
        "win_rate": float(wins.mean()) if len(first) else 0.0,
        "total_pnl": float(pnl.sum()) if len(first) else 0.0,
        "avg_pnl": float(pnl.mean()) if len(first) else 0.0,
        "roi_on_cost": float(pnl.sum() / cost.sum()) if len(first) and float(cost.sum()) > 0.0 else 0.0,
        "active_days": int(len(by_day)),
        "positive_days": int(sum(1 for v in by_day.values() if v > 0.0)),
        "worst_day_pnl": float(day_vals.min()) if len(day_vals) else 0.0,
        "median_day_pnl": float(np.median(day_vals)) if len(day_vals) else 0.0,
        "p25_day_pnl": float(np.quantile(day_vals, 0.25)) if len(day_vals) else 0.0,
        "max_day_pnl": float(day_vals.max()) if len(day_vals) else 0.0,
        "by_hour": dict(sorted(by_hour.items(), key=lambda kv: int(kv[0]))),
    }


def compact(policy: Policy, result: dict[str, Any]) -> dict[str, Any]:
    return {"policy": policy.to_json(), **result}


def eval_splits(splits: dict[str, SearchData], policy: Policy, *,
                delay_s: float, slippage_cap: float,
                fixed_shares: float, price_lo: float, price_hi: float) -> dict[str, Any]:
    return {
        "policy": policy.to_json(),
        **{
            name: evaluate_policy(
                data,
                policy,
                delay_s=delay_s,
                slippage_cap=slippage_cap,
                fixed_shares=fixed_shares,
                price_lo=price_lo,
                price_hi=price_hi,
            )
            for name, data in splits.items()
        },
    }


def hour_filters() -> dict[str, tuple[int, ...] | None]:
    out: dict[str, tuple[int, ...] | None] = {
        "all": None,
        "utc00_08": tuple(range(0, 8)),
        "utc08_20": tuple(range(8, 20)),
        "utc08_14": tuple(range(8, 14)),
        "utc10_20": tuple(range(10, 20)),
        "utc12_20": tuple(range(12, 20)),
        "utc14_20": tuple(range(14, 20)),
        "utc16_20": tuple(range(16, 20)),
        "utc20_24": tuple(range(20, 24)),
    }
    for length in (6, 8, 12):
        for start in range(0, 24, 3):
            out[f"roll{start:02d}_{length}h"] = tuple((start + i) % 24 for i in range(length))
    return out


def make_single_window_policies() -> list[Policy]:
    policies: list[Policy] = []
    hfs = hour_filters()
    for lo, hi in itertools.combinations(TTC_EDGES, 2):
        if hi - lo < 5.0:
            continue
        for ev in EV_GRID:
            for hlabel, hours in hfs.items():
                label = f"single_{lo:g}_{hi:g}_ev{ev:g}_{hlabel}"
                policies.append(Policy(label, (Rule(lo, hi, ev, hours),)))
    return policies


def top_single_rules(may: SearchData, *,
                     fixed_shares: float, price_lo: float, price_hi: float,
                     delay_s: float, slippage_cap: float,
                     min_trades: int) -> dict[str, list[Rule]]:
    out: dict[str, list[Rule]] = {}
    for lo, hi in BUCKETS:
        rows: list[tuple[float, float, Rule]] = []
        for ev in EV_GRID_MATRIX:
            rule = Rule(lo, hi, ev, None)
            policy = Policy(rule.label(), (rule,))
            r = evaluate_policy(
                may,
                policy,
                delay_s=delay_s,
                slippage_cap=slippage_cap,
                fixed_shares=fixed_shares,
                price_lo=price_lo,
                price_hi=price_hi,
            )
            if r["trades"] >= min_trades:
                rows.append((r["total_pnl"], r["avg_pnl"], rule))
        rows.sort(reverse=True, key=lambda x: (x[0], x[1]))
        out[f"{lo:g}-{hi:g}"] = [x[2] for x in rows[:5]]
    return out


def make_matrix_policies(may: SearchData, *,
                         fixed_shares: float, price_lo: float, price_hi: float,
                         delay_s: float, slippage_cap: float,
                         min_bucket_trades: int) -> list[Policy]:
    top = top_single_rules(
        may,
        fixed_shares=fixed_shares,
        price_lo=price_lo,
        price_hi=price_hi,
        delay_s=delay_s,
        slippage_cap=slippage_cap,
        min_trades=min_bucket_trades,
    )
    choices: list[list[Rule | None]] = []
    for lo, hi in BUCKETS:
        choices.append([None] + top.get(f"{lo:g}-{hi:g}", [])[:4])
    policies: list[Policy] = []
    for combo in itertools.product(*choices):
        rules = tuple(r for r in combo if r is not None)
        if not rules:
            continue
        label = "matrix_" + "__".join(r.label() for r in rules)
        policies.append(Policy(label, rules))

    monotonic_options = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]
    for evs in itertools.combinations_with_replacement(monotonic_options, len(BUCKETS)):
        rules = tuple(Rule(lo, hi, ev, None) for (lo, hi), ev in zip(BUCKETS, evs))
        policies.append(Policy("mono_far_higher_" + "_".join(f"{ev:g}" for ev in evs), rules))
    for evs in itertools.combinations_with_replacement(monotonic_options, len(BUCKETS)):
        rev = tuple(reversed(evs))
        rules = tuple(Rule(lo, hi, ev, None) for (lo, hi), ev in zip(BUCKETS, rev))
        policies.append(Policy("mono_close_higher_" + "_".join(f"{ev:g}" for ev in rev), rules))
    return policies


def make_branch_combo_policies() -> list[Policy]:
    policies: list[Policy] = []
    early_windows = [(45.0, 75.0), (50.0, 75.0), (60.0, 75.0), (50.0, 90.0), (60.0, 90.0), (75.0, 90.0)]
    early_hours = {
        "all": None,
        "utc08_20": tuple(range(8, 20)),
        "utc08_14": tuple(range(8, 14)),
        "utc10_20": tuple(range(10, 20)),
        "utc12_20": tuple(range(12, 20)),
        "utc16_20": tuple(range(16, 20)),
    }
    early_evs = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10]
    close_evs = [None, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]
    mid_evs = [None, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    far_evs = [None, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40]
    for elo, ehi in early_windows:
        for hev_label, hours in early_hours.items():
            for early_ev in early_evs:
                for close_ev in close_evs:
                    for mid_ev in mid_evs:
                        for far_ev in far_evs:
                            rules = [Rule(elo, ehi, early_ev, hours)]
                            parts = [f"{elo:g}-{ehi:g}@{early_ev:g}:{hev_label}"]
                            if close_ev is not None:
                                rules.append(Rule(3.0, 15.0, close_ev, None))
                                parts.append(f"3-15@{close_ev:g}")
                            if mid_ev is not None:
                                rules.append(Rule(15.0, 30.0, mid_ev, None))
                                parts.append(f"15-30@{mid_ev:g}")
                            if far_ev is not None:
                                rules.append(Rule(75.0, 90.0, far_ev, hours))
                                parts.append(f"75-90@{far_ev:g}:{hev_label}")
                            policies.append(Policy("branch_" + "__".join(parts), tuple(rules)))
    return policies


def rank_candidates(
    policies: list[Policy],
    splits: dict[str, SearchData],
    *,
    fixed_shares: float,
    price_lo: float,
    price_hi: float,
    delay_s: float,
    slippage_cap: float,
    min_select_trades: int,
    min_holdout_trades: int,
    top_n: int,
    label: str,
) -> dict[str, Any]:
    may_rows: list[tuple[float, float, Policy, dict[str, Any]]] = []
    t0 = time.time()
    for i, policy in enumerate(policies, 1):
        r = evaluate_policy(
            splits["may_select"],
            policy,
            delay_s=delay_s,
            slippage_cap=slippage_cap,
            fixed_shares=fixed_shares,
            price_lo=price_lo,
            price_hi=price_hi,
        )
        if r["trades"] >= min_select_trades and r["total_pnl"] > 0.0:
            may_rows.append((r["total_pnl"], r["avg_pnl"], policy, r))
        if i % 5000 == 0:
            print(f"{label}: checked {i:,}/{len(policies):,}, may_kept={len(may_rows):,}", flush=True)
    may_rows.sort(key=lambda x: (x[0], x[1]), reverse=True)

    full: list[dict[str, Any]] = []
    for _, _, policy, _ in may_rows[:top_n]:
        row = eval_splits(
            splits,
            policy,
            delay_s=delay_s,
            slippage_cap=slippage_cap,
            fixed_shares=fixed_shares,
            price_lo=price_lo,
            price_hi=price_hi,
        )
        full.append(row)
    survivors = [
        r for r in full
        if r["june_holdout"]["trades"] >= min_holdout_trades
        and r["june_holdout"]["total_pnl"] > 0.0
        and r["may_select"]["total_pnl"] > 0.0
    ]
    survivors.sort(
        key=lambda r: (
            r["all_oos"]["total_pnl"],
            r["june_holdout"]["total_pnl"],
            r["all_oos"]["avg_pnl"],
        ),
        reverse=True,
    )
    return {
        "label": label,
        "policies_checked": len(policies),
        "may_positive_kept": len(may_rows),
        "elapsed_s": round(time.time() - t0, 2),
        "top_may": full[:50],
        "survivors": survivors[:80],
    }


def stress_policy(splits: dict[str, SearchData], policy: Policy, *,
                  fixed_shares: float, price_lo: float, price_hi: float) -> list[dict[str, Any]]:
    rows = []
    for delay in DEFAULT_DELAYS:
        for slip in DEFAULT_SLIPPAGE:
            r = evaluate_policy(
                splits["all_oos"],
                policy,
                delay_s=delay,
                slippage_cap=slip,
                fixed_shares=fixed_shares,
                price_lo=price_lo,
                price_hi=price_hi,
            )
            rows.append({"delay_s": delay, "slippage_cap": slip, **r})
    return rows


def print_rows(title: str, rows: list[dict[str, Any]], limit: int = 12) -> None:
    print(f"\n=== {title} ===")
    for r in rows[:limit]:
        label = r["policy"]["label"][:72]
        may, june, allr = r["may_select"], r["june_holdout"], r["all_oos"]
        print(
            f"{label:72s} | "
            f"May {may['trades']:4d} pnl={may['total_pnl']:8.2f} avg={may['avg_pnl']:6.3f} "
            f"days={may['positive_days']}/{may['active_days']} | "
            f"June {june['trades']:4d} pnl={june['total_pnl']:8.2f} avg={june['avg_pnl']:6.3f} "
            f"days={june['positive_days']}/{june['active_days']} | "
            f"All {allr['trades']:4d} pnl={allr['total_pnl']:8.2f} avg={allr['avg_pnl']:6.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=DS)
    ap.add_argument("--artifacts-dir", type=Path, default=ART)
    ap.add_argument("--test-start", default="2026-05-14")
    ap.add_argument("--test-end", default="2026-06-29")
    ap.add_argument("--price-lo", type=float, default=0.10)
    ap.add_argument("--price-hi", type=float, default=0.90)
    ap.add_argument("--book-tol", type=float, default=0.03)
    ap.add_argument("--delay-s", type=float, default=2.0)
    ap.add_argument("--slippage-cap", type=float, default=0.05)
    ap.add_argument("--fixed-shares", type=float, default=5.1)
    ap.add_argument("--min-select-trades", type=int, default=50)
    ap.add_argument("--min-holdout-trades", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=400)
    args = ap.parse_args()

    t0 = time.time()
    test = add_time_columns(load_days(args.dataset_dir, args.test_start, args.test_end))
    if test.empty:
        raise SystemExit("no test rows loaded")
    p = predict_with_artifacts(test, args.artifacts_dir)
    may_mask = (test["date"] <= "2026-05-31").to_numpy()
    june_mask = (test["date"] >= "2026-06-01").to_numpy()
    delays = sorted(set(DEFAULT_DELAYS + [args.delay_s]))
    splits = {
        "may_select": prepare_search_data(test[may_mask].copy(), p[may_mask], name="may_select", book_tol=args.book_tol, delays=delays),
        "june_holdout": prepare_search_data(test[june_mask].copy(), p[june_mask], name="june_holdout", book_tol=args.book_tol, delays=delays),
        "all_oos": prepare_search_data(test.copy(), p, name="all_oos", book_tol=args.book_tol, delays=delays),
    }
    print(
        f"loaded rows={len(test):,} markets={test['market_slug'].nunique():,} "
        f"price=({args.price_lo},{args.price_hi})",
        flush=True,
    )

    single = rank_candidates(
        make_single_window_policies(),
        splits,
        fixed_shares=args.fixed_shares,
        price_lo=args.price_lo,
        price_hi=args.price_hi,
        delay_s=args.delay_s,
        slippage_cap=args.slippage_cap,
        min_select_trades=args.min_select_trades,
        min_holdout_trades=args.min_holdout_trades,
        top_n=args.top_n,
        label="single_window_time",
    )
    matrix = rank_candidates(
        make_matrix_policies(
            splits["may_select"],
            fixed_shares=args.fixed_shares,
            price_lo=args.price_lo,
            price_hi=args.price_hi,
            delay_s=args.delay_s,
            slippage_cap=args.slippage_cap,
            min_bucket_trades=max(10, args.min_select_trades // 4),
        ),
        splits,
        fixed_shares=args.fixed_shares,
        price_lo=args.price_lo,
        price_hi=args.price_hi,
        delay_s=args.delay_s,
        slippage_cap=args.slippage_cap,
        min_select_trades=args.min_select_trades,
        min_holdout_trades=args.min_holdout_trades,
        top_n=args.top_n,
        label="ttc_matrix",
    )
    branch = rank_candidates(
        make_branch_combo_policies(),
        splits,
        fixed_shares=args.fixed_shares,
        price_lo=args.price_lo,
        price_hi=args.price_hi,
        delay_s=args.delay_s,
        slippage_cap=args.slippage_cap,
        min_select_trades=args.min_select_trades,
        min_holdout_trades=args.min_holdout_trades,
        top_n=args.top_n,
        label="branch_combo",
    )
    all_survivors = single["survivors"] + matrix["survivors"] + branch["survivors"]
    all_survivors.sort(
        key=lambda r: (
            r["all_oos"]["total_pnl"],
            r["june_holdout"]["total_pnl"],
            r["all_oos"]["avg_pnl"],
        ),
        reverse=True,
    )
    best_policy = Policy(
        all_survivors[0]["policy"]["label"],
        tuple(Rule(x["ttc_min"], x["ttc_max"], x["ev_thr"],
                   tuple(x["hours_utc"]) if x["hours_utc"] is not None else None)
              for x in all_survivors[0]["policy"]["rules"]),
    ) if all_survivors else None
    stress = stress_policy(
        splits,
        best_policy,
        fixed_shares=args.fixed_shares,
        price_lo=args.price_lo,
        price_hi=args.price_hi,
    ) if best_policy else []
    report = {
        "dataset": str(args.dataset_dir),
        "artifacts_dir": str(args.artifacts_dir),
        "test_start": args.test_start,
        "test_end": args.test_end,
        "selection_split": "may_select",
        "holdout_split": "june_holdout",
        "execution": {
            "price_lo": args.price_lo,
            "price_hi": args.price_hi,
            "delay_s": args.delay_s,
            "slippage_cap": args.slippage_cap,
            "fixed_shares": args.fixed_shares,
            "book_tol": args.book_tol,
        },
        "searches": {
            "single_window_time": single,
            "ttc_matrix": matrix,
            "branch_combo": branch,
        },
        "top_survivors": all_survivors[:100],
        "best_policy_stress": stress,
        "elapsed_s": round(time.time() - t0, 2),
    }
    out = args.artifacts_dir / "deep_policy_search_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_rows("single survivors", single["survivors"])
    print_rows("matrix survivors", matrix["survivors"])
    print_rows("branch survivors", branch["survivors"])
    print_rows("overall top", all_survivors)
    print("\n=== best stress ===")
    for row in stress:
        print(
            f"d={row['delay_s']:g} slip={row['slippage_cap']:.2f} "
            f"n={row['trades']:4d} pnl={row['total_pnl']:8.2f} "
            f"avg={row['avg_pnl']:6.3f} wr={row['win_rate']:.3f} "
            f"days={row['positive_days']}/{row['active_days']}"
        )
    print(f"\nreport -> {out}")
    print(f"elapsed_s={report['elapsed_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
