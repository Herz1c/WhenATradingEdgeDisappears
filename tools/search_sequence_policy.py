"""Search trading policies over market/LGBM/sequence episode predictions.

This is an exploratory policy layer on top of the source-time 5Hz episode
dataset.  It uses realistic delayed execution: decision quote at T, fill quote
at T + delay_s, fill only if the delayed ask is within the slippage cap.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_TCN = ROOT / "artifacts" / "btc_5m_episode_tcn_c64_b7_cal_ttc15_90"
DEFAULT_LGBM = ROOT / "artifacts" / "btc_5m_episode_baselines_allvalid_v1"
DEFAULT_OUT = ROOT / "artifacts" / "btc_5m_sequence_policy_search_tcn_v1"

QUOTE_UP_BID = 0
QUOTE_UP_ASK = 1
QUOTE_DN_BID = 2
QUOTE_DN_ASK = 3
NS = 1_000_000_000


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64, copy=False), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64, copy=False), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def taker_fee(price: np.ndarray | float) -> np.ndarray | float:
    return 0.072 * price * (1.0 - price)


@dataclass(frozen=True)
class PolicySpec:
    ttc_min: float
    ttc_max: float
    ev: float
    price_lo: float
    price_hi: float
    slippage: float
    hour_label: str
    hours_utc: tuple[int, ...] | None

    @property
    def label(self) -> str:
        return (
            f"ttc_{self.ttc_min:g}_{self.ttc_max:g}"
            f"__ev_{self.ev:g}"
            f"__px_{self.price_lo:g}_{self.price_hi:g}"
            f"__slip_{self.slippage:g}"
            f"__{self.hour_label}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "ttc_min": self.ttc_min,
            "ttc_max": self.ttc_max,
            "ev": self.ev,
            "price_lo": self.price_lo,
            "price_hi": self.price_hi,
            "slippage": self.slippage,
            "hour_label": self.hour_label,
            "hours_utc": None if self.hours_utc is None else list(self.hours_utc),
        }


@dataclass
class SplitRows:
    split: str
    n_episodes: int
    ep_idx: np.ndarray
    step_idx: np.ndarray
    date: np.ndarray
    market_slug: np.ndarray
    hour_utc: np.ndarray
    y: np.ndarray
    ttc: np.ndarray
    up_ask: np.ndarray
    dn_ask: np.ndarray
    fill_up_ask: np.ndarray
    fill_dn_ask: np.ndarray
    market_p: np.ndarray
    tcn_logit: np.ndarray
    base_logit: np.ndarray
    delta: np.ndarray
    preds: dict[str, np.ndarray]


def metrics_from_trades(trades: list[dict[str, Any]], n_episodes: int) -> dict[str, Any]:
    pnl = np.asarray([t["pnl"] for t in trades], dtype=np.float64)
    wins = int(sum(1 for t in trades if t["win"]))
    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + float(t["pnl"])
    day_vals = np.asarray([by_day[k] for k in sorted(by_day)], dtype=np.float64)
    equity = np.cumsum(day_vals) if day_vals.size else np.asarray([], dtype=np.float64)
    peak = np.maximum.accumulate(equity) if equity.size else equity
    dd = equity - peak if equity.size else equity
    total_cost = float(sum(t["shares"] * (t["fill"] + float(taker_fee(t["fill"]))) for t in trades))
    return {
        "trades": int(len(trades)),
        "markets": int(n_episodes),
        "trade_rate": float(len(trades) / n_episodes) if n_episodes else 0.0,
        "wins": wins,
        "win_rate": float(wins / len(trades)) if trades else 0.0,
        "total_pnl": float(pnl.sum()) if pnl.size else 0.0,
        "avg_pnl": float(pnl.mean()) if pnl.size else 0.0,
        "median_pnl": float(np.median(pnl)) if pnl.size else 0.0,
        "roi_on_cost": float(pnl.sum() / total_cost) if total_cost > 0 else 0.0,
        "active_days": int(len(by_day)),
        "positive_days": int(sum(1 for v in by_day.values() if v > 0.0)),
        "worst_day": float(day_vals.min()) if day_vals.size else 0.0,
        "best_day": float(day_vals.max()) if day_vals.size else 0.0,
        "max_drawdown": float(dd.min()) if dd.size else 0.0,
        "by_day": dict(sorted(by_day.items())),
    }


def backtest_policy(
    rows: SplitRows,
    p: np.ndarray,
    spec: PolicySpec,
    *,
    shares: float,
    max_trade_sample: int = 25,
) -> dict[str, Any]:
    p = np.clip(p.astype(np.float32, copy=False), 1e-5, 1.0 - 1e-5)
    finite = (
        np.isfinite(p)
        & np.isfinite(rows.up_ask)
        & np.isfinite(rows.dn_ask)
        & np.isfinite(rows.fill_up_ask)
        & np.isfinite(rows.fill_dn_ask)
    )
    eligible = finite & (rows.ttc > spec.ttc_min) & (rows.ttc <= spec.ttc_max)
    if spec.hours_utc is not None:
        eligible &= np.isin(rows.hour_utc, np.asarray(spec.hours_utc, dtype=np.int16))

    ev_up = p - rows.up_ask
    ev_dn = (1.0 - p) - rows.dn_ask
    take_up = (
        eligible
        & (ev_up >= spec.ev)
        & (ev_up >= ev_dn)
        & (rows.up_ask > spec.price_lo)
        & (rows.up_ask < spec.price_hi)
    )
    take_dn = (
        eligible
        & (ev_dn >= spec.ev)
        & (ev_dn > ev_up)
        & (rows.dn_ask > spec.price_lo)
        & (rows.dn_ask < spec.price_hi)
    )
    cand = np.flatnonzero(take_up | take_dn)
    entered = np.zeros(rows.n_episodes, dtype=bool)
    trades: list[dict[str, Any]] = []
    for i in cand:
        ep = int(rows.ep_idx[i])
        if entered[ep]:
            continue
        side_up = bool(take_up[i])
        quote = float(rows.up_ask[i] if side_up else rows.dn_ask[i])
        fill = float(rows.fill_up_ask[i] if side_up else rows.fill_dn_ask[i])
        if not (0.0 < fill < 1.0):
            continue
        if fill > quote + spec.slippage:
            continue
        win = bool(rows.y[i] == 1.0) if side_up else bool(rows.y[i] == 0.0)
        fee = float(taker_fee(fill))
        pnl = shares * ((1.0 if win else 0.0) - fill - fee)
        entered[ep] = True
        trades.append({
            "split": rows.split,
            "market_slug": str(rows.market_slug[i]),
            "date": str(rows.date[i]),
            "ep_idx": ep,
            "step_idx": int(rows.step_idx[i]),
            "side": "UP" if side_up else "DOWN",
            "p": float(p[i]),
            "quote": quote,
            "fill": fill,
            "ev": float(ev_up[i] if side_up else ev_dn[i]),
            "ttc_s": float(rows.ttc[i]),
            "hour_utc": int(rows.hour_utc[i]),
            "win": win,
            "shares": shares,
            "pnl": float(pnl),
        })
    out = metrics_from_trades(trades, rows.n_episodes)
    out["trade_sample"] = trades[:max_trade_sample]
    return out


def _fit_train_platt(train: SplitRows, *, c: float) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=c, solver="lbfgs", max_iter=1000)
    lr.fit(train.tcn_logit.reshape(-1, 1), train.y.astype(np.int8, copy=False))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def _add_lgbm_predictions(rows: SplitRows, dataset: Path, split: str, lgbm_dir: Path) -> None:
    import lightgbm as lgb

    model_path = lgbm_dir / "lgbm_residual_model.txt"
    if not model_path.exists():
        return
    booster = lgb.Booster(model_file=str(model_path))
    z = np.load(dataset / f"{split}.npz", allow_pickle=False)
    x = z["X"][rows.ep_idx, rows.step_idx, :].astype(np.float32, copy=False)
    raw = booster.predict(x, raw_score=True)
    rows.preds["lgbm"] = sigmoid(rows.base_logit + raw.astype(np.float32, copy=False))


def load_split_rows(
    dataset: Path,
    tcn_dir: Path,
    split: str,
    *,
    delay_s: float,
) -> SplitRows:
    pred = np.load(tcn_dir / f"predictions_{split}.npz", allow_pickle=False)
    z = np.load(dataset / f"{split}.npz", allow_pickle=False)
    ep = pred["ep_idx"].astype(np.int32, copy=False)
    step = pred["step_idx"].astype(np.int32, copy=False)
    seq_len = int(z["quotes"].shape[1])
    delay_steps = int(math.ceil(delay_s / 0.2))
    fill_step = step + delay_steps
    ok_fill = fill_step < seq_len
    fill_step_clip = np.minimum(fill_step, seq_len - 1)

    quotes = z["quotes"]
    up_ask = quotes[ep, step, QUOTE_UP_ASK].astype(np.float32, copy=False)
    dn_ask = quotes[ep, step, QUOTE_DN_ASK].astype(np.float32, copy=False)
    fill_up = quotes[ep, fill_step_clip, QUOTE_UP_ASK].astype(np.float32, copy=False)
    fill_dn = quotes[ep, fill_step_clip, QUOTE_DN_ASK].astype(np.float32, copy=False)
    fill_up = np.where(ok_fill, fill_up, np.nan).astype(np.float32, copy=False)
    fill_dn = np.where(ok_fill, fill_dn, np.nan).astype(np.float32, copy=False)
    open_s = z["open_s"][ep].astype(np.float64, copy=False)
    now_s = open_s + step.astype(np.float64) * 0.2
    hour = ((now_s // 3600) % 24).astype(np.int16)

    base = pred["base_logit"].astype(np.float32, copy=False)
    tcn_logit = pred["logit"].astype(np.float32, copy=False)
    delta = pred["delta"].astype(np.float32, copy=False)
    market_p = sigmoid(base)
    rows = SplitRows(
        split=split,
        n_episodes=int(z["y"].shape[0]),
        ep_idx=ep,
        step_idx=step.astype(np.int16, copy=False),
        date=pred["date"].copy(),
        market_slug=pred["market_slug"].copy(),
        hour_utc=hour,
        y=pred["y"].astype(np.float32, copy=False),
        ttc=pred["ttc_s"].astype(np.float32, copy=False),
        up_ask=up_ask,
        dn_ask=dn_ask,
        fill_up_ask=fill_up,
        fill_dn_ask=fill_dn,
        market_p=market_p,
        tcn_logit=tcn_logit,
        base_logit=base,
        delta=delta,
        preds={
            "market": market_p,
            "tcn_raw": sigmoid(tcn_logit),
        },
    )
    return rows


def add_sequence_variants(splits: dict[str, SplitRows]) -> dict[str, Any]:
    train = splits["train"]
    coef, intercept = _fit_train_platt(train, c=0.1)
    for rows in splits.values():
        rows.preds["tcn_train_platt"] = sigmoid(coef * rows.tcn_logit + intercept)
        for beta in (-0.5, -0.25, 0.25, 0.5, 0.75, 1.25):
            rows.preds[f"tcn_resid_beta_{beta:g}"] = sigmoid(rows.base_logit + beta * rows.delta)
        for alpha in (0.25, 0.5, 0.75):
            rows.preds[f"tcn_prob_blend_{alpha:g}"] = (
                (1.0 - alpha) * rows.market_p + alpha * rows.preds["tcn_raw"]
            ).astype(np.float32, copy=False)
    return {"train_platt": {"coef": coef, "intercept": intercept, "C": 0.1}}


def make_policy_specs() -> list[PolicySpec]:
    ttc_windows = [
        (15.0, 90.0),
        (15.0, 75.0),
        (15.0, 60.0),
        (30.0, 75.0),
        (30.0, 90.0),
        (45.0, 90.0),
        (50.0, 75.0),
        (60.0, 90.0),
    ]
    ev_grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
    price_ranges = [(0.10, 0.90), (0.20, 0.80), (0.30, 0.70)]
    slippages = [0.01, 0.03, 0.05]
    hour_filters: list[tuple[str, tuple[int, ...] | None]] = [
        ("all", None),
        ("utc_12_18", tuple(range(12, 18))),
        ("utc_12_20", tuple(range(12, 20))),
    ]
    out = []
    for lo, hi in ttc_windows:
        for ev in ev_grid:
            for px_lo, px_hi in price_ranges:
                for slip in slippages:
                    for h_label, hours in hour_filters:
                        out.append(PolicySpec(lo, hi, ev, px_lo, px_hi, slip, h_label, hours))
    return out


def compact(name: str, spec: PolicySpec, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": name,
        "policy": spec.to_json(),
        "trades": result["trades"],
        "win_rate": result["win_rate"],
        "total_pnl": result["total_pnl"],
        "avg_pnl": result["avg_pnl"],
        "roi_on_cost": result["roi_on_cost"],
        "active_days": result["active_days"],
        "positive_days": result["positive_days"],
        "worst_day": result["worst_day"],
        "max_drawdown": result["max_drawdown"],
    }


def run_grid(
    splits: dict[str, SplitRows],
    specs: list[PolicySpec],
    *,
    shares: float,
    min_trades: int,
) -> dict[str, Any]:
    model_names = sorted(set().union(*(rows.preds.keys() for rows in splits.values())))
    results: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    anchor_spec = PolicySpec(15.0, 75.0, 0.02, 0.10, 0.90, 0.05, "all", None)
    anchor: dict[str, dict[str, Any]] = {split: {} for split in splits}
    for split, rows in splits.items():
        for model in model_names:
            if model not in rows.preds:
                continue
            anchor[split][model] = compact(model, anchor_spec, backtest_policy(rows, rows.preds[model], anchor_spec, shares=shares))

    for split, rows in splits.items():
        print(f"grid split={split} rows={len(rows.y)} models={len(model_names)} specs={len(specs)}", flush=True)
        for model in model_names:
            if model not in rows.preds:
                continue
            p = rows.preds[model]
            for spec in specs:
                r = backtest_policy(rows, p, spec, shares=shares, max_trade_sample=0)
                if r["trades"] < min_trades:
                    continue
                results[split].append(compact(model, spec, r))
        results[split].sort(key=lambda r: (r["total_pnl"], r["avg_pnl"]), reverse=True)
    return {"anchor": anchor, "grid": results}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat = []
    for r in rows:
        p = r["policy"]
        row = {k: v for k, v in r.items() if k != "policy"}
        row.update({
            "ttc_min": p["ttc_min"],
            "ttc_max": p["ttc_max"],
            "ev": p["ev"],
            "price_lo": p["price_lo"],
            "price_hi": p["price_hi"],
            "slippage": p["slippage"],
            "hour_label": p["hour_label"],
        })
        flat.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)


def _choose_selected(report_grid: dict[str, list[dict[str, Any]]], selector_split: str, eval_split: str) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    eval_rows = report_grid[eval_split]
    for model in sorted(set(r["model"] for r in report_grid[selector_split])):
        candidates = [r for r in report_grid[selector_split] if r["model"] == model]
        if not candidates:
            continue
        best = candidates[0]
        key = (
            best["model"],
            best["policy"]["ttc_min"],
            best["policy"]["ttc_max"],
            best["policy"]["ev"],
            best["policy"]["price_lo"],
            best["policy"]["price_hi"],
            best["policy"]["slippage"],
            best["policy"]["hour_label"],
        )
        matched = [
            r for r in eval_rows
            if (
                r["model"],
                r["policy"]["ttc_min"],
                r["policy"]["ttc_max"],
                r["policy"]["ev"],
                r["policy"]["price_lo"],
                r["policy"]["price_hi"],
                r["policy"]["slippage"],
                r["policy"]["hour_label"],
            ) == key
        ]
        if matched:
            row = dict(matched[0])
            row["selected_on"] = selector_split
            row["selector_total_pnl"] = best["total_pnl"]
            row["selector_trades"] = best["trades"]
            chosen.append(row)
    chosen.sort(key=lambda r: r["total_pnl"], reverse=True)
    return chosen


def _write_summary(out_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Sequence Policy Search",
        "",
        f"Dataset: `{report['dataset']}`",
        f"TCN artifacts: `{report['tcn_artifacts']}`",
        f"LGBM artifacts: `{report['lgbm_artifacts']}`",
        f"Delay: `{report['delay_s']}s`, shares: `{report['shares']}`",
        "",
        "## Anchor Policy",
        "",
        "`TTC 15-75, EV>=0.02, price 0.10-0.90, slippage 0.05, all hours`",
        "",
    ]
    for split in ("train", "val", "test"):
        lines.extend([
            f"### {split}",
            "",
            "| model | trades | pnl | avg | winrate | worst day | max DD |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for model, r in sorted(report["anchor"][split].items(), key=lambda kv: kv[1]["total_pnl"], reverse=True):
            lines.append(
                f"| {model} | {r['trades']} | {r['total_pnl']:.2f} | {r['avg_pnl']:.3f} | "
                f"{r['win_rate']:.1%} | {r['worst_day']:.2f} | {r['max_drawdown']:.2f} |"
            )
        lines.append("")

    for title, rows in (
        ("Top Test Policies, Exploratory Oracle", report["top_test"]),
        ("Policies Selected On Val, Evaluated On Test", report["selected_val_to_test"]),
        ("Policies Selected On Train, Evaluated On Test", report["selected_train_to_test"]),
    ):
        lines.extend([
            f"## {title}",
            "",
            "| model | trades | pnl | avg | winrate | ttc | ev | px | slip | hours |",
            "|---|---:|---:|---:|---:|---|---:|---|---:|---|",
        ])
        for r in rows[:20]:
            p = r["policy"]
            lines.append(
                f"| {r['model']} | {r['trades']} | {r['total_pnl']:.2f} | {r['avg_pnl']:.3f} | "
                f"{r['win_rate']:.1%} | {p['ttc_min']:g}-{p['ttc_max']:g} | {p['ev']:.3g} | "
                f"{p['price_lo']:.1f}-{p['price_hi']:.1f} | {p['slippage']:.2f} | {p['hour_label']} |"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--tcn-artifacts", type=Path, default=DEFAULT_TCN)
    ap.add_argument("--lgbm-artifacts", type=Path, default=DEFAULT_LGBM)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--delay-s", type=float, default=2.0)
    ap.add_argument("--shares", type=float, default=5.1)
    ap.add_argument("--min-trades", type=int, default=5)
    args = ap.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    tcn_dir = args.tcn_artifacts if args.tcn_artifacts.is_absolute() else ROOT / args.tcn_artifacts
    lgbm_dir = args.lgbm_artifacts if args.lgbm_artifacts.is_absolute() else ROOT / args.lgbm_artifacts
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    splits = {
        split: load_split_rows(dataset, tcn_dir, split, delay_s=args.delay_s)
        for split in ("train", "val", "test")
    }
    calibration = add_sequence_variants(splits)
    for split, rows in splits.items():
        _add_lgbm_predictions(rows, dataset, split, lgbm_dir)
        print(f"loaded split={split} rows={len(rows.y)} preds={sorted(rows.preds)}", flush=True)

    specs = make_policy_specs()
    grid = run_grid(splits, specs, shares=args.shares, min_trades=args.min_trades)
    report_grid = grid["grid"]
    top_test = report_grid["test"][:100]
    selected_val_to_test = _choose_selected(report_grid, "val", "test")
    selected_train_to_test = _choose_selected(report_grid, "train", "test")

    report = {
        "dataset": str(dataset),
        "tcn_artifacts": str(tcn_dir),
        "lgbm_artifacts": str(lgbm_dir),
        "delay_s": args.delay_s,
        "shares": args.shares,
        "min_trades": args.min_trades,
        "calibration": calibration,
        "elapsed_s": round(time.time() - t0, 2),
        "anchor": grid["anchor"],
        "top_train": report_grid["train"][:100],
        "top_val": report_grid["val"][:100],
        "top_test": top_test,
        "selected_val_to_test": selected_val_to_test,
        "selected_train_to_test": selected_train_to_test,
    }
    (out_dir / "policy_search_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(out_dir / "top_train.csv", report["top_train"])
    _write_csv(out_dir / "top_val.csv", report["top_val"])
    _write_csv(out_dir / "top_test.csv", report["top_test"])
    _write_csv(out_dir / "selected_val_to_test.csv", report["selected_val_to_test"])
    _write_csv(out_dir / "selected_train_to_test.csv", report["selected_train_to_test"])
    _write_summary(out_dir, report)
    print("FINAL", json.dumps({
        "elapsed_s": report["elapsed_s"],
        "anchor_test": report["anchor"]["test"],
        "top_test": report["top_test"][:10],
        "selected_val_to_test": report["selected_val_to_test"][:10],
        "selected_train_to_test": report["selected_train_to_test"][:10],
    }, indent=2), flush=True)
    print(f"report -> {out_dir / 'policy_search_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
