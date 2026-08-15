"""Search active-exit policies for the best TCN entry strategy.

The entry policy is the promising TCN residual strategy found earlier:
    p_strategy = sigmoid(base_logit + beta * delta)
    TTC 50-75, EV >= 0.125, price 0.20-0.80, buy delay 2s, buy slippage 0.03.

Exit policies can still hold to resolution, but may sell on the delayed bid
after stop-loss, take-profit, model deterioration, model overpricing, or a time
exit.  This script is exploratory; use val-selected rows for less biased reads.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_TCN = ROOT / "artifacts" / "btc_5m_episode_tcn_c64_b7_cal_ttc15_90"
DEFAULT_OUT = ROOT / "artifacts" / "tcn_exit_policy_search_v1"

UP_BID = 0
UP_ASK = 1
DN_BID = 2
DN_ASK = 3


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64, copy=False), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def fee(price: float) -> float:
    return float(0.072 * price * (1.0 - price))


@dataclass(frozen=True)
class EntrySpec:
    beta: float = 1.25
    ttc_min: float = 50.0
    ttc_max: float = 75.0
    ev: float = 0.125
    price_lo: float = 0.20
    price_hi: float = 0.80
    delay_s: float = 2.0
    slippage: float = 0.03


@dataclass(frozen=True)
class ExitSpec:
    label: str
    stop_loss: float | None = None
    take_profit: float | None = None
    model_drop: float | None = None
    sell_edge: float | None = None
    time_exit_ttc: float | None = None
    min_exit_ttc: float = 15.0
    delay_s: float = 2.0
    slippage: float = 0.03

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "model_drop": self.model_drop,
            "sell_edge": self.sell_edge,
            "time_exit_ttc": self.time_exit_ttc,
            "min_exit_ttc": self.min_exit_ttc,
            "delay_s": self.delay_s,
            "slippage": self.slippage,
        }


@dataclass
class SplitData:
    split: str
    quotes: np.ndarray
    valid: np.ndarray
    y: np.ndarray
    date: np.ndarray
    market_slug: np.ndarray
    p_strategy: np.ndarray
    pred_valid: np.ndarray

    @property
    def n_ep(self) -> int:
        return int(self.y.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.quotes.shape[1])


def load_split(dataset: Path, tcn_dir: Path, split: str, beta: float) -> SplitData:
    z = np.load(dataset / f"{split}.npz", allow_pickle=False)
    pred = np.load(tcn_dir / f"predictions_{split}.npz", allow_pickle=False)
    n, seq_len = z["valid_mask"].shape
    p = np.full((n, seq_len), np.nan, dtype=np.float32)
    pred_valid = np.zeros((n, seq_len), dtype=bool)
    ep = pred["ep_idx"].astype(np.int32, copy=False)
    step = pred["step_idx"].astype(np.int32, copy=False)
    p_vals = sigmoid(pred["base_logit"].astype(np.float32) + beta * pred["delta"].astype(np.float32))
    p[ep, step] = p_vals
    pred_valid[ep, step] = True
    return SplitData(
        split=split,
        quotes=z["quotes"].astype(np.float32, copy=False),
        valid=z["valid_mask"].astype(bool, copy=False),
        y=z["y"].astype(np.int8, copy=False),
        date=z["date"].copy(),
        market_slug=z["market_slug"].copy(),
        p_strategy=p,
        pred_valid=pred_valid,
    )


def ttc_s(seq_len: int, step: int, cadence_s: float) -> float:
    return float(seq_len * cadence_s - step * cadence_s)


def first_entry(data: SplitData, ep: int, spec: EntrySpec, cadence_s: float) -> dict[str, Any] | None:
    q = data.quotes[ep]
    p = data.p_strategy[ep]
    valid = data.valid[ep] & data.pred_valid[ep]
    buy_delay_steps = int(math.ceil(spec.delay_s / cadence_s))
    for step in np.flatnonzero(valid):
        ttc = ttc_s(data.seq_len, int(step), cadence_s)
        if not (spec.ttc_min < ttc <= spec.ttc_max):
            continue
        p_up = float(p[step])
        up_ask = float(q[step, UP_ASK])
        dn_ask = float(q[step, DN_ASK])
        if not (np.isfinite(p_up) and np.isfinite(up_ask) and np.isfinite(dn_ask)):
            continue
        ev_up = p_up - up_ask
        ev_dn = (1.0 - p_up) - dn_ask
        take_up = (
            ev_up >= spec.ev
            and ev_up >= ev_dn
            and spec.price_lo < up_ask < spec.price_hi
        )
        take_dn = (
            ev_dn >= spec.ev
            and ev_dn > ev_up
            and spec.price_lo < dn_ask < spec.price_hi
        )
        if not (take_up or take_dn):
            continue
        fill_step = int(step) + buy_delay_steps
        if fill_step >= data.seq_len:
            continue
        side = "UP" if take_up else "DOWN"
        quote = up_ask if take_up else dn_ask
        fill = float(q[fill_step, UP_ASK] if take_up else q[fill_step, DN_ASK])
        if not (np.isfinite(fill) and 0.0 < fill < 1.0):
            continue
        if fill > quote + spec.slippage:
            continue
        p_side_entry = p_up if take_up else (1.0 - p_up)
        return {
            "ep": ep,
            "step": int(step),
            "fill_step": fill_step,
            "side": side,
            "side_up": bool(take_up),
            "p_up": p_up,
            "p_side_entry": float(p_side_entry),
            "quote": float(quote),
            "fill": float(fill),
            "ev": float(ev_up if take_up else ev_dn),
            "ttc_s": ttc,
        }
    return None


def exit_reason(
    *,
    exit_spec: ExitSpec,
    side_up: bool,
    entry_fill: float,
    p_side_entry: float,
    p_up: float,
    bid: float,
    ttc: float,
) -> str | None:
    p_side = p_up if side_up else (1.0 - p_up)
    if exit_spec.stop_loss is not None and bid <= entry_fill - exit_spec.stop_loss:
        return "stop_loss"
    if exit_spec.take_profit is not None and bid >= entry_fill + exit_spec.take_profit:
        return "take_profit"
    if exit_spec.model_drop is not None and np.isfinite(p_side) and p_side <= p_side_entry - exit_spec.model_drop:
        return "model_drop"
    if exit_spec.sell_edge is not None and np.isfinite(p_side) and bid - p_side >= exit_spec.sell_edge:
        return "sell_edge"
    if exit_spec.time_exit_ttc is not None and ttc <= exit_spec.time_exit_ttc:
        return "time_exit"
    return None


def simulate_trade(
    data: SplitData,
    entry: dict[str, Any],
    exit_spec: ExitSpec,
    *,
    cadence_s: float,
    shares: float,
) -> dict[str, Any]:
    ep = int(entry["ep"])
    q = data.quotes[ep]
    p = data.p_strategy[ep]
    side_up = bool(entry["side_up"])
    exit_delay_steps = int(math.ceil(exit_spec.delay_s / cadence_s))
    start = int(entry["fill_step"]) + 1
    last_decision = data.seq_len - exit_delay_steps - 1
    exit_trade: dict[str, Any] | None = None
    for step in range(start, max(start, last_decision + 1)):
        ttc = ttc_s(data.seq_len, step, cadence_s)
        if ttc < exit_spec.min_exit_ttc:
            break
        bid = float(q[step, UP_BID] if side_up else q[step, DN_BID])
        p_up = float(p[step]) if data.pred_valid[ep, step] else float("nan")
        if not (np.isfinite(bid) and 0.0 < bid < 1.0):
            continue
        reason = exit_reason(
            exit_spec=exit_spec,
            side_up=side_up,
            entry_fill=float(entry["fill"]),
            p_side_entry=float(entry["p_side_entry"]),
            p_up=p_up,
            bid=bid,
            ttc=ttc,
        )
        if reason is None:
            continue
        fill_step = step + exit_delay_steps
        sell_fill = float(q[fill_step, UP_BID] if side_up else q[fill_step, DN_BID])
        if not (np.isfinite(sell_fill) and 0.0 < sell_fill < 1.0):
            continue
        if sell_fill < bid - exit_spec.slippage:
            continue
        pnl = shares * (sell_fill - float(entry["fill"]) - fee(float(entry["fill"])) - fee(sell_fill))
        exit_trade = {
            "exit_type": "sell",
            "exit_reason": reason,
            "exit_step": step,
            "exit_fill_step": fill_step,
            "exit_ttc_s": ttc,
            "exit_decision_bid": bid,
            "exit_fill": sell_fill,
            "p_up_exit": p_up,
            "pnl": float(pnl),
        }
        break
    if exit_trade is None:
        win = bool(data.y[ep] == 1) if side_up else bool(data.y[ep] == 0)
        pnl = shares * ((1.0 if win else 0.0) - float(entry["fill"]) - fee(float(entry["fill"])))
        exit_trade = {
            "exit_type": "resolve",
            "exit_reason": "hold_to_resolution",
            "exit_step": None,
            "exit_fill_step": None,
            "exit_ttc_s": 0.0,
            "exit_decision_bid": None,
            "exit_fill": None,
            "p_up_exit": None,
            "pnl": float(pnl),
        }
    win = bool(data.y[ep] == 1) if side_up else bool(data.y[ep] == 0)
    return {
        "split": data.split,
        "date": str(data.date[ep]),
        "market_slug": str(data.market_slug[ep]),
        "ep_idx": ep,
        "side": entry["side"],
        "resolved_win": win,
        "entry_step": int(entry["step"]),
        "entry_fill_step": int(entry["fill_step"]),
        "entry_ttc_s": float(entry["ttc_s"]),
        "p_up_entry": float(entry["p_up"]),
        "p_side_entry": float(entry["p_side_entry"]),
        "entry_quote": float(entry["quote"]),
        "entry_fill": float(entry["fill"]),
        "entry_ev": float(entry["ev"]),
        **exit_trade,
    }


def summarize(trades: list[dict[str, Any]], markets: int) -> dict[str, Any]:
    pnl = np.asarray([t["pnl"] for t in trades], dtype=np.float64)
    by_day: dict[str, float] = {}
    exit_counts: dict[str, int] = {}
    wins = 0
    for t in trades:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + float(t["pnl"])
        exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1
        wins += int(bool(t["resolved_win"]))
    day_vals = np.asarray([by_day[k] for k in sorted(by_day)], dtype=np.float64)
    equity = np.cumsum(day_vals) if day_vals.size else day_vals
    dd = equity - np.maximum.accumulate(equity) if equity.size else equity
    return {
        "trades": int(len(trades)),
        "markets": int(markets),
        "trade_rate": float(len(trades) / markets) if markets else 0.0,
        "wins_to_resolution": wins,
        "resolution_win_rate": float(wins / len(trades)) if trades else 0.0,
        "total_pnl": float(pnl.sum()) if pnl.size else 0.0,
        "avg_pnl": float(pnl.mean()) if pnl.size else 0.0,
        "median_pnl": float(np.median(pnl)) if pnl.size else 0.0,
        "active_days": int(len(by_day)),
        "positive_days": int(sum(1 for v in by_day.values() if v > 0.0)),
        "worst_day": float(day_vals.min()) if day_vals.size else 0.0,
        "best_day": float(day_vals.max()) if day_vals.size else 0.0,
        "max_drawdown": float(dd.min()) if dd.size else 0.0,
        "exit_counts": dict(sorted(exit_counts.items())),
        "by_day": dict(sorted(by_day.items())),
    }


def run_policy(data: SplitData, entry_spec: EntrySpec, exit_spec: ExitSpec, *, cadence_s: float, shares: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    for ep in range(data.n_ep):
        entry = first_entry(data, ep, entry_spec, cadence_s)
        if entry is None:
            continue
        trades.append(simulate_trade(data, entry, exit_spec, cadence_s=cadence_s, shares=shares))
    return summarize(trades, data.n_ep), trades


def make_exit_specs() -> list[ExitSpec]:
    specs: list[ExitSpec] = [ExitSpec("hold")]
    slips = [0.01, 0.03, 0.05]
    for slip in slips:
        for ttc in [45.0, 30.0, 15.0]:
            specs.append(ExitSpec(f"time_{ttc:g}_slip_{slip:g}", time_exit_ttc=ttc, slippage=slip))
        for stop in [0.05, 0.10, 0.15, 0.20]:
            specs.append(ExitSpec(f"stop_{stop:g}_slip_{slip:g}", stop_loss=stop, slippage=slip))
        for tp in [0.05, 0.10, 0.15, 0.20]:
            specs.append(ExitSpec(f"take_{tp:g}_slip_{slip:g}", take_profit=tp, slippage=slip))
        for stop in [0.05, 0.10, 0.15, 0.20]:
            for tp in [0.05, 0.10, 0.15, 0.20]:
                specs.append(ExitSpec(f"stop_{stop:g}_take_{tp:g}_slip_{slip:g}", stop_loss=stop, take_profit=tp, slippage=slip))
        for drop in [0.05, 0.10, 0.15, 0.20]:
            specs.append(ExitSpec(f"modeldrop_{drop:g}_slip_{slip:g}", model_drop=drop, slippage=slip))
            specs.append(ExitSpec(f"modeldrop_{drop:g}_stop_0.10_take_0.10_slip_{slip:g}", model_drop=drop, stop_loss=0.10, take_profit=0.10, slippage=slip))
        for edge in [0.0, 0.03, 0.05]:
            specs.append(ExitSpec(f"selledge_{edge:g}_slip_{slip:g}", sell_edge=edge, slippage=slip))
            specs.append(ExitSpec(f"selledge_{edge:g}_stop_0.10_take_0.10_slip_{slip:g}", sell_edge=edge, stop_loss=0.10, take_profit=0.10, slippage=slip))
    return specs


def compact(split: str, spec: ExitSpec, result: dict[str, Any]) -> dict[str, Any]:
    score = result["total_pnl"] / max(1.0, abs(result["max_drawdown"]))
    return {
        "split": split,
        "exit": spec.to_json(),
        "score_pnl_over_dd": float(score),
        "trades": result["trades"],
        "total_pnl": result["total_pnl"],
        "avg_pnl": result["avg_pnl"],
        "median_pnl": result["median_pnl"],
        "resolution_win_rate": result["resolution_win_rate"],
        "active_days": result["active_days"],
        "positive_days": result["positive_days"],
        "worst_day": result["worst_day"],
        "best_day": result["best_day"],
        "max_drawdown": result["max_drawdown"],
        "exit_counts": result["exit_counts"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat = []
    for r in rows:
        row = {k: v for k, v in r.items() if k not in ("exit", "exit_counts")}
        row.update({f"exit_{k}": v for k, v in r["exit"].items()})
        row["exit_counts_json"] = json.dumps(r["exit_counts"], sort_keys=True)
        flat.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)


def write_trades_csv(path: Path, trades: list[dict[str, Any]]) -> None:
    if not trades:
        return
    keys = sorted(set().union(*(t.keys() for t in trades)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(trades)


def choose_by_val(rows_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    val = rows_by_split["val"]
    test = rows_by_split["test"]
    chosen = []
    # Risk-aware objective, not raw PnL.
    filtered = [
        r for r in val
        if r["trades"] >= 5 and r["total_pnl"] > 0 and r["max_drawdown"] >= -10
    ]
    if not filtered:
        filtered = val
    filtered.sort(key=lambda r: (r["score_pnl_over_dd"], r["total_pnl"]), reverse=True)
    for best in filtered[:10]:
        label = best["exit"]["label"]
        matches = [r for r in test if r["exit"]["label"] == label]
        if matches:
            out = dict(matches[0])
            out["selected_on_val_score"] = best["score_pnl_over_dd"]
            out["selected_on_val_pnl"] = best["total_pnl"]
            out["selected_on_val_trades"] = best["trades"]
            chosen.append(out)
    chosen.sort(key=lambda r: (r["score_pnl_over_dd"], r["total_pnl"]), reverse=True)
    return chosen


def write_summary(out_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# TCN Active Exit Policy Search",
        "",
        f"Dataset: `{report['dataset']}`",
        f"TCN artifacts: `{report['tcn_artifacts']}`",
        "",
        "Entry policy:",
        "",
        "```json",
        json.dumps(report["entry_spec"], indent=2),
        "```",
        "",
    ]
    for title, rows in [
        ("Hold Baseline", report["hold_baseline"]),
        ("Top Test By PnL/DD", report["top_test_by_score"][:20]),
        ("Top Test By PnL", report["top_test_by_pnl"][:20]),
        ("Val-Selected Exit Policies Evaluated On Test", report["selected_val_to_test"]),
    ]:
        lines.extend([
            f"## {title}",
            "",
            "| exit | trades | pnl | avg | med | win-to-res | pos days | worst | DD | score | exits |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        iterable = rows.values() if isinstance(rows, dict) else rows
        for r in iterable:
            pos = f"{r['positive_days']}/{r['active_days']}"
            lines.append(
                f"| {r['exit']['label']} | {r['trades']} | {r['total_pnl']:.2f} | {r['avg_pnl']:.3f} | "
                f"{r['median_pnl']:.3f} | {r['resolution_win_rate']:.1%} | {pos} | "
                f"{r['worst_day']:.2f} | {r['max_drawdown']:.2f} | {r['score_pnl_over_dd']:.2f} | "
                f"`{json.dumps(r['exit_counts'], sort_keys=True)}` |"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--tcn-artifacts", type=Path, default=DEFAULT_TCN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--beta", type=float, default=1.25)
    ap.add_argument("--shares", type=float, default=5.1)
    ap.add_argument("--cadence-s", type=float, default=0.2)
    args = ap.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    tcn_dir = args.tcn_artifacts if args.tcn_artifacts.is_absolute() else ROOT / args.tcn_artifacts
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    entry_spec = EntrySpec(beta=args.beta)
    data_by_split = {
        split: load_split(dataset, tcn_dir, split, args.beta)
        for split in ("train", "val", "test")
    }
    exit_specs = make_exit_specs()
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    hold_baseline: dict[str, dict[str, Any]] = {}
    best_trades: list[dict[str, Any]] = []
    for split, data in data_by_split.items():
        print(f"split={split} episodes={data.n_ep} exit_specs={len(exit_specs)}", flush=True)
        rows: list[dict[str, Any]] = []
        for spec in exit_specs:
            result, trades = run_policy(data, entry_spec, spec, cadence_s=args.cadence_s, shares=args.shares)
            row = compact(split, spec, result)
            rows.append(row)
            if spec.label == "hold":
                hold_baseline[split] = row
        rows_by_split[split] = rows
        rows.sort(key=lambda r: (r["score_pnl_over_dd"], r["total_pnl"]), reverse=True)

    top_test_by_score = sorted(rows_by_split["test"], key=lambda r: (r["score_pnl_over_dd"], r["total_pnl"]), reverse=True)
    top_test_by_pnl = sorted(rows_by_split["test"], key=lambda r: r["total_pnl"], reverse=True)
    selected_val_to_test = choose_by_val(rows_by_split)
    if selected_val_to_test:
        label = selected_val_to_test[0]["exit"]["label"]
        spec = next(s for s in exit_specs if s.label == label)
        _, best_trades = run_policy(data_by_split["test"], entry_spec, spec, cadence_s=args.cadence_s, shares=args.shares)

    report = {
        "dataset": str(dataset),
        "tcn_artifacts": str(tcn_dir),
        "entry_spec": entry_spec.__dict__,
        "shares": args.shares,
        "hold_baseline": hold_baseline,
        "top_test_by_score": top_test_by_score[:100],
        "top_test_by_pnl": top_test_by_pnl[:100],
        "selected_val_to_test": selected_val_to_test,
    }
    (out_dir / "exit_policy_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_dir / "stress_grid_test_by_score.csv", top_test_by_score)
    write_csv(out_dir / "stress_grid_test_by_pnl.csv", top_test_by_pnl)
    write_csv(out_dir / "selected_val_to_test.csv", selected_val_to_test)
    write_trades_csv(out_dir / "selected_val_to_test_best_trades.csv", best_trades)
    write_summary(out_dir, report)
    print("FINAL", json.dumps({
        "hold_test": hold_baseline["test"],
        "top_test_by_score": top_test_by_score[:10],
        "top_test_by_pnl": top_test_by_pnl[:10],
        "selected_val_to_test": selected_val_to_test[:10],
    }, indent=2), flush=True)
    print(f"report -> {out_dir / 'exit_policy_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
