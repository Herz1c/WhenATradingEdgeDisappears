"""Explore TCN entry+exit policies outside the original TTC 15-90 window.

Important caveat: the saved TCN was trained with loss on TTC 15-90 only.  This
script forwards the model over the full valid 0-300s episode grid, so policies
with entries outside 15-90 are exploratory extrapolations, not validated model
use.  The goal is to test whether buy/sell logic farther from resolution is a
promising direction worth retraining for.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_episode_tcn import ResidualTCN, _logit_np  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_TCN = ROOT / "artifacts" / "btc_5m_episode_tcn_c64_b7_cal_ttc15_90"
DEFAULT_OUT = ROOT / "artifacts" / "tcn_fullmarket_exit_policy_search_v1"

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
    beta: float
    ttc_min: float
    ttc_max: float
    ev: float
    price_lo: float
    price_hi: float
    delay_s: float
    slippage: float

    @property
    def label(self) -> str:
        return (
            f"b{self.beta:g}_ttc{self.ttc_min:g}-{self.ttc_max:g}"
            f"_ev{self.ev:g}_px{self.price_lo:g}-{self.price_hi:g}"
            f"_slip{self.slippage:g}"
        )

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()


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
        return self.__dict__.copy()


@dataclass
class SplitData:
    split: str
    quotes: np.ndarray
    valid: np.ndarray
    y: np.ndarray
    date: np.ndarray
    market_slug: np.ndarray
    base_logit: np.ndarray
    delta: np.ndarray
    cadence_s: float

    @property
    def n_ep(self) -> int:
        return int(self.y.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.valid.shape[1])

    @property
    def ttc_grid(self) -> np.ndarray:
        return (self.seq_len * self.cadence_s - np.arange(self.seq_len, dtype=np.float32) * self.cadence_s)


def load_model(tcn_dir: Path, n_features: int, device: torch.device) -> tuple[ResidualTCN, dict[str, Any]]:
    report = json.loads((tcn_dir / "tcn_report.json").read_text(encoding="utf-8"))
    model_cfg = report["model"]
    model = ResidualTCN(
        n_features=n_features,
        channels=int(model_cfg["channels"]),
        blocks=int(model_cfg["blocks"]),
        kernel_size=int(model_cfg["kernel_size"]),
        dropout=0.0,
        residual_scale=float(model_cfg.get("residual_scale", 0.5)),
    ).to(device)
    state = torch.load(tcn_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, report


@torch.no_grad()
def predict_delta(model: torch.nn.Module, x: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    outs: list[np.ndarray] = []
    for i in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        outs.append(model(xb).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outs, axis=0)


def load_split(dataset: Path, model: torch.nn.Module, split: str, *, cadence_s: float,
               batch_size: int, device: torch.device, include_mask_channel: bool) -> SplitData:
    z = np.load(dataset / f"{split}.npz", allow_pickle=False)
    x = z["X"].astype(np.float32, copy=False)
    valid = z["valid_mask"].astype(bool, copy=False)
    if include_mask_channel:
        x_model = np.concatenate([x, valid[:, :, None].astype(np.float32)], axis=2)
    else:
        x_model = x
    p_market = z["p_market"].astype(np.float32, copy=False)
    p_market[~np.isfinite(p_market)] = 0.5
    p_market = np.clip(p_market, 1e-5, 1.0 - 1e-5)
    print(f"predict split={split} shape={x_model.shape}", flush=True)
    delta = predict_delta(model, x_model, batch_size=batch_size, device=device)
    return SplitData(
        split=split,
        quotes=z["quotes"].astype(np.float32, copy=False),
        valid=valid,
        y=z["y"].astype(np.int8, copy=False),
        date=z["date"].copy(),
        market_slug=z["market_slug"].copy(),
        base_logit=_logit_np(p_market),
        delta=delta,
        cadence_s=cadence_s,
    )


def make_entry_specs(*, quick: bool) -> list[EntrySpec]:
    betas = [0.5, 1.0, 1.25] if quick else [0.5, 0.75, 1.0, 1.25]
    windows = [
        (50.0, 75.0),
        (75.0, 120.0),
        (75.0, 150.0),
        (90.0, 150.0),
        (90.0, 180.0),
        (90.0, 240.0),
        (90.0, 300.0),
        (120.0, 240.0),
        (120.0, 300.0),
        (150.0, 300.0),
    ]
    if quick:
        windows = [(90.0, 180.0), (90.0, 240.0), (120.0, 240.0), (120.0, 300.0), (150.0, 300.0)]
        evs = [0.0, 0.075, 0.125]
        prices = [(0.20, 0.80)]
    else:
        evs = [0.0, 0.02, 0.05, 0.075, 0.10, 0.125, 0.15]
        prices = [(0.10, 0.90), (0.20, 0.80), (0.30, 0.70)]
    specs: list[EntrySpec] = []
    for beta in betas:
        for lo, hi in windows:
            for ev in evs:
                for px_lo, px_hi in prices:
                    specs.append(EntrySpec(beta, lo, hi, ev, px_lo, px_hi, 2.0, 0.03))
    return specs


def make_exit_specs(*, quick: bool) -> list[ExitSpec]:
    specs: list[ExitSpec] = [ExitSpec("hold")]
    if quick:
        for ttc in [120.0, 90.0, 75.0, 60.0, 45.0, 30.0]:
            specs.append(ExitSpec(f"time_{ttc:g}_slip_0.03", time_exit_ttc=ttc, slippage=0.03))
        for stop in [0.10, 0.20]:
            specs.append(ExitSpec(f"stop_{stop:g}_slip_0.03", stop_loss=stop, slippage=0.03))
        for take in [0.10, 0.20]:
            specs.append(ExitSpec(f"take_{take:g}_slip_0.03", take_profit=take, slippage=0.03))
        for edge in [0.05]:
            specs.append(ExitSpec(f"selledge_{edge:g}_slip_0.03", sell_edge=edge, slippage=0.03))
        return specs
    for slip in [0.01, 0.03, 0.05]:
        for ttc in [120.0, 90.0, 75.0, 60.0, 45.0, 30.0, 15.0]:
            specs.append(ExitSpec(f"time_{ttc:g}_slip_{slip:g}", time_exit_ttc=ttc, slippage=slip))
        for stop in [0.05, 0.10, 0.15, 0.20]:
            specs.append(ExitSpec(f"stop_{stop:g}_slip_{slip:g}", stop_loss=stop, slippage=slip))
        for take in [0.05, 0.10, 0.15, 0.20]:
            specs.append(ExitSpec(f"take_{take:g}_slip_{slip:g}", take_profit=take, slippage=slip))
        for drop in [0.05, 0.10, 0.15, 0.20]:
            specs.append(ExitSpec(f"modeldrop_{drop:g}_slip_{slip:g}", model_drop=drop, slippage=slip))
        for edge in [0.0, 0.03, 0.05, 0.10]:
            specs.append(ExitSpec(f"selledge_{edge:g}_slip_{slip:g}", sell_edge=edge, slippage=slip))
    return specs


def ttc_s(data: SplitData, step: int) -> float:
    return float(data.seq_len * data.cadence_s - step * data.cadence_s)


def find_entries(data: SplitData, spec: EntrySpec) -> list[dict[str, Any]]:
    p = sigmoid(data.base_logit + spec.beta * data.delta)
    q = data.quotes
    ttc = data.ttc_grid
    band = (ttc > spec.ttc_min) & (ttc <= spec.ttc_max)
    valid = data.valid & band[None, :]
    up_ask = q[:, :, UP_ASK]
    dn_ask = q[:, :, DN_ASK]
    ev_up = p - up_ask
    ev_dn = (1.0 - p) - dn_ask
    take_up = (
        valid
        & (ev_up >= spec.ev)
        & (ev_up >= ev_dn)
        & (up_ask > spec.price_lo)
        & (up_ask < spec.price_hi)
    )
    take_dn = (
        valid
        & (ev_dn >= spec.ev)
        & (ev_dn > ev_up)
        & (dn_ask > spec.price_lo)
        & (dn_ask < spec.price_hi)
    )
    cand_ep, cand_step = np.nonzero(take_up | take_dn)
    delay_steps = int(math.ceil(spec.delay_s / data.cadence_s))
    entries: list[dict[str, Any]] = []
    entered = np.zeros(data.n_ep, dtype=bool)
    for ep, step in zip(cand_ep.tolist(), cand_step.tolist()):
        if entered[ep]:
            continue
        fill_step = step + delay_steps
        if fill_step >= data.seq_len:
            continue
        side_up = bool(take_up[ep, step])
        quote = float(up_ask[ep, step] if side_up else dn_ask[ep, step])
        fill = float(q[ep, fill_step, UP_ASK] if side_up else q[ep, fill_step, DN_ASK])
        if not (np.isfinite(fill) and 0.0 < fill < 1.0):
            continue
        if fill > quote + spec.slippage:
            continue
        p_up = float(p[ep, step])
        entered[ep] = True
        entries.append({
            "ep": ep,
            "step": step,
            "fill_step": fill_step,
            "side_up": side_up,
            "side": "UP" if side_up else "DOWN",
            "quote": quote,
            "fill": fill,
            "p_up": p_up,
            "p_side": p_up if side_up else 1.0 - p_up,
            "ev": float(ev_up[ep, step] if side_up else ev_dn[ep, step]),
            "ttc_s": ttc_s(data, step),
        })
    return entries


def reason_to_exit(spec: ExitSpec, *, side_up: bool, entry_fill: float, entry_p_side: float,
                   p_up: float, bid: float, ttc: float) -> str | None:
    p_side = p_up if side_up else 1.0 - p_up
    if spec.stop_loss is not None and bid <= entry_fill - spec.stop_loss:
        return "stop_loss"
    if spec.take_profit is not None and bid >= entry_fill + spec.take_profit:
        return "take_profit"
    if spec.model_drop is not None and np.isfinite(p_side) and p_side <= entry_p_side - spec.model_drop:
        return "model_drop"
    if spec.sell_edge is not None and np.isfinite(p_side) and bid - p_side >= spec.sell_edge:
        return "sell_edge"
    if spec.time_exit_ttc is not None and ttc <= spec.time_exit_ttc:
        return "time_exit"
    return None


def simulate_one(data: SplitData, entry: dict[str, Any], exit_spec: ExitSpec, entry_spec: EntrySpec,
                 *, shares: float) -> dict[str, Any]:
    ep = int(entry["ep"])
    side_up = bool(entry["side_up"])
    q = data.quotes[ep]
    p = sigmoid(data.base_logit[ep] + entry_spec.beta * data.delta[ep])
    delay_steps = int(math.ceil(exit_spec.delay_s / data.cadence_s))
    exit_info: dict[str, Any] | None = None
    for step in range(int(entry["fill_step"]) + 1, data.seq_len - delay_steps):
        ttc = ttc_s(data, step)
        if ttc < exit_spec.min_exit_ttc:
            break
        bid = float(q[step, UP_BID] if side_up else q[step, DN_BID])
        if not (np.isfinite(bid) and 0.0 < bid < 1.0):
            continue
        p_up = float(p[step]) if data.valid[ep, step] else float("nan")
        reason = reason_to_exit(
            exit_spec,
            side_up=side_up,
            entry_fill=float(entry["fill"]),
            entry_p_side=float(entry["p_side"]),
            p_up=p_up,
            bid=bid,
            ttc=ttc,
        )
        if reason is None:
            continue
        fill_step = step + delay_steps
        sell_fill = float(q[fill_step, UP_BID] if side_up else q[fill_step, DN_BID])
        if not (np.isfinite(sell_fill) and 0.0 < sell_fill < 1.0):
            continue
        if sell_fill < bid - exit_spec.slippage:
            continue
        pnl = shares * (sell_fill - float(entry["fill"]) - fee(float(entry["fill"])) - fee(sell_fill))
        exit_info = {
            "exit_type": "sell",
            "exit_reason": reason,
            "exit_step": step,
            "exit_ttc_s": ttc,
            "exit_bid": bid,
            "exit_fill": sell_fill,
            "pnl": float(pnl),
        }
        break
    if exit_info is None:
        win = bool(data.y[ep] == 1) if side_up else bool(data.y[ep] == 0)
        pnl = shares * ((1.0 if win else 0.0) - float(entry["fill"]) - fee(float(entry["fill"])))
        exit_info = {
            "exit_type": "resolve",
            "exit_reason": "hold_to_resolution",
            "exit_step": None,
            "exit_ttc_s": 0.0,
            "exit_bid": None,
            "exit_fill": None,
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
        "entry_ttc_s": float(entry["ttc_s"]),
        "entry_quote": float(entry["quote"]),
        "entry_fill": float(entry["fill"]),
        "entry_ev": float(entry["ev"]),
        "p_up_entry": float(entry["p_up"]),
        "p_side_entry": float(entry["p_side"]),
        **exit_info,
    }


def summarize(trades: list[dict[str, Any]], markets: int) -> dict[str, Any]:
    pnl = np.asarray([t["pnl"] for t in trades], dtype=np.float64)
    by_day: dict[str, float] = {}
    exits: dict[str, int] = {}
    wins = 0
    for t in trades:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + float(t["pnl"])
        exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1
        wins += int(bool(t["resolved_win"]))
    day_vals = np.asarray([by_day[k] for k in sorted(by_day)], dtype=np.float64)
    equity = np.cumsum(day_vals) if day_vals.size else day_vals
    dd = equity - np.maximum.accumulate(equity) if equity.size else equity
    return {
        "trades": int(len(trades)),
        "markets": int(markets),
        "trade_rate": float(len(trades) / markets) if markets else 0.0,
        "wins_to_resolution": int(wins),
        "resolution_win_rate": float(wins / len(trades)) if trades else 0.0,
        "total_pnl": float(pnl.sum()) if pnl.size else 0.0,
        "avg_pnl": float(pnl.mean()) if pnl.size else 0.0,
        "median_pnl": float(np.median(pnl)) if pnl.size else 0.0,
        "active_days": int(len(by_day)),
        "positive_days": int(sum(1 for v in by_day.values() if v > 0.0)),
        "worst_day": float(day_vals.min()) if day_vals.size else 0.0,
        "best_day": float(day_vals.max()) if day_vals.size else 0.0,
        "max_drawdown": float(dd.min()) if dd.size else 0.0,
        "exit_counts": dict(sorted(exits.items())),
        "by_day": dict(sorted(by_day.items())),
    }


def score_result(r: dict[str, Any]) -> float:
    if r["trades"] < 25 or r["total_pnl"] <= 0:
        return -1e9
    return float(r["total_pnl"] / max(1.0, abs(r["max_drawdown"])))


def compact(split: str, entry: EntrySpec, exit_spec: ExitSpec, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split,
        "entry": entry.to_json(),
        "exit": exit_spec.to_json(),
        "score_pnl_over_dd": score_result(result),
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


def run_grid(data: SplitData, entry_specs: list[EntrySpec], exit_specs: list[ExitSpec], *, shares: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, entry_spec in enumerate(entry_specs, 1):
        entries = find_entries(data, entry_spec)
        if idx % (10 if len(entry_specs) <= 100 else 50) == 0:
            print(f"  {data.split}: entry {idx}/{len(entry_specs)} {entry_spec.label} entries={len(entries)}", flush=True)
        if len(entries) < 5:
            continue
        for exit_spec in exit_specs:
            trades = [simulate_one(data, e, exit_spec, entry_spec, shares=shares) for e in entries]
            result = summarize(trades, data.n_ep)
            rows.append(compact(data.split, entry_spec, exit_spec, result))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat = []
    for r in rows:
        row = {k: v for k, v in r.items() if k not in ("entry", "exit", "exit_counts")}
        row.update({f"entry_{k}": v for k, v in r["entry"].items()})
        row.update({f"exit_{k}": v for k, v in r["exit"].items()})
        row["exit_counts_json"] = json.dumps(r["exit_counts"], sort_keys=True)
        flat.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)


def select_val_to_test(val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    val_sorted = [
        r for r in val_rows
        if r["trades"] >= 5 and r["total_pnl"] > 0 and r["score_pnl_over_dd"] > -1e8
    ]
    val_sorted.sort(key=lambda r: (r["score_pnl_over_dd"], r["total_pnl"]), reverse=True)
    out = []
    for r in val_sorted[:25]:
        key = (
            r["entry"]["beta"], r["entry"]["ttc_min"], r["entry"]["ttc_max"], r["entry"]["ev"],
            r["entry"]["price_lo"], r["entry"]["price_hi"], r["exit"]["label"],
        )
        matches = [
            x for x in test_rows
            if (
                x["entry"]["beta"], x["entry"]["ttc_min"], x["entry"]["ttc_max"], x["entry"]["ev"],
                x["entry"]["price_lo"], x["entry"]["price_hi"], x["exit"]["label"],
            ) == key
        ]
        if matches:
            m = dict(matches[0])
            m["selected_on_val_score"] = r["score_pnl_over_dd"]
            m["selected_on_val_pnl"] = r["total_pnl"]
            m["selected_on_val_trades"] = r["trades"]
            out.append(m)
    out.sort(key=lambda r: (r["score_pnl_over_dd"], r["total_pnl"]), reverse=True)
    return out


def write_summary(out_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# TCN Full-Market Active Exit Search",
        "",
        "Caveat: TCN was trained on TTC 15-90. Entries outside that window are exploratory extrapolation.",
        "",
    ]
    for title, rows in [
        ("Top Test By PnL/DD", report["top_test_by_score"][:30]),
        ("Top Test By PnL", report["top_test_by_pnl"][:30]),
        ("Val-Selected Evaluated On Test", report["selected_val_to_test"][:30]),
    ]:
        lines.extend([
            f"## {title}",
            "",
            "| beta | ttc | ev | px | exit | trades | pnl | avg | med | pos days | worst | DD | score | exits |",
            "|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for r in rows:
            e = r["entry"]
            x = r["exit"]
            pos = f"{r['positive_days']}/{r['active_days']}"
            lines.append(
                f"| {e['beta']:g} | {e['ttc_min']:g}-{e['ttc_max']:g} | {e['ev']:.3g} | "
                f"{e['price_lo']:.1f}-{e['price_hi']:.1f} | {x['label']} | {r['trades']} | "
                f"{r['total_pnl']:.2f} | {r['avg_pnl']:.3f} | {r['median_pnl']:.3f} | {pos} | "
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
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--shares", type=float, default=5.1)
    ap.add_argument("--cadence-s", type=float, default=0.2)
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--quick", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--splits", default="train,val,test",
                    help="comma-separated splits to run; use val,test for faster exploratory searches")
    args = ap.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    tcn_dir = args.tcn_artifacts if args.tcn_artifacts.is_absolute() else ROOT / args.tcn_artifacts
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset has 69 features, plus valid mask channel used during training.
    model, model_report = load_model(tcn_dir, n_features=70, device=device)
    t0 = time.time()
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    data_by_split = {
        split: load_split(dataset, model, split, cadence_s=args.cadence_s, batch_size=args.batch_size,
                          device=device, include_mask_channel=True)
        for split in split_names
    }
    entry_specs = make_entry_specs(quick=args.quick)
    exit_specs = make_exit_specs(quick=args.quick)
    rows_by_split = {}
    for split, data in data_by_split.items():
        print(f"grid split={split} entries={len(entry_specs)} exits={len(exit_specs)}", flush=True)
        rows = run_grid(data, entry_specs, exit_specs, shares=args.shares)
        rows.sort(key=lambda r: (r["score_pnl_over_dd"], r["total_pnl"]), reverse=True)
        rows_by_split[split] = rows
        write_csv(out_dir / f"grid_{split}.csv", rows)
    top_test_by_score = rows_by_split.get("test", [])[:200]
    top_test_by_pnl = sorted(rows_by_split.get("test", []), key=lambda r: r["total_pnl"], reverse=True)[:200]
    selected_val_to_test = (
        select_val_to_test(rows_by_split["val"], rows_by_split["test"])
        if "val" in rows_by_split and "test" in rows_by_split else []
    )
    cfg = model_report.get("config", {})
    trained_ttc = f"{cfg.get('ttc_min', 15.0)}-{cfg.get('ttc_max', 90.0)}" if isinstance(cfg, dict) else "15.0-90.0"
    report = {
        "dataset": str(dataset),
        "tcn_artifacts": str(tcn_dir),
        "model_trained_ttc": trained_ttc,
        "caveat": "TCN was trained on TTC 15-90. Entries outside that window are exploratory extrapolation.",
        "elapsed_s": round(time.time() - t0, 2),
        "entry_specs": len(entry_specs),
        "exit_specs": len(exit_specs),
        "quick": args.quick,
        "splits": split_names,
        "top_test_by_score": top_test_by_score,
        "top_test_by_pnl": top_test_by_pnl,
        "selected_val_to_test": selected_val_to_test,
    }
    (out_dir / "fullmarket_exit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_dir / "top_test_by_score.csv", top_test_by_score)
    write_csv(out_dir / "top_test_by_pnl.csv", top_test_by_pnl)
    write_csv(out_dir / "selected_val_to_test.csv", selected_val_to_test)
    write_summary(out_dir, report)
    print("FINAL", json.dumps({
        "elapsed_s": report["elapsed_s"],
        "top_test_by_score": top_test_by_score[:10],
        "top_test_by_pnl": top_test_by_pnl[:10],
        "selected_val_to_test": selected_val_to_test[:10],
    }, indent=2), flush=True)
    print(f"report -> {out_dir / 'fullmarket_exit_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
