"""Backtest a locked strategy JSON through the canonical shared engine.

Reproduces (or refutes) the canonical metrics recorded inside a strategy lock
from committed code only: lock JSON + dataset npz + model.pt. Writes a full
per-trade CSV, per-day PnL, and a summary JSON, and prints a side-by-side
comparison against the lock's own canonical_clean_rerun_metrics.

Usage:
    py tools/backtest_locked_strategy.py                       # lock v1, test split
    py tools/backtest_locked_strategy.py --split val
    py tools/backtest_locked_strategy.py --fill-mode fill_worse --delay-s 3
    py tools/backtest_locked_strategy.py --out artifacts/backtest_repro_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from backtest.episode_strategy_backtester import (  # noqa: E402
    HOLD, load_episode_split, load_lock, run_locked_strategies,
)

DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_TCN = ROOT / "artifacts" / "btc_5m_episode_tcn_c64_b7_cal_ttc15_90"
DEFAULT_LOCK = ROOT / "artifacts" / "strategy_locks" / "tcn_double_strategy_v1_lock.json"
DEFAULT_OUT = ROOT / "artifacts" / "backtest_repro_v1"

TRADE_COLS = [
    "strategy_id", "date", "market_slug", "side", "entry_step", "entry_ttc_s",
    "entry_utc_hour", "entry_quote", "entry_fill", "entry_ev", "p_up_entry",
    "p_side_entry", "shares", "resolved_win", "exit_type", "exit_reason", "pnl",
]


def write_trades_csv(path: Path, trades: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(trades)


def compare(label: str, ours: dict, lock_ref: dict | None) -> list[str]:
    lines = [f"### {label}"]
    keymap = [
        ("trades", "trades"), ("total_pnl", "pnl"), ("avg_pnl", "avg_pnl"),
        ("median_pnl", "median_pnl"), ("positive_days", "positive_days"),
        ("active_days", "active_days"), ("worst_day", "worst_day"),
        ("max_drawdown", "max_drawdown"),
    ]
    for ok, lk in keymap:
        ov = ours.get(ok)
        lv = (lock_ref or {}).get(lk)
        if isinstance(ov, float):
            ov_s = f"{ov:.4f}"
        else:
            ov_s = str(ov)
        if lv is None:
            lines.append(f"  {ok:16s} ours={ov_s}   lock=—")
        else:
            lv_f = float(lv)
            match = "MATCH" if (isinstance(ov, (int, float)) and abs(float(ov) - lv_f) < 0.51) else "DIFF"
            lines.append(f"  {ok:16s} ours={ov_s:>12s}   lock={lv_f:<12.4f} {match}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--tcn-artifacts", type=Path, default=DEFAULT_TCN)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fill-mode", choices=["cap_drop", "fill_worse"], default="cap_drop",
                    help="cap_drop = lock semantics; fill_worse = honest delayed-fill variant")
    ap.add_argument("--delay-s", type=float, default=None,
                    help="override buy delay (latency sensitivity); default = lock value")
    ap.add_argument("--delta-cache", type=Path, default=None,
                    help="npy file with precomputed per-step deltas (skips model forward)")
    ap.add_argument("--torch-threads", type=int, default=8)
    args = ap.parse_args()

    lock_path = args.lock if args.lock.is_absolute() else ROOT / args.lock
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    tcn_dir = args.tcn_artifacts if args.tcn_artifacts.is_absolute() else ROOT / args.tcn_artifacts
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    specs, lock = load_lock(lock_path)
    print(f"lock={lock.get('lock_id')} strategies={[s.strategy_id for s in specs]}")
    t0 = time.time()
    delta = None
    if args.delta_cache is not None and args.delta_cache.exists():
        import numpy as np
        delta = np.load(args.delta_cache)
        print(f"using cached deltas from {args.delta_cache}", flush=True)
    else:
        import torch
        torch.set_num_threads(max(1, args.torch_threads))
        print(f"loading split={args.split} + forwarding TCN (cpu ok, a few minutes)...", flush=True)
    data = load_episode_split(dataset, args.split, tcn_dir=tcn_dir,
                              batch_size=args.batch_size, delta=delta)
    print(f"loaded {data.n_ep} episodes in {time.time() - t0:.1f}s", flush=True)

    result = run_locked_strategies(data, specs, HOLD,
                                   fill_mode=args.fill_mode, delay_s=args.delay_s)

    variant = f"{args.split}_{args.fill_mode}" + (f"_delay{args.delay_s:g}" if args.delay_s is not None else "")
    all_trades = result["combined"]["trades"]
    write_trades_csv(out_dir / f"trades_{variant}.csv", all_trades)

    report = {
        "lock_id": lock.get("lock_id"),
        "lock_path": str(lock_path),
        "dataset": str(dataset),
        "tcn_artifacts": str(tcn_dir),
        "split": args.split,
        "fill_mode": args.fill_mode,
        "delay_s_override": args.delay_s,
        "elapsed_s": round(time.time() - t0, 2),
        "per_strategy": {
            sid: {"spec": s["spec"], "summary": s["summary"]}
            for sid, s in result["per_strategy"].items()
        },
        "combined": result["combined"]["summary"],
    }
    (out_dir / f"summary_{variant}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    # comparison vs the lock's canonical numbers (only meaningful for the
    # default test/cap_drop/no-delay-override run)
    ref = lock.get("canonical_clean_rerun_metrics", {})
    lines: list[str] = [f"split={args.split} fill_mode={args.fill_mode} delay={args.delay_s or 'lock'}"]
    ref_by_sid = {"early_tcn_75_120_utc02_13": ref.get("early"),
                  "late_tcn_50_75_all_day": ref.get("late")}
    for sid, s in result["per_strategy"].items():
        lines += compare(sid, s["summary"], ref_by_sid.get(sid))
    lines += compare("combined", result["combined"]["summary"], ref.get("combined_two_strategy"))
    comb = result["combined"]["summary"]
    ref_comb = ref.get("combined_two_strategy") or {}
    lines.append(f"  overlap_markets  ours={comb.get('overlap_markets')}   lock={ref_comb.get('overlap_markets')}")
    text = "\n".join(lines)
    print(text)
    (out_dir / f"comparison_{variant}.txt").write_text(text, encoding="utf-8")
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
