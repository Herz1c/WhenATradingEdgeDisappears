#!/usr/bin/env python3
"""Run the shadow decision pipeline across many existing dense_close
parquets and aggregate per-day + per-hour metrics.

Designed for fast iteration: takes already-built parquets and slices
them by hour in-process — no slow dataset rebuilds. A 24-hour sweep
finishes in ~5 seconds because the model + decision engine are vectorized.

Usage:
    py -3 tools/shadow_sweep.py \\
        --parquets "data/datasets/resolution_snapshot_dataset_v1_dense_close/2026-05-1[7-9].parquet,data/datasets/resolution_snapshot_dataset_v1_dense_close/2026-05-20.parquet" \\
        --tag may17-20

Or a single glob expanded by the shell:
    py -3 tools/shadow_sweep.py \\
        --parquets data/datasets/resolution_snapshot_dataset_v1_dense_close/2026-05-{17,18,19,20}.parquet \\
        --tag may17-20

Outputs:
  - logs/shadow/<tag>_full.jsonl                — concatenated decision stream
  - docs/shadow_sweep/<tag>_summary.md          — markdown aggregate
  - docs/shadow_sweep/<tag>_per_hour.json       — machine-readable per-hour
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import polars as pl

# Reuse the production engine
from live.shadow_runtime import (   # noqa: E402
    TTC_MIN_S, TTC_MAX_S, SECOND_NS,
    RiskState, decide, load_model, build_X,
)
from feature_cleanup import clean_features  # noqa: E402
from backtest.fees import FeeCalculator      # noqa: E402

os.environ.setdefault("FEATURE_CLEANUP_ENABLED", "1")


def _hour_key(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, UTC).strftime("%Y-%m-%d_%H")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquets", required=True,
                    help="Comma-separated parquet paths (or a shell-expanded list joined by commas).")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    paths = [Path(p.strip()) for p in args.parquets.split(",") if p.strip()]
    for p in paths:
        if not p.exists():
            print(f"FATAL: parquet does not exist: {p}")
            return 2
    print(f"sweeping {len(paths)} parquet(s):")
    for p in paths:
        print(f"  {p}")

    model, feats = load_model()

    sweep_dir = REPO / "docs" / "shadow_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    full_jsonl = REPO / "logs" / "shadow" / f"{args.tag}_full.jsonl"
    full_jsonl.parent.mkdir(parents=True, exist_ok=True)

    per_hour: dict[str, dict] = defaultdict(lambda: {
        "rows": 0, "enters": 0, "realized_fills": 0, "realized_pnl_usd": 0.0,
        "wins": 0, "total_notional": 0.0,
    })
    per_day: dict[str, dict] = defaultdict(lambda: {
        "rows": 0, "enters": 0, "realized_fills": 0, "realized_pnl_usd": 0.0,
        "wins": 0, "total_notional": 0.0,
    })

    grand_state = RiskState()    # Cross-day risk reset: instead share a fresh state per day
    fee_calcs: dict[str, FeeCalculator] = {}

    total_processed = 0
    total_enters = 0
    total_realized = 0
    total_pnl = 0.0

    with full_jsonl.open("w", encoding="utf-8") as fh:
        for parquet in sorted(paths):
            df = pl.read_parquet(parquet)
            df = df.filter(pl.col("up_token_best_bid").is_not_null() &
                           pl.col("up_token_best_ask").is_not_null())
            df = df.sort(["snapshot_ts_ns", "market_slug"])
            df_clean = clean_features(df)
            X = build_X(df_clean, feats)
            p_up_all = model.predict_proba(X)[:, 1]

            up_bid  = df["up_token_best_bid"].to_numpy().astype(float)
            up_ask  = df["up_token_best_ask"].to_numpy().astype(float)
            ttc     = df["t_to_close_s"].to_numpy().astype(float)
            market  = df["market_slug"].to_numpy()
            snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
            close_ts = (
                df["market_close_ts_ns"].to_numpy().astype(np.int64)
                if "market_close_ts_ns" in df.columns
                else snap_ts + (300 * SECOND_NS)
            )
            resolved = (
                df["resolved_side_label"].to_numpy().astype(int)
                if "resolved_side_label" in df.columns
                else np.full(len(df), -1, dtype=int)
            )

            # Fresh risk state per parquet (= per day) — mirrors the live bot
            # which would reset margin/state across UTC days.
            state = RiskState()

            for i in range(len(df)):
                now_ns = int(snap_ts[i])
                state.release_closed(now_ns)

                snap = {
                    "snapshot_ts_ns":    now_ns,
                    "market_slug":       str(market[i]),
                    "up_token_best_bid": float(up_bid[i]),
                    "up_token_best_ask": float(up_ask[i]),
                    "t_to_close_s":      float(ttc[i]),
                }
                d = decide(snap=snap, p_up=float(p_up_all[i]), state=state)

                hk = _hour_key(now_ns)
                dk = hk.split("_")[0]
                per_hour[hk]["rows"] += 1
                per_day [dk]["rows"] += 1
                total_processed += 1

                in_band = TTC_MIN_S <= snap["t_to_close_s"] <= TTC_MAX_S
                if in_band or d["decision"] == "ENTER":
                    fh.write(json.dumps({
                        "snapshot_ts_ns":     now_ns,
                        "market_slug":        snap["market_slug"],
                        "t_to_close_s":       snap["t_to_close_s"],
                        "up_token_best_bid":  snap["up_token_best_bid"],
                        "up_token_best_ask":  snap["up_token_best_ask"],
                        "p_up":               d["p_up"],
                        "edge_dn":            d["edge_dn"],
                        "decision":           d["decision"],
                        "reason":             d["reason"],
                        "notional":           d["notional"],
                    }, separators=(",", ":")) + "\n")

                if d["decision"] == "ENTER":
                    entry_price = 1.0 - snap["up_token_best_bid"]
                    shares = d["notional"] / entry_price
                    fc = fee_calcs.setdefault(dk, FeeCalculator.for_date(dk))
                    fee = fc.taker_fee_usd(price=entry_price, size=shares)

                    per_hour[hk]["enters"] += 1
                    per_day [dk]["enters"] += 1
                    per_hour[hk]["total_notional"] += d["notional"]
                    per_day [dk]["total_notional"] += d["notional"]
                    total_enters += 1

                    if resolved[i] >= 0:
                        win = (resolved[i] == 0)
                        payoff = shares * 1.0 if win else 0.0
                        pnl = payoff - d["notional"] - fee
                        per_hour[hk]["realized_fills"] += 1
                        per_day [dk]["realized_fills"] += 1
                        per_hour[hk]["realized_pnl_usd"] += pnl
                        per_day [dk]["realized_pnl_usd"] += pnl
                        if win:
                            per_hour[hk]["wins"] += 1
                            per_day [dk]["wins"]  += 1
                        total_realized += 1
                        total_pnl += pnl

                    state.add_lot(close_ns=int(close_ts[i]), notional=d["notional"])
                    state.positions_per_market[snap["market_slug"]] += 1
                    state.last_entry_ns_per_market[snap["market_slug"]] = now_ns

            print(f"  {parquet.stem}: rows={len(df):,}  enters={sum(1 for h in per_day if h)} (across all so far)")

    # ── aggregate ─────────────────────────────────────────────────────────────
    per_hour_json = sweep_dir / f"{args.tag}_per_hour.json"
    per_hour_json.write_text(json.dumps(
        {k: per_hour[k] for k in sorted(per_hour)},
        indent=2, default=float
    ))

    wr = (sum(d["wins"] for d in per_day.values()) /
          max(1, sum(d["realized_fills"] for d in per_day.values())))
    weekly_extrap = total_pnl / max(1, len(per_day)) * 7
    pnl_per_hour = [v["realized_pnl_usd"] for v in per_hour.values()]
    enters_per_hour = [v["enters"] for v in per_hour.values()]

    lines: list[str] = []
    add = lines.append
    add(f"# Shadow sweep — `{args.tag}`\n")
    add(f"- parquets: **{len(paths)}**")
    add(f"- days covered: **{len(per_day)}**")
    add(f"- hours covered: **{len(per_hour)}**")
    add(f"- total rows processed: **{total_processed:,}**")
    add(f"- total ENTERs: **{total_enters}**")
    add(f"- total realized fills: **{total_realized}**")
    add(f"- total realized PnL: **${total_pnl:+.2f}**")
    add(f"- overall win rate: **{wr*100:.1f}%**")
    add(f"- mean PnL/hour: ${np.mean(pnl_per_hour):+.2f}")
    add(f"- worst hour: ${min(pnl_per_hour):+.2f} | best hour: ${max(pnl_per_hour):+.2f}")
    add(f"- mean ENTERs/hour: {np.mean(enters_per_hour):.1f}")
    add(f"- weekly PnL extrapolated: **${weekly_extrap:+.2f}**")
    add("")
    add(f"## Per-day")
    add(f"| date | rows | enters | realized | wins | WR | notional | PnL |")
    add(f"|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in sorted(per_day):
        v = per_day[d]
        dwr = (v["wins"] / max(1, v["realized_fills"]) * 100) if v["realized_fills"] else 0
        add(f"| {d} | {v['rows']:,} | {v['enters']} | {v['realized_fills']} | "
            f"{v['wins']} | {dwr:.1f}% | ${v['total_notional']:.2f} | "
            f"${v['realized_pnl_usd']:+.2f} |")
    add("")

    add(f"## Per-hour")
    add(f"| hour (UTC) | rows | enters | realized | PnL |")
    add(f"|---|---:|---:|---:|---:|")
    for h in sorted(per_hour):
        v = per_hour[h]
        add(f"| {h} | {v['rows']:,} | {v['enters']} | {v['realized_fills']} | ${v['realized_pnl_usd']:+.2f} |")

    summary = sweep_dir / f"{args.tag}_summary.md"
    summary.write_text("\n".join(lines))

    print(f"\n=== SWEEP DONE ===")
    print(f"  days covered:    {len(per_day)}")
    print(f"  hours covered:   {len(per_hour)}")
    print(f"  total enters:    {total_enters}")
    print(f"  total realized:  {total_realized}")
    print(f"  win rate:        {wr*100:.1f}%")
    print(f"  total PnL:       ${total_pnl:+.2f}")
    print(f"  weekly extrap:   ${weekly_extrap:+.2f}")
    print(f"  summary:         {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
