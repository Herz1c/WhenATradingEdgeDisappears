#!/usr/bin/env python3
"""Diff the live bot's decisions on date D against what the backtest
decision engine would have produced from the freshly-built
dense_close parquet for the same date.

Three columns of analysis per (market_slug, snapshot_ts_ns):
  1. backtest p_up  vs  live p_up   — feature/model divergence
  2. backtest decision vs live decision — whose ENTERs fire
  3. for each LIVE ENTER, what the backtest would have said:
        - matches?  → bot got an edge that backtest also saw
        - skips?    → bot fired but backtest wouldn't (live-only signal)
  4. for each BACKTEST ENTER missing from live:
        - market was open during live, but bot skipped it
        - root cause: stale feature? risk-blocked? rate-gap?

Run from repo root:
    py -3 tools/diff_live_vs_backtest.py \\
        --date 2026-05-26 \\
        --bot-log  logs/live_bot/decisions_2026-05-26.jsonl \\
        --parquet  data/datasets/resolution_snapshot_dataset_v1_dense_close/2026-05-26.parquet \\
        --output   docs/live_vs_backtest/2026-05-26.md
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("FEATURE_CLEANUP_ENABLED", "1")

import joblib
import numpy as np
import polars as pl

from feature_cleanup import clean_features
from live_bot.decision_engine import (
    THRESHOLD, EDGE_K, MIN_NOTIONAL, MAX_NOTIONAL,
    MAX_POSITIONS_PER_MARKET, MIN_SECONDS_BETWEEN_ENTRIES, MAX_MARGIN_USD,
    TTC_MIN_S, TTC_MAX_S, SECOND_NS, RiskState, decide,
)
from backtest.fees import FeeCalculator

ART = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")


def _load_model():
    model = joblib.load(ART / "model.pkl")
    assert getattr(model, "_calibrator", None) is None, "model must have calibrator stripped"
    feats = list(json.loads((ART / "feature_importance.json").read_text()).keys())
    return model, feats


def _build_X(df_clean: pl.DataFrame, feats: list[str]) -> np.ndarray:
    X = np.zeros((len(df_clean), len(feats)), dtype=np.float32)
    for i, f in enumerate(feats):
        if f in df_clean.columns:
            s = df_clean.get_column(f)
            if s.dtype.is_numeric():
                v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
                X[:, i] = np.where(np.isfinite(v), v, 0.0)
    return X


def replay_backtest(parquet: Path, *, env_min: float = 1.00, env_max: float = 1.00,
                    env_max_pos: int = 2, env_gap_s: float = 10.0) -> list[dict]:
    """Replay the backtest decision engine on the dense_close parquet,
    using identical sizing/risk parameters to what the bot used in live.
    Returns a list of decision dicts (one per snapshot in ttc band)."""
    model, feats = _load_model()
    df = pl.read_parquet(parquet)
    df = df.filter(pl.col("up_token_best_bid").is_not_null() &
                   pl.col("up_token_best_ask").is_not_null())
    df = df.sort(["snapshot_ts_ns", "market_slug"])
    df_clean = clean_features(df)
    X = _build_X(df_clean, feats)
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

    state = RiskState()
    rows: list[dict] = []
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
        rec = {
            "snapshot_ts_ns": now_ns,
            "market_slug":    snap["market_slug"],
            "t_to_close_s":   snap["t_to_close_s"],
            "up_token_best_bid": snap["up_token_best_bid"],
            "p_up_backtest":  d["p_up"],
            "edge_dn_backtest": d["edge_dn"],
            "decision_backtest": d["decision"],
            "reason_backtest": d["reason"],
            "notional_backtest": d["notional"],
            "resolved_side_label": int(resolved[i]) if resolved[i] >= 0 else None,
            "market_close_ts_ns": int(close_ts[i]),
        }
        rows.append(rec)
        if d["decision"] == "ENTER":
            state.add_lot(close_ns=int(close_ts[i]), notional=d["notional"])
            state.positions_per_market[snap["market_slug"]] += 1
            state.last_entry_ns_per_market[snap["market_slug"]] = now_ns
    return rows


def _load_bot_decisions(jsonl: Path) -> list[dict]:
    out = []
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out


def _bucket_ts_to_grid_ns(ts_ns: int) -> int:
    """Snap a wall-clock ns to the dense_close 250ms / 100ms grid.
    Used to map bot decisions (which fire at wall-clock now) to the
    closest backtest snapshot ts."""
    # 250ms grid for ttc > 10s, 100ms grid for ttc <= 10s
    # but for join purposes any 250ms-or-better grid works
    return (ts_ns // 250_000_000) * 250_000_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--bot-log", required=True, type=Path)
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--output",  required=True, type=Path)
    args = ap.parse_args()

    print(f"loading bot log:   {args.bot_log}")
    bot_decs = _load_bot_decisions(args.bot_log)
    print(f"  {len(bot_decs):,} bot decisions")

    print(f"replaying backtest on: {args.parquet}")
    bt_decs = replay_backtest(args.parquet)
    print(f"  {len(bt_decs):,} backtest snapshot rows (post-filter)")

    # Index backtest by (market_slug, snapshot_ts_ns)
    bt_idx: dict[tuple[str, int], dict] = {(r["market_slug"], r["snapshot_ts_ns"]): r for r in bt_decs}
    bot_idx: dict[tuple[str, int], dict] = {}

    # The bot's snapshot_ts_ns IS the dense_close grid ts (we use the same
    # _build_snapshot_row code-path), so direct join should work. If not,
    # try a small tolerance.
    for d in bot_decs:
        bot_idx[(d["market_slug"], d["snapshot_ts_ns"])] = d

    # Categorize
    bot_enters = [d for d in bot_decs if d["decision"] == "ENTER" and not d.get("risk_blocked")]
    bt_enters  = [r for r in bt_decs  if r["decision_backtest"] == "ENTER"]
    print(f"\nbot ENTERs    (non-risk-blocked): {len(bot_enters):,}")
    print(f"backtest ENTERs:                   {len(bt_enters):,}")

    # For each bot ENTER, find matching backtest row
    matched_enters = 0
    bot_only_enters: list[dict] = []
    bot_enters_p_up_delta: list[float] = []
    for be in bot_enters:
        key = (be["market_slug"], be["snapshot_ts_ns"])
        bt = bt_idx.get(key)
        if bt is None:
            bot_only_enters.append(be); continue
        if bt["decision_backtest"] == "ENTER":
            matched_enters += 1
        else:
            bot_only_enters.append({**be, "_bt_reason": bt["reason_backtest"],
                                    "_bt_p_up": bt["p_up_backtest"],
                                    "_bt_edge": bt["edge_dn_backtest"]})
        if be.get("p_up") is not None and bt.get("p_up_backtest") is not None:
            bot_enters_p_up_delta.append(float(be["p_up"]) - float(bt["p_up_backtest"]))

    # For each backtest ENTER, was it in bot log?
    bt_only_enters: list[dict] = []
    bt_in_bot_enter = 0
    bt_in_bot_skip = 0
    bt_in_bot_blocked = 0
    bt_missing = 0
    for bt in bt_enters:
        key = (bt["market_slug"], bt["snapshot_ts_ns"])
        be = bot_idx.get(key)
        if be is None:
            bt_missing += 1
            bt_only_enters.append({**bt, "_bot": "MISSING"}); continue
        if be["risk_blocked"]:
            bt_in_bot_blocked += 1
            bt_only_enters.append({**bt, "_bot": "risk_blocked",
                                   "_bot_reason": be["risk_reason"]}); continue
        if be["decision"] == "ENTER":
            bt_in_bot_enter += 1
        else:
            bt_in_bot_skip += 1
            bt_only_enters.append({**bt, "_bot": "skip", "_bot_reason": be["reason"]})

    # Build the report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    add = lines.append
    add(f"# Live bot vs backtest — {args.date}\n")
    add(f"- bot log:   `{args.bot_log}`")
    add(f"- parquet:   `{args.parquet}`\n")

    add(f"## Counts\n")
    add(f"| metric | value |")
    add(f"|---|---:|")
    add(f"| bot decision rows logged | {len(bot_decs):,} |")
    add(f"| backtest snapshot rows | {len(bt_decs):,} |")
    add(f"| bot ENTERs (non-risk-blocked) | {len(bot_enters):,} |")
    add(f"| backtest ENTERs | {len(bt_enters):,} |")
    add(f"| ENTERs in BOTH | {matched_enters:,} |")
    add(f"| ENTERs bot-only (backtest would skip) | {sum(1 for e in bot_only_enters if e.get('_bt_reason'))} |")
    add(f"| ENTERs bot tried but row missing from backtest | {sum(1 for e in bot_only_enters if not e.get('_bt_reason'))} |")
    add(f"| backtest-only ENTERs blocked in bot by risk gate | {bt_in_bot_blocked:,} |")
    add(f"| backtest-only ENTERs bot SKIPPED | {bt_in_bot_skip:,} |")
    add(f"| backtest-only ENTERs missing from bot log entirely | {bt_missing:,} |")
    add("")

    if bot_enters_p_up_delta:
        arr = np.array(bot_enters_p_up_delta)
        add(f"## `p_up` delta on bot's ENTERs (live − backtest)\n")
        add(f"- n = {len(arr):,}")
        add(f"- mean: {arr.mean():+.4f}")
        add(f"- median: {np.median(arr):+.4f}")
        add(f"- min / max: {arr.min():+.4f}  /  {arr.max():+.4f}")
        add(f"- abs P50 / P95 / P99: {np.percentile(np.abs(arr),50):.4f}  /  "
            f"{np.percentile(np.abs(arr),95):.4f}  /  {np.percentile(np.abs(arr),99):.4f}\n")

    # First 25 bot-only ENTERs (bot fired, backtest would skip)
    bot_only_with_bt = [e for e in bot_only_enters if e.get("_bt_reason")]
    if bot_only_with_bt:
        add(f"## First 25 BOT-ONLY ENTERs (bot fired, backtest says skip)\n")
        add(f"| ts | market | ttc | up_bid | live p_up | live edge | bt p_up | bt edge | bt reason | filled? |")
        add(f"|---|---|---:|---:|---:|---:|---:|---:|---|---|")
        for e in bot_only_with_bt[:25]:
            rc = e.get("receipt") or {}
            filled = "yes" if rc.get("success") and rc.get("filled_size", 0) > 0 else ("KILLED" if rc.get("status") == "error" else "?")
            add(f"| {e['snapshot_iso']} | {e['market_slug'][-15:]} | {e['t_to_close_s']:.1f} | "
                f"{e['up_token_best_bid']:.3f} | {e['p_up']:.4f} | {e['edge_dn']:+.4f} | "
                f"{(e.get('_bt_p_up') or 0):.4f} | {(e.get('_bt_edge') or 0):+.4f} | "
                f"{e.get('_bt_reason')} | {filled} |")
        add("")

    # First 25 backtest-only ENTERs (backtest would fire, bot didn't)
    if bt_only_enters:
        add(f"## First 25 BACKTEST-ONLY ENTERs (backtest fires, bot skips/blocks)\n")
        add(f"| ts | market | ttc | up_bid | bt p_up | bt edge | bot status | bot reason |")
        add(f"|---|---|---:|---:|---:|---:|---|---|")
        from datetime import datetime as _dt
        for e in bt_only_enters[:25]:
            iso = _dt.fromtimestamp(e['snapshot_ts_ns']/1e9, UTC).isoformat(timespec='seconds')
            add(f"| {iso} | {e['market_slug'][-15:]} | {e['t_to_close_s']:.1f} | "
                f"{e['up_token_best_bid']:.3f} | {(e.get('p_up_backtest') or 0):.4f} | "
                f"{(e.get('edge_dn_backtest') or 0):+.4f} | {e.get('_bot','?')} | "
                f"{e.get('_bot_reason','?')[:60]} |")
        add("")

    args.output.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== SUMMARY ===")
    print(f"  matched ENTERs:           {matched_enters:,}")
    print(f"  bot-only ENTERs:          {len(bot_only_with_bt):,} (backtest would skip these)")
    print(f"  backtest-only ENTERs:     {bt_in_bot_skip + bt_in_bot_blocked + bt_missing:,}")
    print(f"    - blocked by risk gate: {bt_in_bot_blocked:,}")
    print(f"    - bot skipped:          {bt_in_bot_skip:,}")
    print(f"    - missing from bot log: {bt_missing:,}")
    if bot_enters_p_up_delta:
        arr = np.array(bot_enters_p_up_delta)
        print(f"  p_up delta (live − bt):  mean={arr.mean():+.4f}  abs_P99={np.percentile(np.abs(arr),99):.4f}")
    print(f"\nReport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
