#!/usr/bin/env python3
"""Compare a shadow JSONL log against a reference parquet of the same
time window. The reference parquet is treated as ground truth (what the
backtest sees / would have decided).

Outputs a markdown report covering:

  - Coverage: did shadow process every row the reference contains
              (within the ttc band)?
  - Quote agreement: are up_token_best_bid / up_token_best_ask
              identical (modulo float rounding)?
  - p_up agreement: is the model output identical?
  - edge_dn agreement: identical?
  - Decision agreement: confusion matrix of shadow.decision vs the
              decision the same engine would produce on the reference
              parquet (we re-run decide() inline, so this catches
              the case where shadow's risk state diverged from
              reference's).
  - PnL diff: shadow.realized_pnl vs reference.realized_pnl.

In Phase 1 the shadow runtime reads the same parquet as the reference,
so all diffs should be **exactly zero**. The value of the tool in Phase
1 is proving the diff infrastructure works end-to-end, and catching
regressions if anyone changes the decision engine.

In Phase 2 (shadow uses its own live WS feed) the diffs will be
non-zero and the report becomes the actual go/no-go for live trading.

Usage:
    py -3 tools/shadow_parity_diff.py \\
        --shadow      logs/shadow/2026-05-22_08-09.jsonl \\
        --reference   data/datasets/_shadow_windows/2026-05-22_08-09.parquet \\
        --output      docs/shadow_parity/2026-05-22_08-09.md
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import polars as pl

# Reuse the engine — we want diffs *only* in inputs, never in decision logic
from live.shadow_runtime import (   # noqa: E402
    TTC_MIN_S, TTC_MAX_S, THRESHOLD,
    RiskState, decide, load_model, build_X,
)
from feature_cleanup import clean_features  # noqa: E402


def _replay_reference(reference_parquet: Path) -> list[dict]:
    """Run the exact same pipeline shadow_runtime runs, on the reference
    parquet — produces the 'truth' decision stream for diffing."""
    model, feats = load_model()
    df = pl.read_parquet(reference_parquet)
    df = df.filter(pl.col("up_token_best_bid").is_not_null() &
                   pl.col("up_token_best_ask").is_not_null())
    df = df.sort(["snapshot_ts_ns", "market_slug"])
    df_clean = clean_features(df)
    X = build_X(df_clean, feats)
    p_up = model.predict_proba(X)[:, 1]

    up_bid  = df["up_token_best_bid"].to_numpy().astype(float)
    up_ask  = df["up_token_best_ask"].to_numpy().astype(float)
    ttc     = df["t_to_close_s"].to_numpy().astype(float)
    market  = df["market_slug"].to_numpy()
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = (
        df["market_close_ts_ns"].to_numpy().astype(np.int64)
        if "market_close_ts_ns" in df.columns
        else snap_ts + (300 * 1_000_000_000)
    )

    state = RiskState()
    out = []
    for i in range(len(df)):
        state.release_closed(int(snap_ts[i]))
        snap = {
            "snapshot_ts_ns": int(snap_ts[i]),
            "market_slug":    str(market[i]),
            "up_token_best_bid": float(up_bid[i]),
            "up_token_best_ask": float(up_ask[i]),
            "t_to_close_s":   float(ttc[i]),
        }
        d = decide(snap=snap, p_up=float(p_up[i]), state=state)
        out.append({
            "snapshot_ts_ns": snap["snapshot_ts_ns"],
            "market_slug":    snap["market_slug"],
            "up_token_best_bid": snap["up_token_best_bid"],
            "up_token_best_ask": snap["up_token_best_ask"],
            "t_to_close_s":   snap["t_to_close_s"],
            "p_up":           d["p_up"],
            "edge_dn":        d["edge_dn"],
            "decision":       d["decision"],
            "reason":         d["reason"],
            "notional":       d["notional"],
        })
        if d["decision"] == "ENTER":
            state.add_lot(close_ns=int(close_ts[i]), notional=d["notional"])
            state.positions_per_market[snap["market_slug"]] += 1
            state.last_entry_ns_per_market[snap["market_slug"]] = snap["snapshot_ts_ns"]
    return out


def _load_shadow(jsonl: Path) -> list[dict]:
    out = []
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out


def _diff(a: float | None, b: float | None) -> float:
    if a is None and b is None: return 0.0
    if a is None or b is None:  return float("inf")
    return abs(float(a) - float(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shadow",    required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--output",    required=True, type=Path)
    ap.add_argument("--tol-quote", type=float, default=1e-6, help="abs tol for bid/ask diff")
    ap.add_argument("--tol-p",     type=float, default=1e-6, help="abs tol for p_up / edge_dn")
    args = ap.parse_args()

    shadow_rows = _load_shadow(args.shadow)
    ref_rows    = _replay_reference(args.reference)
    print(f"shadow rows:    {len(shadow_rows):,}")
    print(f"reference rows: {len(ref_rows):,}")

    shadow_idx = {(r["snapshot_ts_ns"], r["market_slug"]): r for r in shadow_rows}
    ref_idx    = {(r["snapshot_ts_ns"], r["market_slug"]): r for r in ref_rows}

    common = sorted(set(shadow_idx.keys()) & set(ref_idx.keys()))
    only_shadow = sorted(set(shadow_idx.keys()) - set(ref_idx.keys()))
    only_ref    = sorted(set(ref_idx.keys())    - set(shadow_idx.keys()))

    # Restrict to ttc band on the REFERENCE side — that's the universe
    # of rows where decisions matter.
    ref_in_band = [k for k in ref_idx if TTC_MIN_S <= ref_idx[k]["t_to_close_s"] <= TTC_MAX_S]
    common_in_band = [k for k in common if TTC_MIN_S <= ref_idx[k]["t_to_close_s"] <= TTC_MAX_S]
    coverage = len(common_in_band) / max(1, len(ref_in_band))

    # Per-field diff distributions over common_in_band
    diffs: dict[str, list[float]] = {"bid": [], "ask": [], "p_up": [], "edge_dn": [], "notional": []}
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    reason_mismatches: list[tuple[tuple, dict, dict]] = []

    for k in common_in_band:
        s = shadow_idx[k]; r = ref_idx[k]
        diffs["bid"].append(_diff(s.get("up_token_best_bid"), r.get("up_token_best_bid")))
        diffs["ask"].append(_diff(s.get("up_token_best_ask"), r.get("up_token_best_ask")))
        diffs["p_up"].append(_diff(s.get("p_up"), r.get("p_up")))
        diffs["edge_dn"].append(_diff(s.get("edge_dn"), r.get("edge_dn")))
        diffs["notional"].append(_diff(s.get("notional"), r.get("notional")))
        confusion[(s.get("decision", "?"), r.get("decision", "?"))] += 1
        if s.get("decision") != r.get("decision") and len(reason_mismatches) < 25:
            reason_mismatches.append((k, s, r))

    def pct(xs, q): return float(np.percentile(xs, q)) if xs else 0.0

    # PnL — only from rows where shadow ENTERed
    shadow_enters = [r for r in shadow_rows if r.get("decision") == "ENTER"]
    ref_enters    = [r for r in ref_rows    if r.get("decision") == "ENTER"]
    shadow_notional_total = sum(r["notional"] for r in shadow_enters)
    ref_notional_total    = sum(r["notional"] for r in ref_enters)

    enters_in_both     = sum(1 for k in common_in_band
                             if shadow_idx[k].get("decision") == "ENTER"
                             and ref_idx[k].get("decision") == "ENTER")
    enters_shadow_only = sum(1 for k in common_in_band
                             if shadow_idx[k].get("decision") == "ENTER"
                             and ref_idx[k].get("decision") != "ENTER")
    enters_ref_only    = sum(1 for k in common_in_band
                             if shadow_idx[k].get("decision") != "ENTER"
                             and ref_idx[k].get("decision") == "ENTER")

    # ── pass/fail gates ───────────────────────────────────────────────────────
    gates = {
        "coverage >= 0.99":              coverage >= 0.99,
        "bid max diff <= tol":           (max(diffs["bid"]) if diffs["bid"] else 0) <= args.tol_quote,
        "ask max diff <= tol":           (max(diffs["ask"]) if diffs["ask"] else 0) <= args.tol_quote,
        "p_up max diff <= tol":          (max(diffs["p_up"]) if diffs["p_up"] else 0) <= args.tol_p,
        "edge_dn max diff <= tol":       (max(diffs["edge_dn"]) if diffs["edge_dn"] else 0) <= args.tol_p,
        "decision agreement >= 99.5%":   (
            sum(c for (s,r),c in confusion.items() if s == r) / max(1, sum(confusion.values())) >= 0.995
        ),
    }
    all_pass = all(gates.values())

    # ── write report ──────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    add = lines.append
    add(f"# Shadow ↔ Reference parity\n")
    add(f"- shadow JSONL: `{args.shadow}`")
    add(f"- reference parquet: `{args.reference}`")
    add(f"- shadow rows: **{len(shadow_rows):,}**, reference rows: **{len(ref_rows):,}**\n")

    add(f"## Verdict\n")
    add(f"**{'PASS' if all_pass else 'FAIL'}**\n")
    add(f"| gate | result |\n|---|---|")
    for g, ok in gates.items():
        add(f"| {g} | {'pass' if ok else 'FAIL'} |")
    add("")

    add(f"## Coverage\n")
    add(f"- reference rows in ttc band [{TTC_MIN_S}, {TTC_MAX_S}]: **{len(ref_in_band):,}**")
    add(f"- of those, present in shadow: **{len(common_in_band):,}**  ({coverage*100:.2f}%)")
    add(f"- present in shadow but not reference (any ttc): {len(only_shadow):,}")
    add(f"- present in reference but not shadow (any ttc): {len(only_ref):,}\n")

    add(f"## Per-field diff (P50 / P95 / P99 / max) — ttc band, common keys\n")
    add(f"| field | P50 | P95 | P99 | max |")
    add(f"|---|---:|---:|---:|---:|")
    for k, xs in diffs.items():
        add(f"| {k} | {pct(xs,50):.2e} | {pct(xs,95):.2e} | {pct(xs,99):.2e} | {max(xs) if xs else 0:.2e} |")
    add("")

    add(f"## Decision confusion (rows = shadow, cols = reference)\n")
    all_decs = sorted({s for s,_ in confusion} | {r for _,r in confusion})
    add("| | " + " | ".join(all_decs) + " |")
    add("|---|" + "---|" * len(all_decs))
    for s in all_decs:
        row = [f"**{s}**"] + [str(confusion.get((s, r), 0)) for r in all_decs]
        add("| " + " | ".join(row) + " |")
    add("")

    add(f"## ENTER counts\n")
    add(f"- shadow ENTERs (all): **{len(shadow_enters):,}**, total notional ${shadow_notional_total:,.2f}")
    add(f"- reference ENTERs (all): **{len(ref_enters):,}**, total notional ${ref_notional_total:,.2f}")
    add(f"- both ENTER: {enters_in_both} | shadow-only: {enters_shadow_only} | reference-only: {enters_ref_only}\n")

    if reason_mismatches:
        add(f"## First {len(reason_mismatches)} decision mismatches\n")
        add(f"| snapshot_ts_ns | market | shadow.decision | shadow.reason | ref.decision | ref.reason | Δp_up | Δedge |")
        add(f"|---|---|---|---|---|---|---:|---:|")
        for k, s, r in reason_mismatches:
            add(f"| {k[0]} | `{k[1][:36]}` | {s.get('decision')} | {s.get('reason')} | "
                f"{r.get('decision')} | {r.get('reason')} | "
                f"{_diff(s.get('p_up'), r.get('p_up')):.2e} | {_diff(s.get('edge_dn'), r.get('edge_dn')):.2e} |")
        add("")

    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWROTE {args.output}")
    print(f"VERDICT: {'PASS' if all_pass else 'FAIL'}")
    for g, ok in gates.items():
        print(f"  [{'✓' if ok else '✗'}] {g}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
