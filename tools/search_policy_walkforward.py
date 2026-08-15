"""Walk-forward policy re-derivation over the regenerated search universe (Phase 5.1).

The anti-snooping audit killed the historical justification of both locked
slots (selection on test). This script tests the *selection process* honestly:

  On an expanding window of model-OOS days, pick the best candidate from the
  full universe by a fixed rule, then measure it on the NEXT unseen 5-day
  block (1-day embargo). Chain the OOS blocks -> the walk-forward equity of
  the search process itself. If that chained OOS PnL is not positive, no
  amount of re-searching this universe is trustworthy.

Input: artifacts/audit_v1/wrc_universe_daily_pnl.npz (846 candidates x 32
model-OOS days, hold-to-resolution, produced by audit_strategy_snooping.py
through the shared engine). The v1 TCN trained through 2026-05-11, so all 32
test-split days (05-14..07-02) are model-OOS.

Selection rules evaluated (pre-registered here, all reported):
  top1_score   argmax of pnl/max(1,|maxDD|) with >=8 active days in window
  top1_pnl     argmax of total pnl with >=8 active days
  top3_score   equal-weight portfolio of top-3 by score (diversified variant)

Output: artifacts/audit_v1/walkforward_policy_report.json + .md verdict D5.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "audit_v1"
UNIVERSE = OUT_DIR / "wrc_universe_daily_pnl.npz"
SNOOP = OUT_DIR / "snooping_report.json"

MIN_SELECT_DAYS = 10     # first selection window length
BLOCK = 5                # OOS block length (days)
EMBARGO = 1
MIN_ACTIVE_DAYS_FRAC = 0.25   # candidate must trade on >=25% of window days
B_BOOT = 10_000
SEED = 20260711


def score_matrix(daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(score, active_frac) per candidate over the given day columns.
    score = total_pnl / max(1, |maxDD of daily equity|)."""
    totals = daily.sum(axis=1)
    equity = np.cumsum(daily, axis=1)
    dd = equity - np.maximum.accumulate(equity, axis=1)
    max_dd = np.abs(dd.min(axis=1))
    score = totals / np.maximum(1.0, max_dd)
    active = (np.abs(daily) > 1e-9).mean(axis=1)
    return score, active


def run_rule(universe: np.ndarray, labels: list[str], rule: str) -> dict:
    n_cand, n_days = universe.shape
    picks = []
    oos_daily: list[float] = []
    oos_day_idx: list[int] = []
    i = MIN_SELECT_DAYS
    while i + EMBARGO < n_days:
        window = universe[:, :i]
        test_lo = i + EMBARGO
        test_hi = min(test_lo + BLOCK, n_days)
        score, active = score_matrix(window)
        totals = window.sum(axis=1)
        eligible = active >= MIN_ACTIVE_DAYS_FRAC
        if not eligible.any():
            i = test_hi
            continue
        masked_score = np.where(eligible, score, -np.inf)
        masked_pnl = np.where(eligible, totals, -np.inf)
        if rule == "top1_score":
            sel = [int(masked_score.argmax())]
        elif rule == "top1_pnl":
            sel = [int(masked_pnl.argmax())]
        elif rule == "top3_score":
            sel = list(np.argsort(masked_score)[::-1][:3].astype(int))
        else:
            raise ValueError(rule)
        block_daily = universe[sel, test_lo:test_hi].mean(axis=0)
        oos_daily.extend(block_daily.tolist())
        oos_day_idx.extend(range(test_lo, test_hi))
        picks.append({
            "select_days": i,
            "oos_days": [test_lo, test_hi - 1],
            "selected": [labels[s] for s in sel],
            "selected_window_pnl": [round(float(totals[s]), 2) for s in sel],
            "oos_block_pnl": round(float(universe[sel, test_lo:test_hi].mean(axis=0).sum()), 2),
        })
        i = test_hi
    daily = np.asarray(oos_daily)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, daily.size, size=(B_BOOT, daily.size))
    totals_b = daily[idx].sum(axis=1)
    return {
        "rule": rule,
        "oos_days_total": int(daily.size),
        "oos_total_pnl": round(float(daily.sum()), 2),
        "oos_mean_daily": round(float(daily.mean()), 3),
        "oos_positive_days": int((daily > 0).sum()),
        "oos_total_ci95": [round(float(np.percentile(totals_b, 2.5)), 2),
                           round(float(np.percentile(totals_b, 97.5)), 2)],
        "p_total_leq_0": round(float(np.mean(totals_b <= 0.0)), 4),
        "folds": picks,
    }


def main() -> int:
    z = np.load(UNIVERSE, allow_pickle=False)
    universe = z["daily_pnl"]
    labels = [str(x) for x in z["labels"]]
    days = json.load(open(SNOOP))["days"]
    print(f"universe {universe.shape[0]} candidates x {universe.shape[1]} days "
          f"({days[0]}..{days[-1]})")

    results = {rule: run_rule(universe, labels, rule)
               for rule in ("top1_score", "top1_pnl", "top3_score")}

    # D5 verdict: the process ships only if at least the diversified rule has
    # positive chained OOS with CI excluding 0
    verdict = {}
    for rule, r in results.items():
        verdict[rule] = {
            "oos_positive": r["oos_total_pnl"] > 0,
            "ci_excludes_0": r["oos_total_ci95"][0] > 0,
            "passes_d5": bool(r["oos_total_pnl"] > 0 and r["oos_total_ci95"][0] > 0),
        }
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "universe_file": str(UNIVERSE),
        "days": days,
        "params": {"min_select_days": MIN_SELECT_DAYS, "block": BLOCK,
                   "embargo": EMBARGO, "min_active_frac": MIN_ACTIVE_DAYS_FRAC,
                   "bootstrap": B_BOOT, "seed": SEED},
        "results": results,
        "d5_verdicts": verdict,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "walkforward_policy_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    for rule, r in results.items():
        print(f"{rule}: OOS total {r['oos_total_pnl']:+.2f} over {r['oos_days_total']}d, "
              f"CI95 {r['oos_total_ci95']}, P(<=0)={r['p_total_leq_0']:.3f} "
              f"-> D5 {'PASS' if verdict[rule]['passes_d5'] else 'FAIL'}")
    print(f"-> {OUT_DIR / 'walkforward_policy_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
