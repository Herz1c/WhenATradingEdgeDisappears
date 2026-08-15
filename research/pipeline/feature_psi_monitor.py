#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature distribution monitor — flags features whose distribution has shifted
materially between a baseline window and a current window.

Use as a daily watchdog after building new data shards.  Designed to be
runnable from cron; non-zero exit code if any feature exceeds the alert
threshold.

USAGE:
    # Default: compare last 4 days to prior 14 days using tabular dataset
    py -3 feature_psi_monitor.py

    # Explicit windows
    py -3 feature_psi_monitor.py --baseline 2026-04-19:2026-05-06 \\
                                 --current  2026-05-13:2026-05-16 \\
                                 --threshold 0.25

    # JSON output (machine readable)
    py -3 feature_psi_monitor.py --json

EXIT CODES:
    0 - all features within threshold
    2 - one or more features exceed threshold (alert!)
    1 - usage / data error

OUTPUT: docs/feature_psi_monitor_<date>.md (if --report)
"""
from __future__ import annotations
import io, sys, argparse, json
from pathlib import Path
from datetime import date, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import polars as pl
import numpy as np

DATASET = Path("data/datasets/microstructure_sequence_dataset_v1_tabular")

# Default thresholds
THRESH_WARN  = 0.10  # noticeable
THRESH_ALERT = 0.25  # significant (standard PSI convention)
THRESH_CRIT  = 1.00  # catastrophic (what we just saw with BTC features)


def parse_window(s: str) -> list[str]:
    """'2026-05-13:2026-05-16' -> ['2026-05-13', ..., '2026-05-16']"""
    a, b = s.split(":")
    d0 = date.fromisoformat(a)
    d1 = date.fromisoformat(b)
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def default_windows() -> tuple[list[str], list[str]]:
    """Most-recent 4 day shards vs prior 14 day shards available on disk."""
    shards = sorted(DATASET.glob("*.parquet"))
    dates = [s.stem for s in shards if "-" in s.stem]
    if len(dates) < 5:
        raise RuntimeError(f"Need at least 5 daily shards, have {len(dates)}")
    current = dates[-4:]
    baseline = dates[-18:-4] if len(dates) >= 18 else dates[:-4]
    return baseline, current


def load_window(dates: list[str], sample_per_day: int | None = 30_000) -> pl.DataFrame:
    parts = []
    for d in dates:
        p = DATASET / f"{d}.parquet"
        if not p.exists():
            print(f"  WARN missing {d}", file=sys.stderr)
            continue
        df = pl.read_parquet(p)
        if sample_per_day and len(df) > sample_per_day:
            df = df.sample(n=sample_per_day, seed=42)
        parts.append(df)
    return pl.concat(parts, how="diagonal") if parts else pl.DataFrame()


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 100 or len(actual) < 100:
        return float("nan")
    q = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, q))
    if len(edges) < 3:
        return 0.0
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_pct = e_hist / e_hist.sum()
    a_pct = a_hist / max(a_hist.sum(), 1)
    eps = 1e-6
    return float(np.sum((a_pct - e_pct) * np.log((a_pct + eps) / (e_pct + eps))))


def classify(psi_val: float) -> str:
    if not np.isfinite(psi_val):
        return "n/a"
    if psi_val >= THRESH_CRIT:  return "CRITICAL"
    if psi_val >= THRESH_ALERT: return "ALERT"
    if psi_val >= THRESH_WARN:  return "WATCH"
    return "ok"


def compute_drift(baseline: pl.DataFrame, current: pl.DataFrame) -> list[dict]:
    common = [c for c in baseline.columns if c in current.columns]
    out = []
    for c in common:
        try:
            tr = baseline.get_column(c).cast(pl.Float64, strict=False).to_numpy()
            oo = current.get_column(c).cast(pl.Float64, strict=False).to_numpy()
        except Exception:
            continue
        tr = tr.astype(float)
        oo = oo.astype(float)
        ps = psi(tr, oo)
        out.append({
            "feature":     c,
            "baseline_n":  int(np.sum(np.isfinite(tr))),
            "current_n":   int(np.sum(np.isfinite(oo))),
            "baseline_mean": float(np.nanmean(tr)) if np.any(np.isfinite(tr)) else None,
            "current_mean":  float(np.nanmean(oo)) if np.any(np.isfinite(oo)) else None,
            "baseline_std":  float(np.nanstd(tr))  if np.any(np.isfinite(tr)) else None,
            "current_std":   float(np.nanstd(oo))  if np.any(np.isfinite(oo)) else None,
            "psi":           ps,
            "tier":          classify(ps),
        })
    out.sort(key=lambda r: r["psi"] if np.isfinite(r["psi"]) else -1, reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, default=None,
                    help="Date range YYYY-MM-DD:YYYY-MM-DD (default: 14 days before --current)")
    ap.add_argument("--current",  type=str, default=None,
                    help="Date range YYYY-MM-DD:YYYY-MM-DD (default: 4 most-recent shards)")
    ap.add_argument("--threshold", type=float, default=THRESH_ALERT,
                    help=f"Alert threshold (default: {THRESH_ALERT})")
    ap.add_argument("--sample", type=int, default=30_000,
                    help="Sample per day (default: 30000; 0 = all)")
    ap.add_argument("--json", action="store_true", help="Output JSON to stdout")
    ap.add_argument("--report", action="store_true",
                    help="Also write docs/feature_psi_monitor_<today>.md")
    args = ap.parse_args()

    if args.baseline and args.current:
        baseline_dates = parse_window(args.baseline)
        current_dates  = parse_window(args.current)
    else:
        baseline_dates, current_dates = default_windows()

    print(f"Baseline window: {baseline_dates[0]} .. {baseline_dates[-1]} ({len(baseline_dates)} days)", file=sys.stderr)
    print(f"Current window:  {current_dates[0]} .. {current_dates[-1]} ({len(current_dates)} days)", file=sys.stderr)
    print(f"Alert threshold: PSI >= {args.threshold}", file=sys.stderr)

    sample = args.sample if args.sample > 0 else None
    print("Loading baseline...", file=sys.stderr)
    baseline = load_window(baseline_dates, sample)
    print(f"  {len(baseline):,} rows", file=sys.stderr)
    print("Loading current...", file=sys.stderr)
    current  = load_window(current_dates,  sample)
    print(f"  {len(current):,} rows", file=sys.stderr)

    print("Computing PSI per feature...", file=sys.stderr)
    drift = compute_drift(baseline, current)

    alerts = [r for r in drift if np.isfinite(r["psi"]) and r["psi"] >= args.threshold]

    if args.json:
        print(json.dumps({
            "baseline_window": [baseline_dates[0], baseline_dates[-1]],
            "current_window":  [current_dates[0], current_dates[-1]],
            "threshold": args.threshold,
            "n_features": len(drift),
            "n_alerts":   len(alerts),
            "drift":      drift,
        }, indent=2, default=str))
    else:
        print()
        print("=" * 100)
        print(f"FEATURE PSI MONITOR — {len(drift)} features  |  {len(alerts)} above threshold")
        print("=" * 100)
        print(f"{'TIER':<10} {'FEATURE':<40} {'PSI':>8}  baseline μ (σ)            current μ (σ)")
        print("-" * 100)
        for r in drift[:50]:
            tier = r["tier"]
            psi_s = f"{r['psi']:.3f}" if np.isfinite(r["psi"]) else "  n/a "
            bm = f"{r['baseline_mean']:.3g} ({r['baseline_std']:.3g})" if r["baseline_mean"] is not None else "—"
            cm = f"{r['current_mean']:.3g} ({r['current_std']:.3g})" if r["current_mean"] is not None else "—"
            print(f"{tier:<10} {r['feature']:<40} {psi_s:>8}  {bm:<26}{cm}")
        if len(drift) > 50:
            print(f"... +{len(drift)-50} more (use --report or --json for full list)")
        print()

        if alerts:
            print(f"ALERT: {len(alerts)} feature(s) exceed PSI {args.threshold}:")
            for r in alerts:
                print(f"  - {r['feature']:<40} PSI={r['psi']:.3f}  [{r['tier']}]")

    if args.report:
        out = Path(f"docs/feature_psi_monitor_{current_dates[-1]}.md")
        lines = [
            f"# Feature PSI Monitor — {current_dates[-1]}\n\n",
            f"**Baseline window**: {baseline_dates[0]} .. {baseline_dates[-1]} ({len(baseline_dates)} days)\n\n",
            f"**Current window**: {current_dates[0]} .. {current_dates[-1]} ({len(current_dates)} days)\n\n",
            f"**Alert threshold**: PSI >= {args.threshold}\n\n",
            f"**Result**: {len(alerts)} of {len(drift)} features exceed threshold.\n\n",
            "| Tier | Feature | PSI | Baseline μ (σ) | Current μ (σ) |\n",
            "|---|---|---|---|---|\n",
        ]
        for r in drift:
            bm = f"{r['baseline_mean']:.3g} ({r['baseline_std']:.3g})" if r["baseline_mean"] is not None else "—"
            cm = f"{r['current_mean']:.3g} ({r['current_std']:.3g})" if r["current_mean"] is not None else "—"
            psi_s = f"{r['psi']:.3f}" if np.isfinite(r["psi"]) else "n/a"
            lines.append(f"| **{r['tier']}** | `{r['feature']}` | {psi_s} | {bm} | {cm} |\n")
        out.write_text("".join(lines), encoding="utf-8")
        print(f"Report saved to {out}", file=sys.stderr)

    sys.exit(2 if alerts else 0)


if __name__ == "__main__":
    main()
