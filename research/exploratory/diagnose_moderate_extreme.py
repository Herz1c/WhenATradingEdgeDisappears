#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose why 08c_v*/moderate_5c_5s and extreme_10c_5s collapsed/inverted OOS,
while 08c_v*/severe_7c_3s holds up and even improves.

Analysis:
  1. Target base rate drift (train vs OOS) for all 3 targets
  2. Feature distribution drift (KS statistic + PSI) for the 49 features
  3. Per-market OOS AUC distribution for moderate_5c_5s (broken) vs severe_7c_3s (working)
  4. Predicted-probability distribution comparison: did the model output collapse to a constant?
  5. Per-day-of-week / hour-of-day breakdown to spot regime patterns

Inputs:
  Train window: 2026-04-19 .. 2026-05-06  (08c training range; May 6 = test day)
  OOS window:   2026-05-13 .. 2026-05-16

Outputs:
  docs/diagnose_moderate_extreme_report.md
"""

from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json, joblib
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(".") / "src"))
from model_factory.trainers.logistic_regression_trainer import _to_float_array

# Reuse machinery from oos_eval_08bc.py
from oos_eval_08bc import (
    FEAT_49, BTC_FEATS, AGG_FEATS_32,
    add_btc_features, make_X, predict_clf,
)

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_DATES = [f"2026-04-{d:02d}" for d in range(19, 31)] + [f"2026-05-{d:02d}" for d in range(1, 7)]
OOS_DATES   = ["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"]

# Sample size per day for the train load (full OOS is loaded — only 4 days)
TRAIN_SAMPLE_PER_DAY = 20_000

TARGETS = {
    "severe_7c_3s":    (pl.col("mid_move_3s").abs() >= 0.07).cast(pl.Int8),
    "moderate_5c_5s":  (pl.col("mid_move_5s").abs() >= 0.05).cast(pl.Int8),
    "extreme_10c_5s":  (pl.col("mid_move_5s").abs() >= 0.10).cast(pl.Int8),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_one(date: str, sample: int | None = None) -> pl.DataFrame:
    """Load one day of tabular + BTC-enrich."""
    p = Path(f"data/datasets/microstructure_sequence_dataset_v1_tabular/{date}.parquet")
    if not p.exists():
        return pl.DataFrame()
    df = pl.read_parquet(p)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, seed=42)
    df = add_btc_features(df, date)
    return df


def load_window(dates: list[str], sample_per_day: int | None) -> pl.DataFrame:
    parts = []
    for d in dates:
        print(f"  loading {d} ...", end=" ", file=sys.stderr, flush=True)
        df = _load_one(d, sample_per_day)
        print(f"{len(df):,} rows", file=sys.stderr)
        if len(df) > 0:
            parts.append(df)
    return pl.concat(parts, how="diagonal") if parts else pl.DataFrame()


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — >0.25 typically flagged as 'large shift'."""
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 100 or len(actual) < 100:
        return float("nan")
    # Quantile bins from expected
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


def per_market_auc(df: pl.DataFrame, y: np.ndarray, p: np.ndarray, min_n: int = 100) -> pl.DataFrame:
    """Per-market AUC for markets with both classes and >= min_n rows."""
    if "market_slug" not in df.columns:
        return pl.DataFrame()
    tmp = df.select("market_slug").with_columns([
        pl.Series("y", y),
        pl.Series("p", p),
    ])
    rows = []
    for grp, sub in tmp.group_by("market_slug"):
        slug = grp[0] if isinstance(grp, tuple) else grp
        if len(sub) < min_n:
            continue
        y_g = sub["y"].to_numpy().astype(int)
        if len(np.unique(y_g)) < 2:
            continue
        p_g = sub["p"].to_numpy()
        try:
            auc = float(roc_auc_score(y_g, p_g))
        except Exception:
            continue
        rows.append({"market_slug": slug, "n": len(sub), "pos_rate": float(y_g.mean()), "auc": auc})
    return pl.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading TRAINING window (Apr 19 – May 6, 18 days, sampled)...", file=sys.stderr)
    train = load_window(TRAIN_DATES, sample_per_day=TRAIN_SAMPLE_PER_DAY)
    print(f"  Train: {len(train):,} rows total", file=sys.stderr)

    print("Loading OOS window (May 13–16, full)...", file=sys.stderr)
    oos = load_window(OOS_DATES, sample_per_day=None)
    print(f"  OOS:   {len(oos):,} rows total", file=sys.stderr)

    # ── 1. Target base rate drift ─────────────────────────────────────────────
    print("\n[1] Target base-rate drift", file=sys.stderr)
    target_drift = []
    for name, expr in TARGETS.items():
        tr_y = train.select(expr.alias("y")).get_column("y").to_numpy()
        oo_y = oos.select(expr.alias("y")).get_column("y").to_numpy()
        target_drift.append({
            "target":      name,
            "train_base":  float(tr_y.mean()),
            "oos_base":    float(oo_y.mean()),
            "abs_shift":   float(abs(oo_y.mean() - tr_y.mean())),
            "rel_shift":   float((oo_y.mean() - tr_y.mean()) / max(tr_y.mean(), 1e-9)),
        })

    # ── 2. Per-feature distribution drift ─────────────────────────────────────
    print("[2] Per-feature drift (KS + PSI on 49 features)", file=sys.stderr)
    feat_drift = []
    for f in FEAT_49:
        if f not in train.columns or f not in oos.columns:
            continue
        # Cast safely (some features are string/bool/categorical)
        try:
            tr = train.get_column(f).cast(pl.Float64, strict=False).to_numpy()
            oo = oos.get_column(f).cast(pl.Float64, strict=False).to_numpy()
        except Exception:
            continue
        tr = tr.astype(float)
        oo = oo.astype(float)
        tr = tr[np.isfinite(tr)]
        oo = oo[np.isfinite(oo)]
        if len(tr) < 100 or len(oo) < 100:
            continue
        try:
            ks_stat = float(ks_2samp(tr, oo).statistic)
        except Exception:
            ks_stat = float("nan")
        ps = psi(tr, oo)
        feat_drift.append({
            "feature":    f,
            "train_mean": float(np.mean(tr)),
            "oos_mean":   float(np.mean(oo)),
            "train_std":  float(np.std(tr)),
            "oos_std":    float(np.std(oo)),
            "ks":         ks_stat,
            "psi":        ps,
        })
    feat_drift_df = pl.DataFrame(feat_drift).sort("psi", descending=True, nulls_last=True)

    # ── 3. Per-market OOS AUC: broken (moderate) vs working (severe) ─────────
    print("[3] Per-market OOS AUC: severe (working) vs moderate (broken)", file=sys.stderr)
    perm_results: dict[str, pl.DataFrame] = {}
    pred_distributions: dict[str, dict] = {}

    for tname in ["severe_7c_3s", "moderate_5c_5s"]:
        pkl = f"artifacts/model_08c_maker_defense/{tname}/model.pkl"
        if not Path(pkl).exists():
            continue
        model = joblib.load(pkl)
        # Run on OOS for per-market AUC
        oos_y = oos.select(TARGETS[tname].alias("y")).get_column("y").to_numpy()
        X_oos = make_X(oos, FEAT_49)
        p_oos = predict_clf(model, X_oos)
        perm_results[tname] = per_market_auc(oos, oos_y, p_oos)

        # Predicted-probability distribution: train vs OOS
        train_sub = train.sample(n=min(50_000, len(train)), seed=42)
        tr_y = train_sub.select(TARGETS[tname].alias("y")).get_column("y").to_numpy()
        X_tr = make_X(train_sub, FEAT_49)
        p_tr = predict_clf(model, X_tr)
        # Train AUC (sanity — should match test-day-ish)
        tr_auc = float(roc_auc_score(tr_y, p_tr)) if len(np.unique(tr_y)) > 1 else None
        oo_auc = float(roc_auc_score(oos_y, p_oos)) if len(np.unique(oos_y)) > 1 else None
        pred_distributions[tname] = {
            "train_auc":    tr_auc,
            "oos_auc":      oo_auc,
            "train_pred_mean": float(p_tr.mean()),
            "oos_pred_mean":   float(p_oos.mean()),
            "train_pred_std":  float(p_tr.std()),
            "oos_pred_std":    float(p_oos.std()),
            "train_pred_q":   [float(q) for q in np.quantile(p_tr, [0.1, 0.5, 0.9])],
            "oos_pred_q":     [float(q) for q in np.quantile(p_oos, [0.1, 0.5, 0.9])],
        }

    # ── 4. Per-day OOS AUC trend ──────────────────────────────────────────────
    print("[4] Per-day OOS AUC for both targets", file=sys.stderr)
    perday = []
    for tname in ["severe_7c_3s", "moderate_5c_5s", "extreme_10c_5s"]:
        pkl = f"artifacts/model_08c_maker_defense/{tname}/model.pkl"
        if not Path(pkl).exists():
            continue
        model = joblib.load(pkl)
        for d in OOS_DATES:
            sub = _load_one(d)
            y = sub.select(TARGETS[tname].alias("y")).get_column("y").to_numpy()
            if len(np.unique(y)) < 2:
                continue
            X = make_X(sub, FEAT_49)
            p = predict_clf(model, X)
            perday.append({
                "target": tname, "date": d, "n": len(sub),
                "pos_rate": float(y.mean()),
                "auc": float(roc_auc_score(y, p)),
                "pred_mean": float(p.mean()),
            })

    # ── Build report ──────────────────────────────────────────────────────────
    print("\nBuilding report...", file=sys.stderr)
    lines = []
    lines.append("# Diagnosis: Why moderate_5c_5s and extreme_10c_5s collapsed OOS\n")
    lines.append(f"Generated: 2026-05-17\n")
    lines.append(f"Train window: {TRAIN_DATES[0]} .. {TRAIN_DATES[-1]} ({len(train):,} sampled rows)\n")
    lines.append(f"OOS window:   {OOS_DATES[0]} .. {OOS_DATES[-1]} ({len(oos):,} rows, full)\n\n")

    # 1. Target drift
    lines.append("## 1. Target base-rate drift\n")
    lines.append("If the positive class rate shifted dramatically, the model's calibration is wrong before features even matter.\n\n")
    lines.append("| Target | Train base% | OOS base% | Abs shift | Rel shift |\n|---|---|---|---|---|\n")
    for r in target_drift:
        lines.append(f"| `{r['target']}` | {r['train_base']:.2%} | {r['oos_base']:.2%} | {r['abs_shift']:+.2%} | {r['rel_shift']:+.1%} |\n")
    lines.append("\n")

    # 2. Feature drift — top 20 by PSI
    lines.append("## 2. Top-20 features by distribution drift (PSI)\n")
    lines.append("PSI > 0.10 = noticeable, > 0.25 = significant shift.  KS = max gap between empirical CDFs.\n\n")
    lines.append("| # | Feature | Train μ | OOS μ | Train σ | OOS σ | KS | PSI |\n|---|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(feat_drift_df.head(20).to_dicts(), 1):
        lines.append(f"| {i} | `{r['feature']}` | {r['train_mean']:.4g} | {r['oos_mean']:.4g} | {r['train_std']:.4g} | {r['oos_std']:.4g} | {r['ks']:.3f} | **{r['psi']:.3f}** |\n")
    lines.append("\n")

    # 3. Per-day AUC
    lines.append("## 3. Per-day OOS AUC by target\n")
    lines.append("| Target | Date | N | Pos rate | OOS AUC | Pred mean |\n|---|---|---|---|---|---|\n")
    for r in perday:
        lines.append(f"| `{r['target']}` | {r['date']} | {r['n']:,} | {r['pos_rate']:.2%} | {r['auc']:.4f} | {r['pred_mean']:.4f} |\n")
    lines.append("\n")

    # 4. Prediction distribution comparison
    lines.append("## 4. Predicted-probability distribution: train vs OOS\n")
    lines.append("If the model's predictions collapsed to a narrow range OOS, it lost discriminative power.\n\n")
    lines.append("| Target | Train AUC | OOS AUC | Train pred μ | OOS pred μ | Train σ | OOS σ | Train Q10/50/90 | OOS Q10/50/90 |\n|---|---|---|---|---|---|---|---|---|\n")
    for tname, d in pred_distributions.items():
        tr_auc = f"{d['train_auc']:.3f}" if d['train_auc'] else "—"
        oo_auc = f"{d['oos_auc']:.3f}" if d['oos_auc'] else "—"
        lines.append(
            f"| `{tname}` | {tr_auc} | {oo_auc} | {d['train_pred_mean']:.3f} | {d['oos_pred_mean']:.3f} | "
            f"{d['train_pred_std']:.3f} | {d['oos_pred_std']:.3f} | "
            f"{d['train_pred_q'][0]:.3f}/{d['train_pred_q'][1]:.3f}/{d['train_pred_q'][2]:.3f} | "
            f"{d['oos_pred_q'][0]:.3f}/{d['oos_pred_q'][1]:.3f}/{d['oos_pred_q'][2]:.3f} |\n"
        )
    lines.append("\n")

    # 5. Per-market AUC distribution
    lines.append("## 5. Per-market OOS AUC distribution\n")
    lines.append("If AUC degradation is concentrated in a subset of markets, the failure is regime-specific not universal.\n\n")
    for tname, dfm in perm_results.items():
        if len(dfm) == 0:
            continue
        aucs = dfm.get_column("auc").to_numpy()
        lines.append(f"### `{tname}` — {len(dfm)} markets with both classes & n≥100\n\n")
        lines.append(f"  Mean AUC:   {aucs.mean():.4f}\n")
        lines.append(f"  Median:     {np.median(aucs):.4f}\n")
        lines.append(f"  Quartiles:  Q25={np.quantile(aucs, 0.25):.4f}  Q75={np.quantile(aucs, 0.75):.4f}\n")
        lines.append(f"  Range:      min={aucs.min():.4f}  max={aucs.max():.4f}\n")
        lines.append(f"  Markets <0.5 (worse than random): {int(np.sum(aucs < 0.5))} / {len(aucs)} ({np.mean(aucs < 0.5):.1%})\n")
        lines.append(f"  Markets >0.7 (decent signal):     {int(np.sum(aucs > 0.7))} / {len(aucs)} ({np.mean(aucs > 0.7):.1%})\n\n")
        # Histogram in text
        bins = np.linspace(0, 1, 11)
        h, _ = np.histogram(aucs, bins=bins)
        lines.append("  Histogram (AUC bins):\n")
        lines.append("  ```\n")
        for i in range(len(h)):
            bar = "█" * min(50, h[i])
            lines.append(f"    {bins[i]:.1f}–{bins[i+1]:.1f}: {h[i]:5d} {bar}\n")
        lines.append("  ```\n\n")

    # Save
    out = Path("docs/diagnose_moderate_extreme_report.md")
    out.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved to {out}", file=sys.stderr)

    # Also print summary to stdout
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print("\n[1] Target base-rate drift:")
    for r in target_drift:
        print(f"  {r['target']:18}  train={r['train_base']:.2%}  oos={r['oos_base']:.2%}  shift={r['rel_shift']:+.1%}")

    print("\n[2] Top-10 most-shifted features (by PSI):")
    for r in feat_drift_df.head(10).to_dicts():
        print(f"  PSI={r['psi']:.3f}  KS={r['ks']:.3f}  {r['feature']:30}  μ {r['train_mean']:.3g} → {r['oos_mean']:.3g}")

    print("\n[3] Per-day OOS AUC:")
    for r in perday:
        flag = " <- worse than random" if r['auc'] < 0.5 else ""
        print(f"  {r['target']:18} {r['date']}  AUC={r['auc']:.3f}  pos_rate={r['pos_rate']:.2%}  pred_μ={r['pred_mean']:.3f}{flag}")

    print("\n[4] Train vs OOS prediction summary:")
    for tname, d in pred_distributions.items():
        tr = f"{d['train_auc']:.3f}" if d['train_auc'] else "—"
        oo = f"{d['oos_auc']:.3f}" if d['oos_auc'] else "—"
        print(f"  {tname:18}  train_AUC={tr}  oos_AUC={oo}  | pred μ: train={d['train_pred_mean']:.3f} oos={d['oos_pred_mean']:.3f}")

    print("\nFull report -> docs/diagnose_moderate_extreme_report.md")


if __name__ == "__main__":
    main()
