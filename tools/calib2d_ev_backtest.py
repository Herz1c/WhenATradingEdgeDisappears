"""2D calibration (TTC x price) + EV-oriented threshold, evaluated from the
polymarket_l2_last60s_v1 parquet dataset (no raw replay needed).

Pipeline
--------
1. Score the v2 residual-head model on each parquet row (vectorized):
       init_z   = logit(clip(implied_p_up))
       raw      = model.predict(X, raw_score=True, num_iteration=best_iter)
       model_p  = sigmoid(init_z + raw)
2. Fit calibrators on CALIB days:
       - cal_global : single isotonic            (the existing baseline)
       - cal2d_iso  : isotonic *per TTC bucket*   (TTC x prob, monotone)
       - cal2d_grid : (TTC bucket x price decile) shrinkage grid
3. Report reliability (brier / logloss / ECE) for each on VAL and TEST.
4. EV-oriented threshold: sweep the per-share edge threshold on VAL, pick the
   value that MAXIMIZES realized PnL (not logloss), then lock it and report on
   TEST. Compare cal_global vs cal2d on the test days.

Backtest rule (matches tools/backtest_poly_l2_only_bucketed_mp.py):
  - <= 1 entry per market per TTC bucket (5 buckets of 10s over (10, 60]).
  - First qualifying tick within a bucket wins.
  - Qualify: pick higher-EV side; require EV/share >= thr, EV/fill >= min_ev_frac,
             fill in [fill_min, fill_max].
  - PnL: shares = notional / fill; payout = shares if side wins else 0;
         pnl = payout - notional.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "datasets" / "polymarket_l2_last60s_v1"
LOGIT_CLIP = (1e-4, 1 - 1e-4)
BUCKET_LABELS = ["(10,20]", "(20,30]", "(30,40]", "(40,50]", "(50,60]"]


# ----- model scoring --------------------------------------------------------

def load_model(artifact_dir: Path):
    model = joblib.load(artifact_dir / "model.pkl")
    feats = json.loads((artifact_dir / "features.json").read_text())
    cal_global = None
    cg = artifact_dir / "calibrator.pkl"
    if cg.exists():
        cal_global = joblib.load(cg)
    elif getattr(model, "_calibrator", None) is not None:
        cal_global = model._calibrator
    return model, feats, cal_global


def score_model_p(model, feats: List[str], df: pl.DataFrame) -> np.ndarray:
    """Vectorized v2 residual-head scoring -> model P(Up)."""
    X = df.select(feats).to_numpy().astype(np.float32)
    imp = df["implied_p_up"].to_numpy().astype(np.float64)
    imp = np.clip(imp, LOGIT_CLIP[0], LOGIT_CLIP[1])
    init_z = np.log(imp / (1.0 - imp))
    uses_init = bool(getattr(model, "_uses_init_score", False))
    if uses_init:
        raw = model.predict(X, raw_score=True, num_iteration=model.best_iteration)
        p = 1.0 / (1.0 + np.exp(-(init_z + raw)))
    else:
        p = model.predict(X, num_iteration=model.best_iteration)
    return np.clip(p, 1e-6, 1 - 1e-6)


def ttc_bucket(ttc: np.ndarray) -> np.ndarray:
    """Map ttc seconds in (10,60] -> bucket id 0..4."""
    b = np.floor(ttc / 10.0).astype(int)        # 10..60 -> 1..6
    b = np.clip(b, 1, 5) - 1                     # -> 0..4
    return b


# ----- calibrators ----------------------------------------------------------

def fit_iso(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def fit_cal2d_iso(p: np.ndarray, y: np.ndarray, b: np.ndarray) -> Dict[int, IsotonicRegression]:
    out: Dict[int, IsotonicRegression] = {}
    for bk in range(5):
        m = b == bk
        if m.sum() < 500:
            continue
        out[bk] = fit_iso(p[m], y[m])
    return out


def apply_cal2d_iso(cal: Dict[int, IsotonicRegression], p: np.ndarray,
                    b: np.ndarray, fallback: IsotonicRegression) -> np.ndarray:
    out = np.empty_like(p)
    for bk in range(5):
        m = b == bk
        if not m.any():
            continue
        iso = cal.get(bk, fallback)
        out[m] = iso.predict(p[m])
    return out


def fit_cal2d_grid(p: np.ndarray, y: np.ndarray, b: np.ndarray,
                   n_price_bins: int = 10, shrink_k: float = 75.0):
    """(TTC bucket x price decile) empirical calibration with Bayesian shrinkage
    toward the cell's mean model prediction. Dense arrays for fast vectorized
    apply. Returns (edges[5, nb+1], table[5, nb] (NaN where empty), nb)."""
    edges = np.tile(np.linspace(0, 1, n_price_bins + 1), (5, 1))
    table = np.full((5, n_price_bins), np.nan)
    for bk in range(5):
        m = b == bk
        if m.sum() < 500:
            continue
        pb = p[m]
        yb = y[m]
        qs = np.quantile(pb, np.linspace(0, 1, n_price_bins + 1))
        qs[0], qs[-1] = 0.0, 1.0
        qs = np.maximum.accumulate(qs)
        edges[bk] = qs
        idx = np.clip(np.searchsorted(qs, pb, side="right") - 1, 0, n_price_bins - 1)
        for j in range(n_price_bins):
            cm = idx == j
            n = int(cm.sum())
            if n == 0:
                continue
            prior = float(pb[cm].mean())
            table[bk, j] = (yb[cm].sum() + shrink_k * prior) / (n + shrink_k)
    return edges, table, n_price_bins


def apply_cal2d_grid(grid, p: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fully vectorized: per row, bin by its bucket's price edges, look up the
    dense table; NaN cells (empty in fit) fall back to the raw model p."""
    edges, table, n_price_bins = grid
    out = np.array(p, dtype=np.float64)
    for bk in range(5):
        m = b == bk
        if not m.any():
            continue
        qs = edges[bk]
        idx = np.clip(np.searchsorted(qs, p[m], side="right") - 1, 0, n_price_bins - 1)
        vals = table[bk, idx]
        miss = np.isnan(vals)
        vals[miss] = p[m][miss]
        out[m] = vals
    return out


# ----- reliability metrics --------------------------------------------------

def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for j in range(n_bins):
        m = idx == j
        if not m.any():
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return {"brier": brier, "logloss": logloss, "ece": float(ece)}


# ----- EV backtest engine (vectorized over rows) ----------------------------

def backtest(df: pl.DataFrame, cal_p: np.ndarray, thr: float,
             notional: float, fill_min: float, fill_max: float,
             min_ev_frac: float) -> pl.DataFrame:
    """Returns one row per (market, bucket) entry with pnl. df must contain
    market_slug, recv_ts_ns, ttc_s, up_best_ask, down_best_ask, label_up."""
    ttc = df["ttc_s"].to_numpy()
    up_ask = df["up_best_ask"].to_numpy().astype(np.float64)
    dn_ask = df["down_best_ask"].to_numpy().astype(np.float64)
    ev_up = np.where((up_ask > 0) & (up_ask < 1), cal_p - up_ask, -1.0)
    ev_dn = np.where((dn_ask > 0) & (dn_ask < 1), (1.0 - cal_p) - dn_ask, -1.0)

    pick_up = ev_up >= ev_dn
    side_ev = np.where(pick_up, ev_up, ev_dn)
    fill = np.where(pick_up, up_ask, dn_ask)
    side = np.where(pick_up, 1, 0)  # 1=UP, 0=DOWN

    safe_fill = np.where(fill > 0, fill, 1.0)
    ev_frac = np.where(fill > 0, side_ev / safe_fill, -1.0)
    # Simple guard rules: no buys below fill_min / above fill_max, and ttc >= 10.
    qual = (side_ev >= thr) & (ev_frac >= min_ev_frac) \
        & (fill >= fill_min) & (fill <= fill_max) \
        & (ttc >= 10.0)

    sel = ["market_slug", "recv_ts_ns", "label_up"]
    if "date" in df.columns:
        sel.append("date")
    work = df.select(sel).with_columns([
        pl.Series("bucket", ttc_bucket(ttc)),
        pl.Series("side", side),
        pl.Series("fill", fill),
        pl.Series("side_ev", side_ev),
        pl.Series("qual", qual),
    ]).filter(pl.col("qual"))

    if work.height == 0:
        return work

    # First qualifying tick per (market, bucket).
    entries = (work.sort("recv_ts_ns")
                   .group_by(["market_slug", "bucket"])
                   .first())

    won = ((entries["side"] == 1) & (entries["label_up"] == 1)) \
        | ((entries["side"] == 0) & (entries["label_up"] == 0))
    shares = notional / entries["fill"].to_numpy()
    pnl = shares * won.to_numpy().astype(np.float64) - notional
    return entries.with_columns([
        pl.Series("won", won.cast(pl.Int8)),
        pl.Series("shares", shares),
        pl.Series("pnl", pnl),
    ])


def bt_summary(entries: pl.DataFrame, notional: float) -> Dict:
    if entries.height == 0:
        return {"n": 0, "pnl": 0.0, "roi": 0.0, "win_rate": float("nan")}
    n = entries.height
    pnl = float(entries["pnl"].sum())
    wins = int(entries["won"].sum())
    return {"n": n, "pnl": pnl, "roi": pnl / (notional * n),
            "win_rate": wins / n,
            "avg_pnl": pnl / n}


# ----- main -----------------------------------------------------------------

def load_days(days: List[str]) -> pl.DataFrame:
    parts = []
    for d in days:
        p = DATASET / f"{d}.parquet"
        if not p.exists():
            print(f"  WARNING: missing {p.name}")
            continue
        parts.append(pl.read_parquet(p))
    if not parts:
        raise SystemExit("no parquet days loaded")
    return pl.concat(parts, how="vertical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default="artifacts_cleaned/poly_l2_only_v2")
    ap.add_argument("--calib-days", default="2026-05-24,2026-05-25,2026-05-26,2026-05-27,2026-05-28,2026-05-29")
    ap.add_argument("--val-days", default="2026-05-30,2026-05-31,2026-06-01,2026-06-02")
    ap.add_argument("--test-days", default="2026-06-10,2026-06-11")
    ap.add_argument("--notional", type=float, default=5.0)
    ap.add_argument("--fill-min", type=float, default=0.10)
    ap.add_argument("--fill-max", type=float, default=0.90)
    ap.add_argument("--min-ev-frac", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    art = (REPO / args.artifact_dir) if not Path(args.artifact_dir).is_absolute() else Path(args.artifact_dir)
    calib_days = [d for d in args.calib_days.split(",") if d]
    val_days = [d for d in args.val_days.split(",") if d]
    test_days = [d for d in args.test_days.split(",") if d]

    print(f"loading model from {art}")
    model, feats, cal_global = load_model(art)

    print(f"loading CALIB {calib_days}")
    df_cal = load_days(calib_days)
    print(f"loading VAL   {val_days}")
    df_val = load_days(val_days)
    print(f"loading TEST  {test_days}")
    df_test = load_days(test_days)
    print(f"rows: calib={df_cal.height:,}  val={df_val.height:,}  test={df_test.height:,}")

    # Score model_p for each split.
    print("scoring model ...")
    p_cal = score_model_p(model, feats, df_cal)
    p_val = score_model_p(model, feats, df_val)
    p_test = score_model_p(model, feats, df_test)
    y_cal = df_cal["label_up"].to_numpy().astype(np.float64)
    y_val = df_val["label_up"].to_numpy().astype(np.float64)
    y_test = df_test["label_up"].to_numpy().astype(np.float64)
    b_cal = ttc_bucket(df_cal["ttc_s"].to_numpy())
    b_val = ttc_bucket(df_val["ttc_s"].to_numpy())
    b_test = ttc_bucket(df_test["ttc_s"].to_numpy())

    # Fit calibrators on CALIB.
    print("fitting calibrators ...")
    iso_global_fit = fit_iso(p_cal, y_cal)          # our own global isotonic
    cal2d_iso = fit_cal2d_iso(p_cal, y_cal, b_cal)
    cal2d_grid = fit_cal2d_grid(p_cal, y_cal, b_cal)

    def make_variants(p, b):
        out = {"raw": np.clip(p, 1e-6, 1 - 1e-6),
               "iso_global": iso_global_fit.predict(p),
               "cal2d_iso": apply_cal2d_iso(cal2d_iso, p, b, iso_global_fit),
               "cal2d_grid": apply_cal2d_grid(cal2d_grid, p, b)}
        if cal_global is not None:
            out["shipped_global"] = np.clip(cal_global.predict(p), 1e-6, 1 - 1e-6)
        return out

    var_val = make_variants(p_val, b_val)
    var_test = make_variants(p_test, b_test)

    # Reliability report.
    print("\n=== RELIABILITY (brier / logloss / ECE) ===")
    print(f"{'variant':<16} {'split':<6} {'brier':>8} {'logloss':>9} {'ece':>8}")
    rel_report = {}
    for split, vars_, y in [("VAL", var_val, y_val), ("TEST", var_test, y_test)]:
        rel_report[split] = {}
        for name, pv in vars_.items():
            r = reliability(pv, y)
            rel_report[split][name] = r
            print(f"{name:<16} {split:<6} {r['brier']:>8.5f} {r['logloss']:>9.5f} {r['ece']:>8.5f}")

    # Per-TTC-bucket reliability on TEST for the two calibrators of interest.
    print("\n=== TEST reliability by TTC bucket (iso_global vs cal2d_iso) ===")
    print(f"{'bucket':<10} {'n':>7} {'ece_glob':>9} {'ece_2d':>9} {'brier_glob':>11} {'brier_2d':>10}")
    bucket_rel = {}
    for bk in range(5):
        m = b_test == bk
        if not m.any():
            continue
        rg = reliability(var_test["iso_global"][m], y_test[m])
        r2 = reliability(var_test["cal2d_iso"][m], y_test[m])
        bucket_rel[BUCKET_LABELS[bk]] = {"n": int(m.sum()), "glob": rg, "cal2d": r2}
        print(f"{BUCKET_LABELS[bk]:<10} {int(m.sum()):>7} {rg['ece']:>9.5f} {r2['ece']:>9.5f} "
              f"{rg['brier']:>11.5f} {r2['brier']:>10.5f}")

    # EV-oriented threshold sweep on VAL (maximize PnL), per calibrator.
    sweep = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    print("\n=== EV-threshold sweep on VAL (objective = PnL, not logloss) ===")
    bt_cols = ["market_slug", "recv_ts_ns", "ttc_s", "up_best_ask", "down_best_ask", "label_up", "date"]
    df_val_bt = df_val.select(bt_cols)
    df_test_bt = df_test.select(bt_cols)

    chosen: Dict[str, float] = {}
    for calname in ["iso_global", "cal2d_iso", "cal2d_grid"]:
        print(f"\n-- calibrator: {calname} --")
        print(f"{'thr':>5} {'n':>6} {'pnl':>9} {'roi%':>7} {'win%':>6}")
        best = (-1e18, None)
        for thr in sweep:
            e = backtest(df_val_bt, var_val[calname], thr, args.notional,
                         args.fill_min, args.fill_max, args.min_ev_frac)
            s = bt_summary(e, args.notional)
            print(f"{thr:>5.2f} {s['n']:>6} {s['pnl']:>+9.2f} "
                  f"{100*s['roi'] if s['n'] else 0:>+6.2f} "
                  f"{100*s['win_rate'] if s['n'] else 0:>5.1f}")
            # Pick max PnL with a minimum trade count guard.
            if s["n"] >= 20 and s["pnl"] > best[0]:
                best = (s["pnl"], thr)
        chosen[calname] = best[1] if best[1] is not None else 0.05
        print(f"  -> chosen EV threshold (max PnL, n>=20): {chosen[calname]}")

    # Lock thresholds, report on TEST.
    print("\n=== TEST results (locked threshold from VAL) ===")
    print(f"{'calibrator':<12} {'thr':>5} {'n':>5} {'pnl':>9} {'roi%':>7} {'win%':>6} {'avg':>8}")
    test_report = {}
    test_entries_for_dump = {}
    for calname in ["iso_global", "cal2d_iso", "cal2d_grid"]:
        thr = chosen[calname]
        e = backtest(df_test_bt, var_test[calname], thr, args.notional,
                     args.fill_min, args.fill_max, args.min_ev_frac)
        s = bt_summary(e, args.notional)
        test_report[calname] = {"thr": thr, **s}
        test_entries_for_dump[calname] = e
        print(f"{calname:<12} {thr:>5.2f} {s['n']:>5} {s['pnl']:>+9.2f} "
              f"{100*s['roi'] if s['n'] else 0:>+6.2f} "
              f"{100*s['win_rate'] if s['n'] else 0:>5.1f} "
              f"{s.get('avg_pnl', 0):>+8.4f}")

    # Per-day + per-bucket breakdown for the best 2D calibrator on TEST.
    print("\n=== TEST per-day x calibrator ===")
    for calname in ["iso_global", "cal2d_iso"]:
        e = test_entries_for_dump[calname]
        if e.height == 0:
            continue
        print(f"\n-- {calname} (thr={chosen[calname]}) --")
        for d in test_days:
            ed = e.filter(pl.col("date") == d)
            if ed.height == 0:
                print(f"  {d}: n=0")
                continue
            pnl = float(ed["pnl"].sum()); n = ed.height; wr = float(ed["won"].mean())
            print(f"  {d}: n={n:>3}  pnl=${pnl:+8.2f}  roi={100*pnl/(args.notional*n):+6.2f}%  wr={100*wr:5.1f}%")

    # Save report.
    out_path = Path(args.out) if args.out else (art / "backtests" /
                f"calib2d_ev_{'_'.join(test_days)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "artifact_dir": str(art),
        "calib_days": calib_days, "val_days": val_days, "test_days": test_days,
        "params": {"notional": args.notional, "fill_min": args.fill_min,
                   "fill_max": args.fill_max, "min_ev_frac": args.min_ev_frac},
        "reliability": rel_report,
        "bucket_reliability_test": bucket_rel,
        "chosen_ev_thresholds": chosen,
        "test_results": test_report,
    }, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
