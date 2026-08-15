"""Train poly_l2_only_v2 — residual-target LightGBM.

Why v2: v1 predicts P(Up) from scratch and "learns" most of its skill by
mimicking the market's implied probability. The Brier skill it reports is
real but lives in the BULK of the distribution. When we actually trade,
we filter to the tails where the model disagrees with the market — exactly
the bins where v1's overconfidence is least controlled.

v2 fix: pass `logit(implied_p_up)` as the LightGBM `init_score`. The model
starts from the market's view and is only ever scored on the *correction*
it adds. The training loss explicitly punishes wrong corrections; agreeing
with the market gives no free win.

Mechanics:
  init_score      = logit(market_p)
  raw_pred        = booster.predict(X, raw_score=True)
  final_p         = sigmoid(init_score + raw_pred)
  (LightGBM applies init_score internally during training and prediction
   ONLY if you pass raw_score=True; otherwise predict returns
   sigmoid(raw_pred) by itself. We have to add init_score back manually.)

The extra rows-per-market bump (1500 -> 3000) helps the model see the
late-tail regime where v1's edge concentrated.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from poly_l2_only.extractor import FEATURE_COLUMNS  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "datasets" / "polymarket_l2_last60s_v1"
ARTIFACT_DIR = REPO_ROOT / "artifacts_cleaned" / "poly_l2_only_v2"

DROP_FEATURES = {"neg_risk", "spread_diff", "tick_size", "mid_sum", "bb_sum", "ba_sum"}
ACTIVE_FEATURES: List[str] = [c for c in FEATURE_COLUMNS if c not in DROP_FEATURES]

ROWS_PER_MARKET_TRAIN = 3000
VAL_MARKET_FRAC = 0.15
RANDOM_SEED = 42
LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=3000,
    feature_fraction=0.80,
    bagging_fraction=0.80,
    bagging_freq=5,
    lambda_l2=2.0,
    max_bin=255,
    verbosity=-1,
    seed=RANDOM_SEED,
)
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 60

PRED_CHUNK_ROWS = 1_000_000
LOGIT_CLIP = (1e-4, 1 - 1e-4)


def ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, LOGIT_CLIP[0], LOGIT_CLIP[1])
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def load_split(split: str, rows_per_market: int):
    print(f"[{ts()}] loading {split} (rows/market={rows_per_market}) ...", flush=True)
    lf = pl.scan_parquet(str(DATA_DIR / "*.parquet")).filter(pl.col("split") == split)
    sampled = (
        lf.with_columns(pl.int_range(pl.len()).over("market_slug").alias("_idx_in_mkt"))
          .filter(pl.col("_idx_in_mkt") < rows_per_market)
          .drop("_idx_in_mkt")
          .collect(engine="streaming")
    )
    print(f"[{ts()}]   {sampled.height:,} rows, {sampled['market_slug'].n_unique():,} markets",
          flush=True)
    X = sampled.select(ACTIVE_FEATURES).to_numpy().astype(np.float32, copy=False)
    y = sampled["label_up"].to_numpy().astype(np.int8, copy=False)
    market_p = sampled["implied_p_up"].to_numpy().astype(np.float32, copy=False)
    slugs = sampled["market_slug"].to_list()
    uniq = {s: i for i, s in enumerate(dict.fromkeys(slugs))}
    g = np.fromiter((uniq[s] for s in slugs), dtype=np.int32, count=len(slugs))
    return X, y, market_p, g, sorted(uniq.keys())


def evaluate(y_true: np.ndarray, p_pred: np.ndarray, p_market: np.ndarray, label: str) -> dict:
    out = {
        "n_rows": int(len(y_true)),
        "label_up_frac": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, p_pred)),
        "pr_auc": float(average_precision_score(y_true, p_pred)),
        "brier": float(brier_score_loss(y_true, p_pred)),
        "logloss": float(log_loss(y_true, np.clip(p_pred, 1e-6, 1 - 1e-6))),
        "market_auc": float(roc_auc_score(y_true, p_market)),
        "market_brier": float(brier_score_loss(y_true, p_market)),
        "market_logloss": float(log_loss(y_true, np.clip(p_market, 1e-6, 1 - 1e-6))),
    }
    out["brier_skill_vs_market"] = (
        1.0 - out["brier"] / out["market_brier"]) if out["market_brier"] > 0 else 0.0
    out["logloss_delta_vs_market"] = out["market_logloss"] - out["logloss"]
    print(f"=== {label} ===")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k:32s} = {v:.6f}")
        else:
            print(f"  {k:32s} = {v}")
    print()
    return out


def calib_table(y_true: np.ndarray, p_pred: np.ndarray, bins: int = 10) -> List[dict]:
    edges = np.quantile(p_pred, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    rows = []
    for i in range(bins):
        m = (p_pred >= edges[i]) & (p_pred < edges[i + 1])
        if not m.any():
            continue
        rows.append({"bin": i, "n": int(m.sum()),
                     "avg_pred": float(p_pred[m].mean()),
                     "actual_freq": float(y_true[m].mean())})
    return rows


def fill_bin_diagnostic(y_true: np.ndarray, p_pred: np.ndarray,
                        p_market: np.ndarray) -> List[dict]:
    """Per fill-price decile (=implied_p_up decile), report model vs market
    calibration. This is the tradeable-bin calibration the v1 backtest
    exposed as broken."""
    edges = np.quantile(p_market, np.linspace(0, 1, 11))
    edges[0] = -np.inf; edges[-1] = np.inf
    rows = []
    for i in range(10):
        m = (p_market >= edges[i]) & (p_market < edges[i + 1])
        if not m.any():
            continue
        # Up-side EV (per share) the model would chase, treating p_market as fill.
        ev_up_avg = float((p_pred[m] - p_market[m]).mean())
        rows.append({
            "bin": i,
            "n": int(m.sum()),
            "market_p_avg": float(p_market[m].mean()),
            "model_p_avg": float(p_pred[m].mean()),
            "actual_p_avg": float(y_true[m].mean()),
            "ev_up_per_share_avg": ev_up_avg,
            "model_overconfidence": float(p_pred[m].mean()) - float(y_true[m].mean()),
        })
    return rows


def chunked_predict(model: lgb.Booster, parquet_glob: str, split: str
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Streaming OOS prediction. Returns (y_true, p_pred_final, p_market)."""
    lf = pl.scan_parquet(parquet_glob).filter(pl.col("split") == split).select(
        ACTIVE_FEATURES + ["label_up"]
    )
    df = lf.collect(engine="streaming")
    n = df.height
    print(f"[{ts()}] OOS predict on {n:,} rows ...", flush=True)
    mk_idx = ACTIVE_FEATURES.index("implied_p_up")
    y_chunks, p_chunks, m_chunks = [], [], []
    for start in range(0, n, PRED_CHUNK_ROWS):
        sl = df.slice(start, PRED_CHUNK_ROWS)
        X = sl.select(ACTIVE_FEATURES).to_numpy().astype(np.float32, copy=False)
        market_p = X[:, mk_idx].astype(np.float32, copy=True)
        init_z = logit(market_p)
        raw = model.predict(X, raw_score=True).astype(np.float32, copy=False)
        p_final = sigmoid(init_z + raw)
        y_chunks.append(sl["label_up"].to_numpy().astype(np.int8))
        m_chunks.append(market_p)
        p_chunks.append(p_final.astype(np.float32))
    return (np.concatenate(y_chunks), np.concatenate(p_chunks), np.concatenate(m_chunks))


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    t_global = time.time()

    # --- 1. Load + split train/val ---
    X_all, y_all, mk_all, g_all, train_slugs = load_split("train", ROWS_PER_MARKET_TRAIN)
    n_markets = len(train_slugs)
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n_markets)
    val_market_set = set(perm[: int(round(n_markets * VAL_MARKET_FRAC))].tolist())
    val_mask = np.fromiter((g in val_market_set for g in g_all), dtype=bool, count=len(g_all))

    X_tr, y_tr, mk_tr = X_all[~val_mask], y_all[~val_mask], mk_all[~val_mask]
    X_va, y_va, mk_va = X_all[val_mask], y_all[val_mask], mk_all[val_mask]
    print(f"[{ts()}] train rows={len(y_tr):,}, val rows={len(y_va):,}", flush=True)
    print(f"[{ts()}] train_label_up={y_tr.mean():.4f}, val_label_up={y_va.mean():.4f}",
          f"market_p mean train={mk_tr.mean():.4f}, val={mk_va.mean():.4f}", flush=True)
    init_score_tr = logit(mk_tr)
    init_score_va = logit(mk_va)

    del X_all, y_all, mk_all, g_all

    # --- 2. Train ---
    dtr = lgb.Dataset(X_tr, label=y_tr, init_score=init_score_tr,
                      feature_name=ACTIVE_FEATURES, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, init_score=init_score_va,
                      feature_name=ACTIVE_FEATURES, reference=dtr, free_raw_data=False)
    print(f"[{ts()}] training v2 (residual head) ...", flush=True)
    t0 = time.time()
    booster = lgb.train(
        LGB_PARAMS,
        dtr,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtr, dva],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(50)],
    )
    train_secs = time.time() - t0
    best_iter = booster.best_iteration
    print(f"[{ts()}] done in {train_secs:.1f}s, best_iter={best_iter}", flush=True)

    # --- 3. Val metrics — both the residual head alone and the final p ---
    raw_va = booster.predict(X_va, raw_score=True).astype(np.float32)
    p_va = sigmoid(init_score_va + raw_va).astype(np.float32)
    val_metrics = evaluate(y_va, p_va, mk_va, "VAL (final p = sigmoid(market_logit + correction))")

    # Magnitude of the correction the model adds:
    corr_abs = np.abs(raw_va)
    print(f"  correction |logit| mean = {corr_abs.mean():.4f}, "
          f"p95 = {np.quantile(corr_abs, 0.95):.4f}, max = {corr_abs.max():.4f}")
    # Fraction of rows where the model meaningfully disagrees with the market:
    disagree = (np.abs(p_va - mk_va) > 0.02).mean()
    print(f"  rows where |p_model - p_market| > 0.02 : {disagree*100:.1f}%")
    print()

    val_calib = calib_table(y_va, p_va)
    val_fill_diag = fill_bin_diagnostic(y_va, p_va, mk_va)

    # Optional isotonic on top (kept off by default — init_score head shouldn't need it).
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_va, y_va)
    p_va_cal = iso.transform(p_va).astype(np.float32)
    val_metrics_cal = evaluate(y_va, p_va_cal, mk_va, "VAL (isotonic on top)")

    del X_tr, y_tr, mk_tr, X_va, y_va, mk_va, init_score_tr, init_score_va, raw_va, dtr, dva

    # --- 4. OOS ---
    y_te, p_te, mk_te = chunked_predict(booster, str(DATA_DIR / "*.parquet"), "test")
    oos_metrics = evaluate(y_te, p_te, mk_te, "OOS (final p)")
    p_te_cal = iso.transform(p_te).astype(np.float32)
    oos_metrics_cal = evaluate(y_te, p_te_cal, mk_te, "OOS (isotonic on top)")
    oos_calib = calib_table(y_te, p_te)
    oos_fill_diag = fill_bin_diagnostic(y_te, p_te, mk_te)

    print("=== OOS FILL-PRICE DECILE DIAGNOSTIC ===")
    print(f"  {'bin':>3} {'n':>9} {'mkt_p':>8} {'model_p':>8} {'actual':>8} {'ev_up/sh':>10} {'overconf':>10}")
    for r in oos_fill_diag:
        print(f"  {r['bin']:>3} {r['n']:>9,} {r['market_p_avg']:>8.4f} {r['model_p_avg']:>8.4f} "
              f"{r['actual_p_avg']:>8.4f} {r['ev_up_per_share_avg']:+10.4f} {r['model_overconfidence']:+10.4f}")
    print()

    # Per-TTC breakdown on OOS
    lf_ttc = (pl.scan_parquet(str(DATA_DIR / "*.parquet"))
              .filter(pl.col("split") == "test").select(["ttc_s"])
              .collect(engine="streaming"))
    ttc = lf_ttc["ttc_s"].to_numpy().astype(np.float32)
    ttc_rows = []
    for lo, hi in [(50, 60), (40, 50), (30, 40), (20, 30), (10, 20)]:
        m = (ttc < lo) & (ttc >= hi)
        if not m.any():
            continue
        bm = float(brier_score_loss(y_te[m], p_te[m]))
        mm = float(brier_score_loss(y_te[m], mk_te[m]))
        ttc_rows.append({"ttc_range": f"[{hi},{lo})", "n": int(m.sum()),
                         "model_brier": bm, "market_brier": mm,
                         "brier_skill_vs_market": 1.0 - bm / mm if mm > 0 else 0.0,
                         "model_auc": float(roc_auc_score(y_te[m], p_te[m])),
                         "market_auc": float(roc_auc_score(y_te[m], mk_te[m]))})
    print("=== OOS BY TTC ===")
    for r in ttc_rows:
        print(f"  {r['ttc_range']}: n={r['n']:>9,}  brier mdl={r['model_brier']:.5f} "
              f"mkt={r['market_brier']:.5f}  skill={r['brier_skill_vs_market']:+.4f}  "
              f"auc mdl={r['model_auc']:.4f} mkt={r['market_auc']:.4f}")
    print()

    # --- 5. Save ---
    booster._calibrator = iso  # type: ignore[attr-defined]
    booster._uses_init_score = True  # type: ignore[attr-defined]
    joblib.dump(booster, ARTIFACT_DIR / "model.pkl")
    joblib.dump(iso, ARTIFACT_DIR / "calibrator.pkl")
    (ARTIFACT_DIR / "features.json").write_text(json.dumps(ACTIVE_FEATURES, indent=2))
    fi_gain = dict(zip(booster.feature_name(), booster.feature_importance("gain").tolist()))
    (ARTIFACT_DIR / "feature_importance.json").write_text(json.dumps({"gain": fi_gain}, indent=2))

    manifest = {
        "version": "poly_l2_only_v2",
        "dataset": "polymarket_l2_last60s_v1",
        "design": "residual head: init_score = logit(implied_p_up); final p = sigmoid(init + correction)",
        "feature_count": len(ACTIVE_FEATURES),
        "dropped_features": sorted(DROP_FEATURES),
        "rows_per_market_train": ROWS_PER_MARKET_TRAIN,
        "val_market_frac": VAL_MARKET_FRAC,
        "random_seed": RANDOM_SEED,
        "lgb_params": LGB_PARAMS,
        "num_boost_round_max": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(best_iter),
        "train_secs": round(train_secs, 1),
        "metrics_val": val_metrics,
        "metrics_val_calibrated": val_metrics_cal,
        "metrics_oos": oos_metrics,
        "metrics_oos_calibrated": oos_metrics_cal,
        "calibration_val": val_calib,
        "calibration_oos": oos_calib,
        "oos_fill_price_decile_diagnostic": oos_fill_diag,
        "oos_by_ttc_bin": ttc_rows,
        "total_secs": round(time.time() - t_global, 1),
    }
    (ARTIFACT_DIR / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{ts()}] saved → {ARTIFACT_DIR}")
    print(f"[{ts()}] total wall time: {time.time() - t_global:.1f}s")


if __name__ == "__main__":
    main()
