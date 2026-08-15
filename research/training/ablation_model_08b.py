"""Ablation: Model 8b WITH vs WITHOUT BTC volatility features.

Trains the same architecture on:
  A) Token L2 features only            (49 - 6 = 43 features)
  B) Token L2 + BTC vol features       (49 features, same as Model 8b)

Compares AUC / Average Precision / recall@precision to quantify BTC contribution.
"""
import sys, json, os
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

from model_factory.trainers.logistic_regression_trainer import _to_float_array
from train_move_size_v2 import (
    load_split, BASE_FEATURES, DERIVED_FEATURES, ALL_FEATURES,
    TRAIN_DATES, VAL_DATES, TEST_DATES, BIG_MOVE_THRESHOLD,
)


def fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, label, n_estimators=600):
    model = lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss",
        num_leaves=63, min_child_samples=500, learning_rate=0.05,
        n_estimators=n_estimators, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])

    proba_va = model.predict_proba(X_va)[:, 1]
    proba_te = model.predict_proba(X_te)[:, 1]

    metrics = {
        "label": label,
        "n_features": X_tr.shape[1],
        "best_iter": int(model.best_iteration_ or 0),
        "val_auc":  float(roc_auc_score(y_va, proba_va)),
        "val_ap":   float(average_precision_score(y_va, proba_va)),
        "test_auc": float(roc_auc_score(y_te, proba_te)),
        "test_ap":  float(average_precision_score(y_te, proba_te)),
    }

    # Recall at fixed precision (0.50 and 0.60)
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_te, proba_te)
    for target_prec in [0.50, 0.55, 0.60]:
        # find recall at the lowest threshold where precision >= target
        mask = prec >= target_prec
        if mask.any():
            metrics[f"recall_at_prec{int(target_prec*100)}"] = float(rec[mask].max())
        else:
            metrics[f"recall_at_prec{int(target_prec*100)}"] = 0.0

    # Recall at fixed selection percent
    for top_pct in [0.10, 0.20, 0.30]:
        k = int(top_pct * len(proba_te))
        idx = np.argsort(-proba_te)[:k]
        captured = int(y_te[idx].sum())
        total_pos = int(y_te.sum())
        metrics[f"recall_top{int(top_pct*100):02d}pct"] = float(captured / max(total_pos, 1))
        metrics[f"precision_top{int(top_pct*100):02d}pct"] = float(captured / max(k, 1))

    return metrics, proba_te


if __name__ == "__main__":
    print("\n" + "#"*72)
    print("  ABLATION: Model 8b WITH vs WITHOUT BTC vol features")
    print("#"*72)

    print("\n--- Loading data ---")
    tr = load_split(TRAIN_DATES, "train")
    va = load_split(VAL_DATES,   "val")
    te = load_split(TEST_DATES,  "test")

    y_tr = tr["big_move"].to_numpy().astype(int)
    y_va = va["big_move"].to_numpy().astype(int)
    y_te = te["big_move"].to_numpy().astype(int)

    print("\n--- A) Token L2 only (no BTC vol features) ---")
    X_tr_a, _ = _to_float_array(tr.select(BASE_FEATURES))
    X_va_a, _ = _to_float_array(va.select(BASE_FEATURES))
    X_te_a, _ = _to_float_array(te.select(BASE_FEATURES))
    print(f"  n_features={X_tr_a.shape[1]}")
    m_a, _ = fit_eval(X_tr_a, y_tr, X_va_a, y_va, X_te_a, y_te, "L2_only")

    print("\n--- B) Token L2 + BTC vol features ---")
    X_tr_b, _ = _to_float_array(tr.select(ALL_FEATURES))
    X_va_b, _ = _to_float_array(va.select(ALL_FEATURES))
    X_te_b, _ = _to_float_array(te.select(ALL_FEATURES))
    print(f"  n_features={X_tr_b.shape[1]}")
    m_b, _ = fit_eval(X_tr_b, y_tr, X_va_b, y_va, X_te_b, y_te, "L2_plus_BTC")

    print("\n\n" + "="*72)
    print("  ABLATION RESULTS  |  positive_rate (test) = {:.4f}".format(y_te.mean()))
    print("="*72)
    print(f"  {'Metric':<28s}  {'L2 only':>12s}  {'L2 + BTC':>12s}  {'delta':>10s}")
    print(f"  {'-'*28}  {'-'*12}  {'-'*12}  {'-'*10}")

    def row(name, key, fmt=".4f"):
        a, b = m_a[key], m_b[key]
        d = b - a
        print(f"  {name:<28s}  {a:>12{fmt}}  {b:>12{fmt}}  {d:>+10{fmt}}")

    row("n_features",                 "n_features", "d")
    row("best_iter",                  "best_iter",  "d")
    row("val AUC",                    "val_auc")
    row("val Avg Precision",          "val_ap")
    row("test AUC",                   "test_auc")
    row("test Avg Precision",         "test_ap")
    row("test recall@prec=0.50",      "recall_at_prec50")
    row("test recall@prec=0.55",      "recall_at_prec55")
    row("test recall@prec=0.60",      "recall_at_prec60")
    row("test recall in top 10%",     "recall_top10pct")
    row("test recall in top 20%",     "recall_top20pct")
    row("test recall in top 30%",     "recall_top30pct")
    row("test precision in top 10%",  "precision_top10pct")
    row("test precision in top 20%",  "precision_top20pct")
    row("test precision in top 30%",  "precision_top30pct")

    print()
    # Verdict
    delta_ap = m_b["test_ap"] - m_a["test_ap"]
    delta_auc = m_b["test_auc"] - m_a["test_auc"]
    if delta_ap > 0.005 or delta_auc > 0.003:
        print(f"  VERDICT: BTC features HELP  (dAP={delta_ap:+.4f}, dAUC={delta_auc:+.4f})")
    else:
        print(f"  VERDICT: BTC features MARGINAL  (dAP={delta_ap:+.4f}, dAUC={delta_auc:+.4f})")

    out = Path("artifacts/model_08b_big_move_classifier/ablation_btc.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"L2_only": m_a, "L2_plus_BTC": m_b}, f, indent=2)
    print(f"\n  saved to {out}")
