"""
Model 7 Confidence Calibration Analysis
========================================
Answers the real question: when the model says "strong UP signal",
is it actually right? And when it says "I don't know", do big moves happen anyway?

Metrics computed:
  1. Accuracy by confidence bucket -- is high confidence actually more accurate?
  2. Mean |actual_move| by confidence bucket -- does the model's uncertainty
     correlate with move magnitude?
  3. High-confidence wrong rate -- how often does a strong signal fail?
  4. Missed big moves -- when model is uncertain, how often does a big move occur?
  5. Expected Calibration Error (granular) -- is the model honest about its confidence?
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import polars as pl
import joblib
from pathlib import Path
from model_factory.trainers.logistic_regression_trainer import _to_float_array
import yaml


# ---------------------------------------------------------------------------
# Load model and test data
# ---------------------------------------------------------------------------

MODEL_PATH = "artifacts/model_07_microstructure_direction/tabular/lightgbm/model.pkl"
TEST_DATE  = "2026-05-06"
DATASET_PATH = Path("data/datasets/microstructure_sequence_dataset_v1_tabular")
FEATURE_SET_PATH = "config/feature_sets/model_07_microstructure_direction_features_v1.yaml"

model = joblib.load(MODEL_PATH)

# Load feature set
with open(FEATURE_SET_PATH, encoding="utf-8") as f:
    fs = yaml.safe_load(f)
feature_cols = fs["feature_columns"]
target_col   = fs["target_columns"]["primary"]  # mid_move_sign_5s

# Load test parquet — include leakage columns for post-hoc analysis
test_file = DATASET_PATH / f"{TEST_DATE}.parquet"
# Load features + target + mid_move_5s (leakage, for move-size analysis)
EXTRA_COLS = [target_col, "mid_move_5s", "abs_delta_to_strike",
              "t_to_close_s", "phase_bucket", "sequence_feature_eligible"]
ANALYSIS_COLS = list(dict.fromkeys(feature_cols + EXTRA_COLS))  # deduplicate, preserve order

schema = pl.scan_parquet(str(test_file)).collect_schema()
available = set(schema.names())
load_cols = [c for c in ANALYSIS_COLS if c in available]

df = pl.read_parquet(str(test_file), columns=load_cols)

# Apply eligibility filter
if "sequence_feature_eligible" in df.columns:
    df = df.filter(pl.col("sequence_feature_eligible"))

# Filter to non-zero moves only (same as training)
y_raw = df[target_col].to_numpy().astype(int)
valid = y_raw != 0
df_v  = df.filter(pl.Series(valid))
y_true = np.where(y_raw[valid] > 0, 1, 0).astype(int)
move_size = df_v["mid_move_5s"].to_numpy().astype(float) if "mid_move_5s" in df_v.columns else None
abs_delta = df_v["abs_delta_to_strike"].fill_null(0.0).to_numpy().astype(float) if "abs_delta_to_strike" in df_v.columns else None

# Get predictions
X_feat = df_v.select(feature_cols)
X_arr, _ = _to_float_array(X_feat)
raw_prob  = model.predict_proba(X_arr)[:, 1]
calibrator = getattr(model, "_calibrator", None)
y_prob = calibrator.predict(raw_prob) if calibrator else raw_prob
y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)

n_total = len(y_true)
overall_acc = float(np.mean((y_prob >= 0.5).astype(int) == y_true))


def sep(title=""):
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# 1. Accuracy by confidence bucket
# ---------------------------------------------------------------------------
sep("1. ACCURACY BY CONFIDENCE BUCKET")
print(f"  n={n_total:,}  overall_acc={overall_acc:.4f}")
print()

# Symmetric buckets: distance from 0.5
# Strong signal = high distance from 0.5 (either very high or very low)
# Convert to "confidence in predicted direction": max(p, 1-p)
confidence = np.maximum(y_prob, 1 - y_prob)

buckets = [
    (0.50, 0.55, "0.50-0.55  (no signal)"),
    (0.55, 0.60, "0.55-0.60  (weak)     "),
    (0.60, 0.65, "0.60-0.65  (moderate) "),
    (0.65, 0.70, "0.65-0.70  (good)     "),
    (0.70, 0.75, "0.70-0.75  (strong)   "),
    (0.75, 0.80, "0.75-0.80  (very str.)"),
    (0.80, 1.01, "0.80+      (max conf.)"),
]

print(f"  {'Confidence':25s}  {'n':>7s}  {'pct':>6s}  {'accuracy':>9s}  {'mean|move|':>10s}  {'wrong_rate':>10s}")
print(f"  {'-'*25}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*10}")
for lo, hi, label in buckets:
    mask = (confidence >= lo) & (confidence < hi)
    if not mask.any():
        continue
    n_b   = int(mask.sum())
    acc_b = float(np.mean((y_prob[mask] >= 0.5).astype(int) == y_true[mask]))
    wrong = 1.0 - acc_b
    pct   = 100.0 * n_b / n_total
    mean_move = float(np.mean(np.abs(move_size[mask]))) if move_size is not None else float("nan")
    print(f"  {label}  {n_b:>7,}  {pct:>5.1f}%  {acc_b:>8.4f}   {mean_move:>10.5f}   {wrong:>9.4f}")


# ---------------------------------------------------------------------------
# 2. High-confidence wrong rate (the dangerous cases)
# ---------------------------------------------------------------------------
sep("2. HIGH-CONFIDENCE WRONG PREDICTIONS")
print("  When model is VERY sure but wrong -- these are the dangerous misfires.")
print()
for thresh, label in [(0.65, "conf > 0.65"), (0.70, "conf > 0.70"), (0.75, "conf > 0.75"), (0.80, "conf > 0.80")]:
    mask = confidence >= thresh
    if not mask.any():
        continue
    n_b   = int(mask.sum())
    wrong_mask = mask & ((y_prob >= 0.5).astype(int) != y_true)
    n_wrong = int(wrong_mask.sum())
    wrong_rate = n_wrong / n_b
    mean_move_wrong = float(np.mean(np.abs(move_size[wrong_mask]))) if move_size is not None and wrong_mask.any() else float("nan")
    mean_move_all   = float(np.mean(np.abs(move_size[mask]))) if move_size is not None else float("nan")
    print(f"  {label}: n={n_b:,}  wrong={n_wrong:,} ({wrong_rate:.1%})  "
          f"mean|move| when wrong={mean_move_wrong:.5f}  mean|move| all={mean_move_all:.5f}")


# ---------------------------------------------------------------------------
# 3. Missed large moves (uncertain model, big move happens)
# ---------------------------------------------------------------------------
sep("3. MISSED LARGE MOVES (uncertain model, big move occurs)")
print("  When model says 'I don't know' (confidence < 0.55) but a big move happens.")
print()
if move_size is not None:
    uncertain = confidence < 0.55
    n_uncertain = int(uncertain.sum())
    for move_thresh, label in [(0.02, "|move| > 0.02"), (0.03, "|move| > 0.03"), (0.05, "|move| > 0.05")]:
        big_move = np.abs(move_size) >= move_thresh
        missed = uncertain & big_move
        n_missed = int(missed.sum())
        n_big = int(big_move.sum())
        pct_of_uncertain = 100 * n_missed / n_uncertain if n_uncertain > 0 else 0
        pct_of_big = 100 * n_missed / n_big if n_big > 0 else 0
        print(f"  {label}: n_big_moves={n_big:,}  "
              f"missed_when_uncertain={n_missed:,} ({pct_of_big:.1f}% of big moves)  "
              f"pct_of_uncertain={pct_of_uncertain:.1f}%")


# ---------------------------------------------------------------------------
# 4. Expected Calibration Error -- granular
# ---------------------------------------------------------------------------
sep("4. CALIBRATION: does P=0.70 mean 70% accuracy?")
print("  If model says 70% UP -- does it happen 70% of the time?")
print()
cal_buckets = [(i/20, (i+1)/20) for i in range(10, 21)]  # 0.50 to 1.05 in 0.05 steps
print(f"  {'Pred prob range':20s}  {'n':>7s}  {'mean_pred':>9s}  {'actual_acc':>10s}  {'gap (cal err)':>13s}")
print(f"  {'-'*20}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*13}")
for lo, hi in cal_buckets:
    # Use raw probabilities (both sides: near-0 and near-1 mapped to symmetric confidence)
    mask_hi = (y_prob >= lo) & (y_prob < hi)
    mask_lo = (y_prob >= (1-hi)) & (y_prob < (1-lo))
    mask = mask_hi | mask_lo
    if not mask.any():
        continue
    n_b = int(mask.sum())
    mean_pred = float(np.mean(np.where(y_prob[mask] >= 0.5, y_prob[mask], 1 - y_prob[mask])))
    act_acc   = float(np.mean((y_prob[mask] >= 0.5).astype(int) == y_true[mask]))
    gap = act_acc - mean_pred
    print(f"  {lo:.2f}-{hi:.2f} (conf {max(lo,1-hi):.2f}-{max(hi,1-lo):.2f})  "
          f"{n_b:>7,}  {mean_pred:>9.4f}  {act_acc:>10.4f}  {gap:>+12.4f}")


# ---------------------------------------------------------------------------
# 5. Summary verdict
# ---------------------------------------------------------------------------
sep("5. VERDICT")
high_conf_mask = confidence >= 0.70
if high_conf_mask.any():
    hc_acc = float(np.mean((y_prob[high_conf_mask] >= 0.5).astype(int) == y_true[high_conf_mask]))
    hc_n   = int(high_conf_mask.sum())
    hc_pct = 100 * hc_n / n_total
    print(f"  High-confidence predictions (conf>=0.70): {hc_n:,} rows ({hc_pct:.1f}% of moves)")
    print(f"  Accuracy on those: {hc_acc:.4f}  (wrong {1-hc_acc:.4f})")

low_conf_mask = confidence < 0.55
if low_conf_mask.any() and move_size is not None:
    lc_mean_move = float(np.mean(np.abs(move_size[low_conf_mask])))
    hc_mean_move = float(np.mean(np.abs(move_size[high_conf_mask]))) if high_conf_mask.any() else float("nan")
    print(f"  Mean |move| when model uncertain (conf<0.55): {lc_mean_move:.5f}")
    print(f"  Mean |move| when model confident (conf>=0.70): {hc_mean_move:.5f}")
    print()
    if hc_mean_move > lc_mean_move:
        print("  GOOD: model is MORE confident when moves are LARGER")
    else:
        print("  WARNING: model confidence does NOT correlate with move size")
