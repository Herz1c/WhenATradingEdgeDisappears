# Quant Audit Findings — All Trained Models
**Date:** 2026-05-12
**Scope:** Models 01, 02, 05, 06, 07, 08, 08b, 08c (v1+v2+v3)
**Approach:** Reload every saved model, evaluate on canonical test day (May 6),
compare to honest baselines, probe for systemic infrastructure issues.

---

## TL;DR

**6 critical weaknesses found**, of which **3 are deployment-blocking** (models
crash or produce wrong answers in production), **3 are methodological**
(reported metrics are misleading).

The trading-grade verdict: **only 3 out of 13 model.pkl artifacts are truly
deployment-ready** as-is: Model 02 dense_close LGBM, Model 05 LGBM, Model 07 LGBM.
Everything else needs at least one fix before going live.

---

## Findings ranked by severity

### CRITICAL #1 — All linear models crash on NaN at inference

**Affected:** Model 01 LR (3 variants), Model 02 LR (coarse), Model 05 LR,
Model 06 Ridge, Model 07 LR, Model 08 LR. **6+ artifacts.**

**Symptom:**
```
joblib.load('artifacts/.../linear_regression/model.pkl').predict_proba(X)
# → ValueError: Input X contains NaN. LogisticRegression does not accept
#   missing values encoded as NaN natively.
```

**Root cause:** `LogisticRegressionTrainer._fit` imputes NaN→0 inside the
training loop but the imputation step is **not part of the saved Pipeline
object**. The bare `LogisticRegression` ends up in `model.pkl`. At inference,
test parquet rows commonly contain NaN (mid_return on day boundary, missing
imbalance during one-sided book, etc.) and load+predict crashes.

**Severity:** DEPLOYMENT-BLOCKING. Cannot use these models in any live system.

**Fix:** Wrap the trainer's sklearn estimator in a `Pipeline([imputer, scaler, lr])`
so the saved artifact contains the full preprocessing chain. (Implemented below.)

---

### CRITICAL #2 — Calibrator damages predictions on tiny-val models

**Affected:** Model 01 preopen LGBM (most severely), Model 01 first15s LGBM.

**Symptom (Model 01 preopen LGBM on test day):**
| Metric | Raw | Calibrated |
|--------|----:|-----------:|
| Log loss | 0.697 | **0.743 (worse!)** |
| AUC | 0.533 | 0.536 |

**Root cause:** Isotonic regression with 14 thresholds fit on only ~284 val
markets overfits to val. On test, the overfit step-function makes log loss
*worse* than the raw probabilities. Compounded with the val-on-val ECE issue
(below), this was completely invisible from the original training reports.

**Severity:** PRODUCTION-WRONG. The deployed model has lower test-set
quality than the un-calibrated model would.

**Fix:** Skip calibration when val_n_positive < 100 OR val_n_total < 1000.
For larger sets, use isotonic with monotonic constraint and bin-min thresholds.
(Implemented below.)

---

### CRITICAL #3 — "Gate PASS" was decided against a strawman baseline

**Affected:** Model 01 (all 6 variants reported PASS), Model 02 (less affected).

**The strawman:** `BaseTrainer._compute_baselines` includes a
`delta_to_strike` baseline computed as `sigmoid(delta * 0.5)`. Since
`delta_to_strike` is in **dollars** (e.g. +$23.5), the sigmoid output is
~0.9999 for almost every row. The resulting log_loss is **1.35** — far
above the no-information constant baseline (0.69). Any model that doesn't
predict 0.99 everywhere beats this baseline trivially.

**Honest re-evaluation of Model 01 (test day = May 6):**

| Variant | LogLoss (cal) | vs `constant_base_rate` (0.691) | Honest verdict |
|---------|--------------:|--------------------------------:|----------------|
| preopen/LGBM | 0.743 | +0.052 | **FAILS** (calibration hurts) |
| preopen/LR | imputed needed | — | likely fail |
| first15s/LGBM | 0.692 | +0.001 | **TIE** (no signal) |
| first15s/LR | imputed needed | — | likely fail |
| **first30s/LGBM** | **0.662** | **-0.029** | **PASS** (4.2% lift) |
| first30s/LR | imputed needed | — | likely fail |

**Only ONE of six Model 01 variants beats the honest baseline.** The
reported "all 6 PASS" was an artifact of the strawman gate.

**Severity:** TRUST-DAMAGING. Reports overstate model quality.

**Fix:** Replace `delta_to_strike` baseline with a properly fitted
logistic regression on `[delta_to_strike]` only, or remove it. Keep
`constant_base_rate` as the primary gate criterion. (Implemented below.)

---

### MAJOR #4 — Val-on-val calibration ECE leakage (cosmetic)

**Affected:** All classifiers (01, 02, 05, 07).

**Symptom:** Training reports show "val ECE = 0.0" for every model. This is
not a sign of perfect calibration — it's the artifact of fitting isotonic
regression on the val set then evaluating ECE on that same val set.

**Severity:** METHODOLOGICAL. Doesn't change deployment behavior but
misrepresents calibration quality. Real calibration is only visible on test.

**Fix:** Either (a) drop val_ECE from reports, (b) split val into
calibrate/evaluate halves, or (c) calibrate on a held-out fraction of train.
(Implemented below.)

---

### MAJOR #5 — Single test day eval is statistically unreliable

**Affected:** All models.

**Symptom:** Every "test" result is from May 6, 2026. One day's worth of
data (288 markets for Model 01, ~270K rows for Models 07/08) is at the mercy
of regime variance. Model 01's test-day failure could be a bad day; Model 02's
dense_close 50% lift could be a lucky day.

**Severity:** STATISTICAL. Cannot make confident claims about generalization
from a single sample.

**Fix:** Roll the test day across the last 7 train days and compute mean ± std.
(Framework provided below.)

---

### MAJOR #6 — Model 02 LightGBM raw probabilities are unusable

**Affected:** Model 02 dense_close LGBM.

**Symptom (test day, raw vs calibrated):**
| Metric | Raw | Calibrated |
|--------|----:|-----------:|
| Log loss | 0.833 | 0.348 |
| AUC | 0.936 | 0.935 |

The calibrator does massive work here. AUC barely changes (ranking is good),
but raw probabilities are wildly miscalibrated — many predictions near 0 or 1
when the true label is the opposite. If the calibrator's pickle is corrupted,
the model becomes useless.

**Severity:** OPERATIONAL. Single point of failure on calibrator artifact.

**Mitigation:** Document that this model's `_calibrator` attribute is REQUIRED
for sensible probabilities. Consider switching LGBM to use
`init_score` adjustment or just exposing the AUC as the ranking score.

---

### MEDIUM #7 — Model 06 target is mislabeled

The target `hold_to_close_edge_vs_mid` is always positive (train mean = +0.36).
Sign accuracy is trivially 99% — every prediction has the right sign because
the only sign is "+". The model is **not a mispricing detector**; it's a
magnitude/edge predictor.

**Fix:** Rename `model_06_mispricing` → `model_06_edge_magnitude` OR redefine
target as signed mispricing relative to a fair-value benchmark.

---

### MEDIUM #8 — Model 08 loses to constant_zero baseline on raw MAE

Already documented in 08c v1/v3 reports. Registry now says CONDITIONAL_PASS
with note. **Verified:** test MAE=0.0139 vs baseline=0.0136. Superseded by
binary classifier 08b and 8c for production use.

---

### MEDIUM #9 — Eligibility filter inconsistency across models

Models use different eligibility columns: `training_feature_eligible` (01, 05, 06),
`sequence_feature_eligible` (07, 08), `is_trainable` (02). This may create
selection bias if filters drop different fractions of "hard" rows.

**Recommendation:** Audit each filter's drop rate and confirm filtered rows
are not systematically different from kept rows.

---

### MEDIUM #10 — Model 07 forced to predict UP/DOWN on rows with no move

Model 07 filters out zero-move rows (61.6% of data) at training. At inference
in production, it has no way to say "no move likely." Pairing with Model 8b
(big-move binary) is necessary to gate the directional prediction.

---

### MINOR #11 — Calibrator overcompensates on Model 05 LR

Model 05 LR raw log_loss=1.57 vs calibrated=0.358. Same pattern as Model 02:
calibrator does heavy lifting. Likely fine because LGBM was the deployed pick,
but the LR artifact is functionally an isotonic regressor over an
under-trained linear input.

---

## Action plan — fixes implemented in this session

### Fix 1: Linear models with proper sklearn Pipeline (NaN imputer included)

See `src/model_factory/trainers/logistic_regression_trainer.py` and
`linear_regression_trainer.py` — wrap in `Pipeline([SimpleImputer, StandardScaler, ...])`.

### Fix 2: Calibration guardrails

`BaseTrainer._calibrate` updated:
- Skip calibration if `len(val) < 1000` or `min(val_positive, val_negative) < 100`.
- When calibrating, use train sub-sample (10% held aside) not val.
- Log a warning when skipped.

### Fix 3: Baseline cleanup

Remove the silly `delta_to_strike` sigmoid baseline. Replace with:
- `logistic_on_delta_only` — fitted on train, eval on val/test.

### Fix 4: Multi-day rolling test eval

New script `eval_models_rolling.py` — for each model, walk forward through
the last 7 days (Apr 30 - May 6), use each as test with prior days as train,
and report mean ± std AUC / log_loss.

---

## Per-model status summary

| Model | Train report PASS | Honest test verdict | Deployment status |
|-------|:-----------------:|---------------------|-------------------|
| 01 preopen LGBM | PASS | **FAIL** | Do not deploy |
| 01 first15s LGBM | PASS | TIE | Marginal; needs more data |
| **01 first30s LGBM** | PASS | **PASS** | **OK (with calibration removed)** |
| 01 LR variants (3) | PASS | unknown | Broken: NaN crash at inference |
| 02 coarse LGBM | PASS | PASS | OK |
| 02 coarse LR | PASS | unknown | Broken: NaN crash at inference |
| **02 dense_close LGBM** | PASS | **PASS** | **OK (calibrator critical!)** |
| **05 dense_close LGBM** | PASS | **PASS** | **OK** |
| 05 dense_close LR | PASS | weak PASS | Broken: NaN crash at inference |
| **06 LGBM** | CONDITIONAL | PASS | OK |
| 06 Ridge | CONDITIONAL | weak PASS | Broken: NaN crash at inference |
| **07 LGBM** | PASS | **PASS** | **OK (cleanest result in project)** |
| 07 LR | PASS | PASS | Broken: NaN crash at inference |
| 08 LGBM | CONDITIONAL | FAIL (loses to zero) | Superseded by 08b/08c |
| 08 LR | CONDITIONAL | FAIL | Superseded |
| **08b LGBM** | PASS | PASS | **OK** |
| **08c v3 extreme_10c_5s agg+event** | PASS | PASS | **OK (deployed)** |

**Deployment-ready (no fix needed): 5 models**
- Model 02 dense_close LGBM
- Model 05 dense_close LGBM
- Model 06 LGBM
- Model 07 LGBM
- Model 08b LGBM
- Model 08c v3 agg+event (3 sub-models)

**Conditionally deployment-ready (after Fix 1 applied): +6 models**
- Model 01 LR variants × 3
- Model 02 coarse LR
- Model 05 LR
- Model 06 Ridge
- Model 07 LR
- Model 08 LR (but superseded anyway)

**Should not deploy:**
- Model 01 preopen LGBM (calibration damages)
- Model 01 first15s LGBM (no signal)
- Model 08 (superseded)

---

## Notes for future work

1. **Add multi-day eval to every training run** (mean ± std AUC/MAE over 7 rolling test days).
2. **Wrap all linear estimators in Pipeline** — never save bare sklearn LR/Ridge.
3. **Remove calibration when val is small** — Model 01 won't be useful until we have 60+ training days.
4. **Drop strawman baselines** from gate decisions — keep only `constant` and one
   non-trivial heuristic per model.
5. **Sanity-check raw vs calibrated** every time — if calibrator does too much
   work, the model is brittle.
6. **Validate model artifacts at save time** — try a `model.predict(...)` round trip
   on a held-out batch with NaN before declaring training complete.
