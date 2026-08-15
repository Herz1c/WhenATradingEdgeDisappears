# Research scripts (earlier phases)

These are the scripts from the supervised-modelling phases, kept for
provenance rather than reuse. They ran from the repository root and expect
datasets that are not distributed (see `docs/DATA_AVAILABILITY.md`), so treat
them as a record of what was done, not as a maintained interface.

85 scripts in total.

The maintained tooling lives in [`tools/`](../tools/README.md).

## Model training (`training/`, 19)

Per-model training entry points from the supervised phase (Models 01-08c). `*_cleaned.py` variants retrain on the leakage-audited feature set.

| Script | Purpose |
|---|---|
| `ablation_model_08b.py` | Ablation: Model 8b WITH vs WITHOUT BTC volatility features. |
| `retrain_all_cleaned.py` | Retrain orchestrator — produces and runs cleaned-feature variants of every |
| `train_adverse_selection.py` | Model 04 Adverse Selection / Markout: Ridge baseline + LightGBM regression. |
| `train_adverse_selection_to_close.py` | Model 04b: Adverse Selection / Markout-to-CLOSE regression. |
| `train_bias.py` | Run Model 1 Bias training: LR baseline + LightGBM on all three dataset variants. |
| `train_closing_flip.py` | Model 5 Closing Flip: LR baseline + LightGBM on dense_close. |
| `train_fair_resolution.py` | Model 2 Fair-Resolution: LR baseline + LightGBM on coarse and dense_close. |
| `train_fill_probability.py` | Model 03 Fill Probability: LR baseline + LightGBM primary. |
| `train_maker_defense.py` | Maker-Defense Classifier (Model 8c): dramatic-move early-warning. |
| `train_maker_defense_cleaned.py` | Maker-Defense Classifier (Model 8c): dramatic-move early-warning. |
| `train_maker_defense_v2.py` | Maker-Defense v2: adds vol-clustering and recent-jump features. |
| `train_maker_defense_v2_cleaned.py` | Maker-Defense v2: adds vol-clustering and recent-jump features. |
| `train_maker_defense_v3_eventtime.py` | Maker-Defense v3: Event-time microstructure features from event64 dataset. |
| `train_maker_defense_v3_eventtime_cleaned.py` | Maker-Defense v3: Event-time microstructure features from event64 dataset. |
| `train_microstructure_direction.py` | Model 7 Microstructure Direction: LR baseline + LightGBM on tabular variant. |
| `train_mispricing.py` | Model 6 Mispricing: Ridge regression baseline + LightGBM on dense_close. |
| `train_move_size.py` | Model 8 Move Size: Linear Regression baseline + LightGBM on tabular variant. |
| `train_move_size_v2.py` | Model 8b: Binary big-move classifier with BTC volatility features. |
| `train_move_size_v2_cleaned.py` | Model 8b: Binary big-move classifier with BTC volatility features. |

## Evaluation and audits (`evaluation/`, 16)

Out-of-sample evaluations, leakage/trojan audits, and phase-level analyses.

| Script | Purpose |
|---|---|
| `analyze_model_07_confidence.py` | Model 7 Confidence Calibration Analysis |
| `analyze_model_08.py` | Model 8 Move Size -- Post-hoc Analysis |
| `analyze_phase6_held_to_close_pnl.py` | Phase 6 DEFINITIVE viability test — actual held-to-close PnL. |
| `analyze_phase6_maker_first_viability.py` | Phase 6 Strategic Analysis: Is Maker-First Viable? |
| `analyze_phase6_pnl_with_model4b.py` | Phase 6 — Re-rank top-K analysis using Model 4b (markout_to_close) |
| `audit_all_models.py` | Comprehensive quant audit of all trained models. |
| `compare_cleaned_vs_baseline.py` | Compare cleaned-feature retrains (artifacts_cleaned/) against the contaminated |
| `make_cleaned_oos_evals.py` | Generate _cleaned.py variants of the OOS eval scripts pointing at artifacts_cleaned/. |
| `oos_eval_08bc.py` | OOS evaluation for model_08b and all model_08c variants on 2026-05-13..16. |
| `oos_eval_08bc_cleaned.py` | OOS evaluation for model_08b and all model_08c variants on 2026-05-13..16. |
| `oos_eval_event_only.py` | OOS evaluation for the trojan-immune 08c v3 event_only and agg_plus_event variants. |
| `oos_eval_may13_16.py` | Out-of-sample evaluation: all trained models on 2026-05-13 through 2026-05-16. |
| `oos_eval_may13_16_cleaned.py` | Out-of-sample evaluation: all trained models on 2026-05-13 through 2026-05-16. |
| `oos_eval_zero_trojan.py` | Zero-trojan OOS re-evaluation. |
| `rolling_eval_phase6.py` | Phase 6 — Multi-day rolling evaluation of maker-first strategy. |
| `trojan_audit.py` | Trojan-horse feature audit across all 33 model artifacts. |

## Exploratory analysis (`exploratory/`, 42)

Single-question scripts written to settle one point and kept for provenance. These are working notes in code form, not maintained interfaces.

| Script | Purpose |
|---|---|
| `debug_audit_part2.py` | Deeper audit: stale-quote arbitrage check + latency stress + market efficiency check. |
| `debug_audit_strategy.py` | Honest audit of the Model 02 DOWN taker strategy. |
| `debug_audit_sub_second.py` | Sub-second latency analysis using bid LIFETIME distribution. |
| `debug_brutal_audit.py` | BRUTAL AUDIT — Model 02 DOWN-side taker strategy. |
| `debug_calibrator_isolation.py` | Isolate the effect of the calibrator on each model. |
| `debug_directional_maker.py` | Validate the directional-maker config: maker quotes gated on Model 04b sign. |
| `debug_edge_decay_check.py` | Two questions to test the 'retail-inefficiency hypothesis': |
| `debug_extreme_configs.py` | Test extreme configs to find any profitable parameterization. |
| `debug_fee_stress.py` | Fee stress test: re-grade winner under 1.0x, 1.2x, 1.5x, and 2.0x fee scaling. |
| `debug_hour_x_price.py` | Joint distribution: hour-of-day × entry-price-band → win rate. |
| `debug_intent_engine.py` | Validate the ExecutionIntentEngine with hand-tuned filter configs. |
| `debug_model02_2x10.py` | Compare: 1 entry/market vs 2 entries/market with min 10s gap, otherwise identical. |
| `debug_model02_accuracy.py` | Model 02 accuracy metrics on OOS resolution_dense_close. |
| `debug_model02_dedup.py` | Re-grade Model 02 DOWN strategy with per-market position dedup. |
| `debug_model02_strategy.py` | Model 02-only trading strategy on resolution_dense_close. |
| `debug_model02_taker.py` | Direct test: how profitable is buying when model 02 fair_value > mid + threshold? |
| `debug_model02_taker_explore.py` | Exhaustive exploration of Model 02 as a pure taker signal. |
| `debug_model02_taker_grade.py` | Official 20-MC-week grade of the Model 02 taker strategy with proper lifecycle. |
| `debug_model02_taker_stress.py` | Stress tests for Model 02 DOWN-side taker strategy. |
| `debug_model02_threshold_pick.py` | Pick the deployment threshold for the new Model 02 DOWN-side taker. |
| `debug_model02_threshold_pick_1usd.py` | Same threshold sweep as debug_model02_threshold_pick.py BUT with |
| `debug_model02_threshold_pick_old.py` | SAME threshold sweep as debug_model02_threshold_pick.py but with the |
| `debug_model04b_taker.py` | Test model 04b (markout-to-close) as a direct taker signal. |
| `debug_new_model_raw_vs_cal.py` | Compare new Model 02 (raw vs calibrated) vs the archived old Model 02 |
| `debug_new_oos_check.py` | Quick Model 02 DOWN strategy sanity check on new May 17-20 OOS data. |
| `debug_official_grade.py` | Official Calmar-style grade of the hand-tuned winning config across 20 MC weeks. |
| `debug_per_model_edge.py` | Per-model honest-edge scorecard. |
| `debug_perf_single_week.py` | Time a single week of backtest after precompute optimization. |
| `debug_plot_2x10.py` | Plot: 1 entry/market vs 2 entries/market (10s gap), side by side. |
| `debug_plot_mc_weeks.py` | Plot the 20 MC OOS weeks for the Model 02 DOWN strategy. |
| `debug_pure_oos.py` | Pure OOS validation: grade winner ONLY on days outside training window. |
| `debug_quick_oos.py` | Fast in-sample vs OOS comparison. |
| `debug_regime_check.py` | Regime check — was the OOS window a single BTC regime? |
| `debug_sensitivity_sweep.py` | Sensitivity sweep ±20% on each parameter of the winning config. |
| `debug_signed_pred_band.py` | Test the signed_pred band (0.05, 0.10] strategy across multiple weeks. |
| `debug_signed_pred_realistic.py` | Realistic test of signed_pred band strategy with proper margin model. |
| `debug_sizing_2x10s_max2_50.py` | Scheme C (linear-in-edge sizing) with: |
| `debug_sizing_2x10s_max3.py` | Scheme C (linear-in-edge sizing) BUT with: |
| `debug_sizing_schemes.py` | Compare position-sizing schemes on the recommended v2 model + p65 threshold. |
| `debug_strategy_fills.py` | Debug: load 200 anchors from a single day and trace why no fills happen. |
| `debug_two_models_apples.py` | Apples-to-apples head-to-head on the SAME OOS data (May 15-20): |
| `diagnose_moderate_extreme.py` | Diagnose why 08c_v*/moderate_5c_5s and extreme_10c_5s collapsed/inverted OOS, |

## Pipeline utilities (`pipeline/`, 8)

Feature builders, monitors and orchestration helpers from the same phase.

| Script | Purpose |
|---|---|
| `backtest_pnl.py` | End-to-end P&L backtest of the hybrid maker/taker strategy. |
| `build_btc_tick_features.py` | Build sub-second BTC tick features for each microstructure anchor. |
| `feature_psi_monitor.py` | Feature distribution monitor — flags features whose distribution has shifted |
| `rebuild_manifest.py` | Rebuild a corrupted .ws.manifest.json from the raw .jsonl.zst file. |
| `run_intent_search.py` | Optuna search over IntentFilter parameters using the execution_intent engine. |
| `run_strategy_search.py` | Quiet-mode strategy search driver. |
| `run_with_oom_fix.py` | Wrapper: set OOM-safe env vars + cleanup env vars, then exec a training script. |
| `sizing_analysis_phase6.py` | Phase 6 — Small-money sizing analysis. |
