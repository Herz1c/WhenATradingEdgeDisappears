# Trojan-Horse Feature Audit
Generated: 2026-05-17

**Hypothesis under test**: Are there other models (beyond the already-broken `moderate_5c_5s` / `extreme_10c_5s`) that over-rely on features with catastrophic distribution drift between training and OOS?

**Trojan-HARD features** (PSI > 1.0 in train vs OOS):
- `live_applied_bias_age_seconds`
- `live_btc_usd`
- `price_to_beat`

**Trojan-SOFT features** (PSI 0.05–0.25 — watch for over-reliance):
- `absolute_trade_size_3s`, `absolute_trade_size_5s`, `depth_change_rate_5s`, `event_count_15s`, `event_count_1s`, `event_count_3s`, `event_count_5s`, `quote_churn_rate_5s`, `quote_update_count_1s`, `quote_update_count_3s`, `quote_update_count_5s`, `signed_trade_size_3s`, `signed_trade_size_5s`, `spread_max_5s`, `spread_mean_5s`, `spread_min_5s`, `trade_count_1s`, `trade_count_3s`, `trade_count_5s`

**Risk tiers** (based on share of total importance on HARD features):
- **CRITICAL**: ≥30% on HARD features OR any HARD feature in top 5 → expect failure on regime shift
- **HIGH**:     15–30% → vulnerable
- **MODERATE**: 5–15% → monitor
- **LOW**:      <5% → robust

## Ranked Vulnerability

| Model | N feat | HARD% | SOFT% | Risk | Top trojan features | Top-5 features overall |
|---|---|---|---|---|---|---|
| `model_04b_markout_to_close/lightgbm` | 50 | **21.8%** | 9.9% | **CRITICAL** | `price_to_beat` (10%), `live_btc_usd` (6%), `live_applied_bias_age_seconds` (6%) | `price_to_beat` (10%), `abs_delta_to_strike` (7%), `price_level` (6%), `live_btc_usd` (6%), `state_ask_depth_total` (6%) |
| `model_02_fair_resolution/coarse/lightgbm` | 74 | **21.5%** | 0.0% | **CRITICAL** | `price_to_beat` (15%), `live_applied_bias_age_seconds` (4%), `live_btc_usd` (2%) | `price_to_beat` (15%), `snapshot_minute_utc` (13%), `snapshot_second_of_day_utc` (12%), `live_bias_observation_count` (8%), `binance_usdm_mid` (5%) |
| `model_08c_maker_defense_v3_eventtime/moderate_5c_5s/agg_only` | 32 | **20.6%** | 28.7% | **CRITICAL** | `price_to_beat` (7%), `live_applied_bias_age_seconds` (7%), `live_btc_usd` (6%) | `abs_delta_to_strike` (9%), `sequence_completeness_rate` (8%), `price_to_beat` (7%), `live_applied_bias_age_seconds` (7%), `event_count_15s` (7%) |
| `model_04_adverse_selection/lightgbm` | 50 | **20.3%** | 11.1% | **CRITICAL** | `price_to_beat` (9%), `live_applied_bias_age_seconds` (6%), `live_btc_usd` (5%) | `price_to_beat` (9%), `state_ask_depth_total` (6%), `price_level` (6%), `state_bid_depth_total` (6%), `live_applied_bias_age_seconds` (6%) |
| `model_08c_maker_defense_v3_eventtime/extreme_10c_5s/agg_only` | 32 | **20.0%** | 30.5% | **CRITICAL** | `live_applied_bias_age_seconds` (7%), `price_to_beat` (7%), `live_btc_usd` (6%) | `abs_delta_to_strike` (9%), `sequence_completeness_rate` (9%), `event_count_15s` (8%), `live_applied_bias_age_seconds` (7%), `t_since_open_s` (7%) |
| `model_08c_maker_defense_v3_eventtime/severe_7c_3s/agg_only` | 32 | **19.2%** | 44.9% | **CRITICAL** | `price_to_beat` (8%), `live_applied_bias_age_seconds` (6%), `live_btc_usd` (5%) | `event_count_15s` (10%), `sequence_completeness_rate` (8%), `price_to_beat` (8%), `event_count_3s` (7%), `event_count_1s` (7%) |
| `model_08c_maker_defense/moderate_5c_5s` | 49 | **18.3%** | 29.2% | **CRITICAL** | `live_btc_usd` (6%), `live_applied_bias_age_seconds` (6%), `price_to_beat` (6%) | `abs_delta_to_strike` (7%), `sequence_completeness_rate` (7%), `live_btc_usd` (6%), `live_applied_bias_age_seconds` (6%), `price_to_beat` (6%) |
| `model_08c_maker_defense_v2/extreme_10c_5s` | 58 | **18.1%** | 27.2% | **CRITICAL** | `live_applied_bias_age_seconds` (7%), `price_to_beat` (6%), `live_btc_usd` (5%) | `sequence_completeness_rate` (8%), `abs_delta_to_strike` (7%), `live_applied_bias_age_seconds` (7%), `price_to_beat` (6%), `t_since_open_s` (6%) |
| `model_08b_big_move_classifier/tabular_btc/lightgbm` | 49 | **18.0%** | 30.9% | **CRITICAL** | `live_applied_bias_age_seconds` (7%), `price_to_beat` (6%), `live_btc_usd` (5%) | `live_applied_bias_age_seconds` (7%), `imbalance_mean_5s` (7%), `abs_delta_to_strike` (6%), `event_count_15s` (6%), `price_to_beat` (6%) |
| `model_08c_maker_defense/extreme_10c_5s` | 49 | **17.6%** | 28.8% | **CRITICAL** | `live_applied_bias_age_seconds` (6%), `live_btc_usd` (6%), `price_to_beat` (5%) | `sequence_completeness_rate` (7%), `abs_delta_to_strike` (7%), `live_applied_bias_age_seconds` (6%), `t_since_open_s` (6%), `live_btc_usd` (6%) |
| `model_08c_maker_defense_v2/moderate_5c_5s` | 58 | **16.9%** | 28.7% | **CRITICAL** | `price_to_beat` (6%), `live_applied_bias_age_seconds` (6%), `live_btc_usd` (5%) | `sequence_completeness_rate` (7%), `abs_delta_to_strike` (7%), `price_to_beat` (6%), `event_count_15s` (6%), `live_applied_bias_age_seconds` (6%) |
| `model_08c_maker_defense/severe_7c_3s` | 49 | **16.7%** | 41.4% | **CRITICAL** | `live_applied_bias_age_seconds` (6%), `price_to_beat` (6%), `live_btc_usd` (5%) | `event_count_15s` (7%), `sequence_completeness_rate` (7%), `live_applied_bias_age_seconds` (6%), `price_to_beat` (6%), `live_btc_usd` (5%) |
| `model_08c_maker_defense_v2/severe_7c_3s` | 58 | **16.2%** | 37.5% | **CRITICAL** | `live_applied_bias_age_seconds` (6%), `price_to_beat` (5%), `live_btc_usd` (5%) | `event_count_15s` (7%), `sequence_completeness_rate` (6%), `live_applied_bias_age_seconds` (6%), `price_to_beat` (5%), `abs_delta_to_strike` (5%) |
| `model_02_fair_resolution/dense_close/lightgbm` | 74 | **14.9%** | 0.0% | **CRITICAL** | `price_to_beat` (8%), `live_applied_bias_age_seconds` (5%), `live_btc_usd` (3%) | `snapshot_second_of_day_utc` (9%), `price_to_beat` (8%), `snapshot_minute_utc` (7%), `live_bias_observation_count` (7%), `binance_usdm_mid` (6%) |
| `model_08c_maker_defense_v3_eventtime/severe_7c_3s/agg_plus_event` | 93 | **14.3%** | 25.8% | **CRITICAL** | `live_btc_usd` (5%), `price_to_beat` (5%), `live_applied_bias_age_seconds` (5%) | `event_count_15s` (6%), `sequence_completeness_rate` (5%), `live_btc_usd` (5%), `evt_last_event_dt_s` (5%), `price_to_beat` (5%) |
| `model_08c_maker_defense_v3_eventtime/moderate_5c_5s/agg_plus_event` | 93 | **14.3%** | 18.9% | **CRITICAL** | `live_applied_bias_age_seconds` (5%), `live_btc_usd` (5%), `price_to_beat` (5%) | `sequence_completeness_rate` (5%), `abs_delta_to_strike` (5%), `live_applied_bias_age_seconds` (5%), `live_btc_usd` (5%), `event_count_15s` (5%) |
| `model_08c_maker_defense_v3_eventtime/extreme_10c_5s/agg_plus_event` | 93 | **13.8%** | 20.4% | **CRITICAL** | `price_to_beat` (5%), `live_btc_usd` (5%), `live_applied_bias_age_seconds` (4%) | `sequence_completeness_rate` (6%), `abs_delta_to_strike` (5%), `event_count_15s` (5%), `price_to_beat` (5%), `t_since_open_s` (5%) |
| `model_03_fill_probability/lightgbm` | 50 | **12.6%** | 25.2% | **CRITICAL** | `price_to_beat` (5%), `live_applied_bias_age_seconds` (4%), `live_btc_usd` (4%) | `price_level` (7%), `state_stale_s` (7%), `t_since_open_s` (6%), `event_count_5s` (5%), `price_to_beat` (5%) |
| `model_07_microstructure_direction/tabular/lightgbm` | 43 | **11.8%** | 33.9% | **MODERATE** | `live_applied_bias_age_seconds` (4%), `price_to_beat` (4%), `live_btc_usd` (4%) | `imbalance_last` (8%), `imbalance_mean_5s` (7%), `t_since_open_s` (7%), `imbalance_slope_5s` (6%), `mid_return_5s` (5%) |
| `model_05_closing_flip/dense_close/lightgbm` | 74 | **11.1%** | 0.0% | **MODERATE** | `price_to_beat` (5%), `live_applied_bias_age_seconds` (4%), `live_btc_usd` (2%) | `snapshot_second_of_day_utc` (9%), `live_bias_observation_count` (7%), `binance_usdm_mid` (6%), `abs_delta_to_strike` (5%), `snapshot_minute_utc` (5%) |
| `model_06_mispricing/dense_close/lightgbm` | 74 | **10.5%** | 0.0% | **CRITICAL** | `price_to_beat` (5%), `live_applied_bias_age_seconds` (3%), `live_btc_usd` (2%) | `abs_delta_to_strike` (7%), `delta_to_strike` (6%), `snapshot_second_of_day_utc` (5%), `price_to_beat` (5%), `live_bias_observation_count` (4%) |
| `model_08_move_size/tabular/lightgbm` | 43 | **9.4%** | 36.7% | **MODERATE** | `live_applied_bias_age_seconds` (3%), `price_to_beat` (3%), `live_btc_usd` (3%) | `imbalance_last` (8%), `imbalance_mean_5s` (7%), `depth_change_rate_5s` (6%), `abs_delta_to_strike` (5%), `spread_max_5s` (5%) |
| `model_01_bias/preopen/lightgbm` | 93 | **4.4%** | 0.0% | **LOW** | `live_applied_bias_age_seconds` (3%), `price_to_beat` (1%), `live_btc_usd` (1%) | `spot_perp_divergence_usd` (6%), `delta_to_strike` (5%), `btc_return_60s` (4%), `btc_return_1800s` (4%), `btc_return_10s` (4%) |
| `model_01_bias/first15s/lightgbm` | 93 | **3.6%** | 0.0% | **LOW** | `live_applied_bias_age_seconds` (3%), `price_to_beat` (0%), `live_btc_usd` (0%) | `delta_to_strike` (6%), `cross_source_spread_usd` (5%), `spot_perp_divergence_usd` (4%), `btc_return_1800s` (4%), `abs_delta_to_strike` (4%) |
| `model_01_bias/first30s/lightgbm` | 93 | **2.5%** | 0.0% | **LOW** | `live_applied_bias_age_seconds` (2%), `price_to_beat` (1%), `live_btc_usd` (0%) | `delta_to_strike` (6%), `btc_return_60s` (4%), `cross_source_spread_usd` (4%), `btc_return_180s` (4%), `btc_return_1800s` (4%) |
| `model_08c_maker_defense_v3_eventtime/extreme_10c_5s/event_only` | 61 | **0.0%** | 0.0% | **LOW** | — | `evt_dt_mean` (5%), `evt_dt_std` (4%), `evt_bid_depth_ratio` (4%), `evt_ask_depth_ratio` (4%), `evt_last_event_dt_s` (4%) |
| `model_08c_maker_defense_v3_eventtime/moderate_5c_5s/event_only` | 61 | **0.0%** | 0.0% | **LOW** | — | `evt_dt_mean` (5%), `evt_dt_std` (4%), `evt_dt_max` (4%), `evt_bid_depth_ratio` (4%), `evt_micro_max_abs_diff` (4%) |
| `model_08c_maker_defense_v3_eventtime/severe_7c_3s/event_only` | 61 | **0.0%** | 0.0% | **LOW** | — | `evt_last_event_dt_s` (8%), `evt_dt_mean` (6%), `evt_dt_std` (5%), `evt_window_span_s` (4%), `evt_dt_max` (4%) |
| `model_01_bias/first15s/logistic_regression` | — | **—** | — | **n/a** | — |  |
| `model_01_bias/first30s/logistic_regression` | — | **—** | — | **n/a** | — |  |
| `model_01_bias/preopen/logistic_regression` | — | **—** | — | **n/a** | — |  |
| `model_02_fair_resolution/coarse/logistic_regression` | — | **—** | — | **n/a** | — |  |
| `model_03_fill_probability/logistic_regression` | — | **—** | — | **n/a** | — |  |
| `model_04_adverse_selection/linear_regression` | — | **—** | — | **n/a** | — |  |
| `model_04b_markout_to_close/linear_regression` | — | **—** | — | **n/a** | — |  |
| `model_05_closing_flip/dense_close/logistic_regression` | — | **—** | — | **n/a** | — |  |
| `model_06_mispricing/dense_close/linear_regression` | — | **—** | — | **n/a** | — |  |
| `model_07_microstructure_direction/tabular/logistic_regression` | — | **—** | — | **n/a** | — |  |
| `model_08_move_size/tabular/linear_regression` | — | **—** | — | **n/a** | — |  |

## Risk Distribution

### CRITICAL (19)
- `model_04b_markout_to_close/lightgbm`
- `model_02_fair_resolution/coarse/lightgbm`
- `model_08c_maker_defense_v3_eventtime/moderate_5c_5s/agg_only`
- `model_04_adverse_selection/lightgbm`
- `model_08c_maker_defense_v3_eventtime/extreme_10c_5s/agg_only`
- `model_08c_maker_defense_v3_eventtime/severe_7c_3s/agg_only`
- `model_08c_maker_defense/moderate_5c_5s`
- `model_08c_maker_defense_v2/extreme_10c_5s`
- `model_08b_big_move_classifier/tabular_btc/lightgbm`
- `model_08c_maker_defense/extreme_10c_5s`
- `model_08c_maker_defense_v2/moderate_5c_5s`
- `model_08c_maker_defense/severe_7c_3s`
- `model_08c_maker_defense_v2/severe_7c_3s`
- `model_02_fair_resolution/dense_close/lightgbm`
- `model_08c_maker_defense_v3_eventtime/severe_7c_3s/agg_plus_event`
- `model_08c_maker_defense_v3_eventtime/moderate_5c_5s/agg_plus_event`
- `model_08c_maker_defense_v3_eventtime/extreme_10c_5s/agg_plus_event`
- `model_03_fill_probability/lightgbm`
- `model_06_mispricing/dense_close/lightgbm`

### HIGH (0)
_(none)_

### MODERATE (3)
- `model_07_microstructure_direction/tabular/lightgbm`
- `model_05_closing_flip/dense_close/lightgbm`
- `model_08_move_size/tabular/lightgbm`

### LOW (6)
- `model_01_bias/preopen/lightgbm`
- `model_01_bias/first15s/lightgbm`
- `model_01_bias/first30s/lightgbm`
- `model_08c_maker_defense_v3_eventtime/extreme_10c_5s/event_only`
- `model_08c_maker_defense_v3_eventtime/moderate_5c_5s/event_only`
- `model_08c_maker_defense_v3_eventtime/severe_7c_3s/event_only`

### n/a (11)
- `model_01_bias/first15s/logistic_regression`
- `model_01_bias/first30s/logistic_regression`
- `model_01_bias/preopen/logistic_regression`
- `model_02_fair_resolution/coarse/logistic_regression`
- `model_03_fill_probability/logistic_regression`
- `model_04_adverse_selection/linear_regression`
- `model_04b_markout_to_close/linear_regression`
- `model_05_closing_flip/dense_close/logistic_regression`
- `model_06_mispricing/dense_close/linear_regression`
- `model_07_microstructure_direction/tabular/logistic_regression`
- `model_08_move_size/tabular/linear_regression`

