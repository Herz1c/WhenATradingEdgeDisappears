# delta_to_strike vs Historical Chainlink

## Scope

- Date range: `2026-04-19` to `2026-04-23`
- Compared value: stored Phase3 `delta_to_strike` versus hypothetical `historical_chainlink_price - price_to_beat`.
- Since `price_to_beat` is identical on both sides, the measured error is exactly `live_btc_usd - historical_chainlink_price`.
- Primary historical source: `chainlink_public_delayed` using its displayed Chainlink event timestamp, joined by as-of `chainlink_ts <= snapshot_ts`.
- Secondary source: `chainlink_onchain` using `round_data.updated_at_s`; included as a sanity check because it is a different / sparse verification feed.

## Source Coverage

| source | files | raw_rows | deduped_events | first_event_ts | last_event_ts |
| --- | --- | --- | --- | --- | --- |
| public_delayed_display_ts | 97 | 37027 | 17125 | 2026-04-19T22:01:25+00:00 | 2026-04-22T17:47:30+00:00 |
| onchain_updated_at | 97 | 8331 | 131 | 2026-04-19T21:55:23+00:00 | 2026-04-23T23:56:35+00:00 |

## Dataset Metrics

| dataset | source | matched_pct | medAE | p75 | p90 | p95 | p99 | mean_signed | median_signed | within_5_pct | within_10_pct | age_p95_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resolution_snapshot_dataset_v1_coarse | public_delayed_display_ts | 97.7952 | 11.4004 | 388.1541 | 969.7108 | 1144.3901 | 1405.4724 | -231.2418 | -4.5963 | 30.8008 | 47.1687 | 91527.4000 |
| resolution_snapshot_dataset_v1_coarse | onchain_updated_at | 97.7952 | 92.8963 | 174.2093 | 250.2968 | 291.0357 | 366.3536 | 5.7050 | 5.7409 | 3.3222 | 6.7434 | 3367.0000 |
| resolution_snapshot_dataset_v1_dense_close | public_delayed_display_ts | 97.7726 | 11.1044 | 407.3498 | 971.2682 | 1142.0307 | 1412.3654 | -231.5068 | -4.4857 | 30.9521 | 47.9600 | 91643.9950 |
| resolution_snapshot_dataset_v1_dense_close | onchain_updated_at | 97.7726 | 92.3813 | 174.7716 | 247.8491 | 286.3319 | 358.8356 | 4.2522 | 4.8137 | 3.0229 | 6.6303 | 3391.0000 |
| market_interval_dataset_v1_preopen | public_delayed_display_ts | 97.6068 | 11.3984 | 391.9179 | 973.8652 | 1137.2265 | 1416.8180 | -232.1276 | -3.8198 | 31.5236 | 47.8109 | 91334.0000 |
| market_interval_dataset_v1_preopen | onchain_updated_at | 97.6068 | 92.0987 | 173.7191 | 249.1803 | 284.5824 | 348.4538 | 2.7190 | 3.5486 | 2.8897 | 6.5674 | 3371.4000 |
| market_interval_dataset_v1_first15s | public_delayed_display_ts | 97.6068 | 13.6351 | 386.3390 | 974.9823 | 1145.8556 | 1417.2999 | -232.9315 | -4.8531 | 29.2469 | 44.6585 | 91350.0000 |
| market_interval_dataset_v1_first15s | onchain_updated_at | 97.6068 | 92.8655 | 172.0632 | 253.7924 | 284.6977 | 352.2478 | 3.4693 | 3.4411 | 2.9772 | 6.4799 | 3376.0000 |
| market_interval_dataset_v1_first30s | public_delayed_display_ts | 97.6068 | 10.9153 | 382.2577 | 979.2518 | 1144.8563 | 1401.4152 | -232.1598 | -4.5966 | 31.1734 | 47.1979 | 91065.0000 |
| market_interval_dataset_v1_first30s | onchain_updated_at | 97.6068 | 97.1145 | 174.1556 | 252.6896 | 291.4090 | 352.1768 | 2.9864 | 2.1420 | 3.5026 | 7.2680 | 3391.0000 |

## Dataset Metrics With Fresh Historical Chainlink

| dataset | source | max_age_s | matched_pct | medAE | p75 | p90 | p95 | p99 | mean_signed | median_signed | within_5_pct | within_10_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resolution_snapshot_dataset_v1_coarse | public_delayed_display_ts | 300 | 67.8515 | 5.8715 | 12.7821 | 24.4845 | 34.0899 | 61.3435 | -0.0958 | -0.2898 | 44.3156 | 67.8153 |
| resolution_snapshot_dataset_v1_coarse | public_delayed_display_ts | 3600 | 68.9034 | 6.0104 | 13.4007 | 26.5381 | 38.9077 | 116.9916 | 1.7379 | -0.1561 | 43.6390 | 66.7801 |
| resolution_snapshot_dataset_v1_coarse | onchain_updated_at | 300 | 10.6238 | 44.6607 | 93.4497 | 172.3676 | 226.9481 | 308.7852 | -1.0071 | -3.6782 | 6.6698 | 12.6619 |
| resolution_snapshot_dataset_v1_coarse | onchain_updated_at | 3600 | 97.3182 | 92.7626 | 174.0789 | 250.2384 | 291.1626 | 366.4608 | 5.7956 | 5.7968 | 3.3345 | 6.7618 |
| resolution_snapshot_dataset_v1_dense_close | public_delayed_display_ts | 300 | 67.7909 | 5.7631 | 12.5100 | 24.3944 | 33.2943 | 57.7420 | -0.0382 | -0.1915 | 44.4798 | 68.8534 |
| resolution_snapshot_dataset_v1_dense_close | public_delayed_display_ts | 3600 | 68.8165 | 5.8941 | 13.1522 | 26.3530 | 38.2813 | 115.5348 | 1.8853 | -0.0784 | 43.8169 | 67.8272 |
| resolution_snapshot_dataset_v1_dense_close | onchain_updated_at | 300 | 10.5934 | 44.8246 | 98.9758 | 163.2868 | 214.7989 | 305.6512 | -3.0089 | -4.4303 | 5.6585 | 11.9732 |
| resolution_snapshot_dataset_v1_dense_close | onchain_updated_at | 3600 | 97.3635 | 92.2255 | 174.8331 | 247.8183 | 286.0614 | 358.9717 | 4.4089 | 5.0675 | 3.0356 | 6.6286 |
| market_interval_dataset_v1_preopen | public_delayed_display_ts | 300 | 67.6068 | 5.6980 | 12.6969 | 24.7610 | 35.7941 | 59.6990 | 0.2521 | -0.1121 | 45.5120 | 68.9001 |
| market_interval_dataset_v1_preopen | public_delayed_display_ts | 3600 | 68.7179 | 5.7528 | 13.3420 | 27.8115 | 42.1128 | 106.2048 | 2.2507 | 0.0586 | 44.7761 | 67.7861 |
| market_interval_dataset_v1_preopen | onchain_updated_at | 300 | 10.8547 | 49.9599 | 102.0685 | 160.9885 | 194.7168 | 284.2154 | -8.5741 | -5.5142 | 3.9370 | 11.0236 |
| market_interval_dataset_v1_preopen | onchain_updated_at | 3600 | 97.4359 | 91.9653 | 173.8191 | 249.2865 | 284.6643 | 348.4568 | 2.7135 | 3.5486 | 2.8947 | 6.5789 |
| market_interval_dataset_v1_first15s | public_delayed_display_ts | 300 | 67.7778 | 6.2802 | 15.2125 | 27.5566 | 39.9942 | 65.1819 | 0.1844 | -0.1291 | 42.1185 | 64.3127 |
| market_interval_dataset_v1_first15s | public_delayed_display_ts | 3600 | 68.8034 | 6.4670 | 15.9653 | 31.0773 | 48.1380 | 106.1114 | 2.1027 | -0.0274 | 41.4907 | 63.3540 |
| market_interval_dataset_v1_first15s | onchain_updated_at | 300 | 10.5128 | 48.6495 | 101.6062 | 166.9066 | 218.4758 | 276.2394 | -8.2887 | 1.4245 | 4.8780 | 9.7561 |
| market_interval_dataset_v1_first15s | onchain_updated_at | 3600 | 97.0940 | 92.5965 | 172.1522 | 253.9542 | 285.0931 | 352.5095 | 3.6651 | 3.4411 | 2.9930 | 6.5141 |
| market_interval_dataset_v1_first30s | public_delayed_display_ts | 300 | 67.7778 | 5.9160 | 12.4184 | 25.8882 | 33.6857 | 70.2273 | -0.3418 | -0.3554 | 44.8928 | 67.8436 |
| market_interval_dataset_v1_first30s | public_delayed_display_ts | 3600 | 68.8034 | 6.0336 | 13.9542 | 27.3560 | 38.5449 | 106.7280 | 1.5321 | -0.2402 | 44.2236 | 66.8323 |
| market_interval_dataset_v1_first30s | onchain_updated_at | 300 | 10.5128 | 50.6476 | 108.0954 | 164.3962 | 235.6579 | 288.5146 | -10.1792 | -4.4922 | 8.1301 | 13.8211 |
| market_interval_dataset_v1_first30s | onchain_updated_at | 3600 | 96.9231 | 97.1145 | 174.1556 | 252.5490 | 291.5808 | 352.3421 | 3.2163 | 2.1420 | 3.5273 | 7.3192 |

## Dense Close By Date vs Public Delayed

| date | rows | median_abs_error_usd | p90_abs_error_usd | p95_abs_error_usd | p99_abs_error_usd | mean_error_usd | within_5_usd_pct | within_10_usd_pct | chainlink_age_p95_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-19 | 7200 | 6.4114 | 22.7652 | 30.0913 | 43.5463 | -0.7238 | 41.9167 | 63.6944 | 12.7500 |
| 2026-04-20 | 86330 | 5.7326 | 24.8658 | 33.7594 | 53.0526 | -0.0598 | 44.4388 | 68.6691 | 63.3000 |
| 2026-04-21 | 81116 | 5.4543 | 22.9050 | 31.6051 | 57.4858 | 0.2187 | 46.6862 | 71.5321 | 80.7500 |
| 2026-04-22 | 82200 | 9.2605 | 247.7462 | 297.6423 | 563.6865 | -36.2675 | 32.8102 | 51.9440 | 18412.2625 |
| 2026-04-23 | 86336 | 905.1665 | 1269.4491 | 1397.7408 | 1511.4285 | -885.7849 | 0.0000 | 0.0000 | 104535.5625 |

## Dense Close By Date vs Fresh Public Delayed (age <= 300s)

| date | matched_pct | rows | median_abs_error_usd | p90_abs_error_usd | p95_abs_error_usd | p99_abs_error_usd | mean_error_usd | within_5_usd_pct | within_10_usd_pct | chainlink_age_p95_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-19 | 100.0000 | 7200 | 6.4114 | 22.7652 | 30.0913 | 43.5463 | -0.7238 | 41.9167 | 63.6944 | 12.7500 |
| 2026-04-20 | 99.9190 | 86330 | 5.7326 | 24.8658 | 33.7594 | 53.0526 | -0.0598 | 44.4388 | 68.6691 | 63.3000 |
| 2026-04-21 | 95.5430 | 81116 | 5.4543 | 22.9050 | 31.6051 | 57.4858 | 0.2187 | 46.6862 | 71.5321 | 80.7500 |
| 2026-04-22 | 73.5192 | 63300 | 6.2885 | 25.8936 | 36.2058 | 71.3065 | -0.2601 | 42.0000 | 66.2591 | 51.5000 |
| 2026-04-23 | 0.0000 | 0 | None | None | None | None | None | None | None | None |

## Dense Close Bias/Eligibility Slices vs Public Delayed

| state | rows | median_abs_error_usd | p90_abs_error_usd | p95_abs_error_usd | p99_abs_error_usd | mean_error_usd | within_5_usd_pct | within_10_usd_pct | chainlink_age_p95_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bias_active | 237946 | 5.8505 | 26.3239 | 38.4382 | 115.6698 | 1.9001 | 43.9974 | 67.8978 | 89.7500 |
| bias_neutralized | 103436 | 808.4976 | 1222.7352 | 1366.1126 | 1501.9440 | -772.4532 | 0.8972 | 2.0805 | 103642.7250 |
| training_feature_eligible | 342979 | 11.1019 | 971.3012 | 1142.0307 | 1412.3654 | -231.5063 | 30.9555 | 47.9677 | 91644.3100 |
| complete_feature_matrix_eligible | 289772 | 10.9601 | 983.0356 | 1140.7554 | 1407.5653 | -238.0156 | 31.1738 | 48.1841 | 92792.1125 |

## Dense Close Bias/Eligibility Slices vs Fresh Public Delayed (age <= 300s)

| state | matched_pct | rows | median_abs_error_usd | p90_abs_error_usd | p95_abs_error_usd | p99_abs_error_usd | mean_error_usd | within_5_usd_pct | within_10_usd_pct | chainlink_age_p95_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bias_active | 66.7652 | 234346 | 5.7401 | 24.3126 | 33.2696 | 58.1290 | -0.0528 | 44.6733 | 68.9408 | 63.0000 |
| bias_neutralized | 0.5128 | 1800 | 6.9593 | 17.7921 | 28.8712 | 32.3251 | 2.7003 | 30.2222 | 77.5556 | 11.9000 |
| training_feature_eligible | 67.7487 | 237798 | 5.7626 | 24.3906 | 33.2928 | 57.7425 | -0.0306 | 44.4861 | 68.8664 | 61.7500 |
| complete_feature_matrix_eligible | 56.2339 | 197381 | 5.5665 | 22.8138 | 32.0752 | 54.4271 | -0.0355 | 45.5713 | 70.3553 | 62.2500 |
