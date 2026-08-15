# Chainlink Proxy v2 Research

> Superseded for official `live_reference_events_v1` pricing by `docs/spot_only_synthetic_chainlink_validation_report.md`.
> This file remains as a historical residual-window research report.

## Scope

- Target: minimize synthetic Chainlink BTC/USD error without data leakage.
- Evaluation rows: Chainlink public page event timestamps from the experiment cache.
- Base proxy: `spot_premium_calibrated_v1` from the existing experiment cache.
- Strict calibration rule: Chainlink calibration observations are usable only if their event timestamp is at least `1800s` older than the prediction timestamp.
- Fallback: if no strict calibration window exists yet, the base proxy is emitted unchanged.

## Strict 1800s-Safe Results

| candidate | delay_s | window_s | agg | medAE | p90 | p95 | within_3 | active | safe |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| spot_premium_calibrated_v1 | 0 | n/a | none | 5.2582 | 14.1803 | 19.5016 | 30.34% | 100.00% | yes |
| baseline_plus_last_30m_residual | 1800 | n/a | last | 4.3526 | 12.3979 | 17.0682 | 36.22% | 99.12% | yes |
| baseline_plus_mean_residual_1800s_ending_30m_old | 1800 | 1800 | mean | 3.5227 | 9.4535 | 12.1985 | 43.43% | 98.27% | yes |
| baseline_plus_mean_residual_1200s_ending_30m_old | 1800 | 1200 | mean | 3.5300 | 9.4051 | 12.1943 | 43.74% | 98.06% | yes |
| baseline_plus_mean_residual_900s_ending_30m_old | 1800 | 900 | mean | 3.5402 | 9.3727 | 12.1590 | 43.52% | 97.92% | yes |
| baseline_plus_mean_residual_1500s_ending_30m_old | 1800 | 1500 | mean | 3.5449 | 9.4004 | 12.1856 | 43.48% | 98.21% | yes |
| baseline_plus_mean_residual_2100s_ending_30m_old | 1800 | 2100 | mean | 3.5475 | 9.4673 | 12.2464 | 43.54% | 98.42% | yes |
| baseline_plus_mean_residual_2400s_ending_30m_old | 1800 | 2400 | mean | 3.5531 | 9.5024 | 12.3310 | 43.31% | 98.57% | yes |
| baseline_plus_median_residual_1800s_ending_30m_old | 1800 | 1800 | median | 3.5536 | 9.5741 | 12.2666 | 43.35% | 98.27% | yes |
| baseline_plus_median_residual_1200s_ending_30m_old | 1800 | 1200 | median | 3.5574 | 9.4198 | 12.2494 | 43.21% | 98.06% | yes |

## Best Strict Candidate By Date

| date | rows | active | medAE | p90 | within_3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-04-19 | 569 | 73.99% | 2.8782 | 8.0564 | 51.85% |
| 2026-04-20 | 6037 | 100.00% | 3.7976 | 9.4003 | 40.75% |
| 2026-04-21 | 5801 | 100.00% | 3.2480 | 8.1936 | 46.35% |
| 2026-04-22 | 4439 | 96.78% | 3.6849 | 11.2823 | 42.17% |

## Diagnostic Only: Non-1800s Availability

| candidate | delay_s | window_s | agg | medAE | p90 | p95 | within_3 | active | safe |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| diagnostic_baseline_plus_median_residual_300s_delay_0s | 0 | 300 | median | 2.2826 | 6.7202 | 9.5596 | 60.68% | 100.00% | diagnostic |
| diagnostic_baseline_plus_median_residual_300s_delay_5s | 5 | 300 | median | 2.5047 | 6.9884 | 9.8113 | 57.53% | 99.97% | diagnostic |
| diagnostic_baseline_plus_median_residual_300s_delay_10s | 10 | 300 | median | 2.5043 | 6.9925 | 9.8113 | 57.53% | 99.97% | diagnostic |
| diagnostic_baseline_plus_median_residual_300s_delay_30s | 30 | 300 | median | 2.5673 | 7.1160 | 9.9022 | 56.60% | 99.91% | diagnostic |
| diagnostic_baseline_plus_median_residual_300s_delay_60s | 60 | 300 | median | 2.5930 | 7.1940 | 9.9368 | 56.16% | 99.86% | diagnostic |

These diagnostic candidates are useful for understanding the upper bound if the public page is treated as available shortly after the displayed Chainlink timestamp. They are not strict-training-safe under the current roadmap/policy contract.

## Conclusion

- Best strict candidate: `baseline_plus_mean_residual_1800s_ending_30m_old`.
- Strict median absolute error: `3.5227 USD`.
- Strict p90 absolute error: `9.4535 USD`.
- This improves materially over the current base proxy, but it does not reproduce a sub-2.5 USD median error under the strict 1800s calibration floor.
