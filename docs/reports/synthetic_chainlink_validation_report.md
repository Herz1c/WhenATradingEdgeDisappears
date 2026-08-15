# Synthetic Chainlink Validation Report

> Superseded for official `live_reference_events_v1` pricing by `docs/spot_only_synthetic_chainlink_validation_report.md`.
> This file remains as a historical multi-source candidate comparison.

## Scope

- Task: reconstruct a synthetic BTC/USD Chainlink proxy from recorded venue mids only
- Input sources: Binance Spot WS, Binance USD-M WS, Hyperliquid WS
- Validation-only source: Chainlink delayed page
- Excluded by policy: RTDS, all Polymarket-side price inputs, active files, non-finalized or non-clean files
- Historical base proxy validated here: `spot_premium_calibrated_v1`
- Current official live-reference stream: `live_reference_events_v1.synthetic_corrected = binance_spot_mid * 1.00029 + mean(delayed_chainlink_residual[t-60m,t-30m])`

## Validation Table

| Candidate | Matched Events | Coverage | Mean Error USD | Median Abs Error USD | P90 Abs Error USD | P99 Abs Error USD | Within +/-1 USD | Within +/-3 USD | Within +/-10 USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spot_premium_calibrated_v1 | 16846 | 98.37% | 1.3937 | 5.2582 | 14.1803 | 29.7714 | 10.35% | 30.34% | 78.22% |
| binance_spot_raw | 16846 | 98.37% | -21.9070 | 22.7550 | 31.7177 | 42.5720 | 0.75% | 2.32% | 7.66% |
| median_spot_only | 16846 | 98.37% | -21.9070 | 22.7550 | 31.7177 | 42.5720 | 0.75% | 2.32% | 7.66% |
| volume_weighted_mid | 17119 | 99.96% | -56.9840 | 57.3398 | 69.5499 | 87.4925 | 0.01% | 0.04% | 0.15% |
| median_all | 17119 | 99.96% | -56.9811 | 58.3822 | 70.7240 | 82.3202 | 0.02% | 0.06% | 0.21% |
| trimmed_mean | 17119 | 99.96% | -56.9811 | 58.3822 | 70.7240 | 82.3202 | 0.02% | 0.06% | 0.21% |

## Selected Candidate

- Winner: `spot_premium_calibrated_v1`
- Median absolute error: 5.2582 USD
- P90 absolute error: 14.1803 USD
- Matched events: 16846

## Error Breakdown: Time Of Day

| Bucket | Count | Mean Error USD | Median Abs Error USD | P90 Abs Error USD |
| --- | --- | --- | --- | --- |
| 00-03 UTC | 2780 | 0.2491 | 4.7148 | 10.6448 |
| 04-07 UTC | 2986 | 5.5273 | 5.8317 | 22.8706 |
| 08-11 UTC | 2862 | 3.5577 | 4.9754 | 12.9450 |
| 12-15 UTC | 3193 | -0.8201 | 5.0761 | 14.6140 |
| 16-19 UTC | 2684 | 1.2590 | 6.2872 | 12.8241 |
| 20-23 UTC | 2341 | -1.9912 | 4.7874 | 11.6779 |

## Error Breakdown: Day Of Week

| Bucket | Count | Mean Error USD | Median Abs Error USD | P90 Abs Error USD |
| --- | --- | --- | --- | --- |
| Monday | 6037 | -4.0180 | 4.9641 | 12.3643 |
| Tuesday | 5801 | 1.9919 | 4.3183 | 11.0283 |
| Wednesday | 4439 | 8.4725 | 8.2921 | 21.8106 |
| Sunday | 569 | -2.5120 | 3.3638 | 8.6418 |

## Error Breakdown: Binance Spot Realized Volatility

| Bucket | Count | Mean Error USD | Median Abs Error USD | P90 Abs Error USD |
| --- | --- | --- | --- | --- |
| low | 5616 | 0.9643 | 4.8828 | 12.7597 |
| medium | 5615 | 1.7937 | 5.0779 | 14.1634 |
| high | 5615 | 1.4232 | 5.8666 | 15.9876 |

## Error Breakdown: Source Availability

| Bucket | Count | Mean Error USD | Median Abs Error USD | P90 Abs Error USD |
| --- | --- | --- | --- | --- |
| 2 sources | 1270 | 2.6275 | 5.9166 | 15.3401 |
| 3 sources | 15576 | 1.2931 | 5.2261 | 14.0123 |

## Error Breakdown: Proximity To Chainlink Update

| Bucket | Count | Mean Error USD | Median Abs Error USD | P90 Abs Error USD |
| --- | --- | --- | --- | --- |
| 0-5s_after_update | 84116 | 1.4399 | 5.7928 | 17.3039 |
| 5-10s_after_update | 54593 | 1.4622 | 7.6335 | 24.3543 |
| 10-15s_after_update | 6299 | 2.1325 | 9.2390 | 27.7756 |
| 15-20s_after_update | 6215 | 1.8012 | 10.7616 | 31.4123 |
| 20-25s_after_update | 2488 | 0.1414 | 11.4765 | 29.7955 |
| 25-30s_after_update | 2255 | -0.5615 | 11.7540 | 32.9182 |
| 0-5s_before_next_update | 67313 | 1.5020 | 9.6086 | 31.1841 |
| other | 15257 | -1.4907 | 18.3956 | 51.3426 |
