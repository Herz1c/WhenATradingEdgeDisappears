# Known Failure Modes

> Superseded for official `live_reference_events_v1` pricing by `docs/spot_only_synthetic_chainlink_validation_report.md`.
> This file remains as a historical multi-source builder note.

Historical selected builder: `spot_premium_calibrated_v1`

## Structural Limits

- The synthetic builder never uses RTDS or Polymarket data.
- Any source older than 30 seconds is treated as missing.
- The builder returns `NaN` when fewer than 2 fresh sources are available.

## Conditions With Elevated Error

### Time of day

- `00-03 UTC`: median abs error 4.7148 USD, p90 10.6448 USD, count 2780
- `04-07 UTC`: median abs error 5.8317 USD, p90 22.8706 USD, count 2986
- `08-11 UTC`: median abs error 4.9754 USD, p90 12.9450 USD, count 2862
- `12-15 UTC`: median abs error 5.0761 USD, p90 14.6140 USD, count 3193
- `16-19 UTC`: median abs error 6.2872 USD, p90 12.8241 USD, count 2684
- `20-23 UTC`: median abs error 4.7874 USD, p90 11.6779 USD, count 2341

### Day of week

- `Monday`: median abs error 4.9641 USD, p90 12.3643 USD, count 6037
- `Tuesday`: median abs error 4.3183 USD, p90 11.0283 USD, count 5801
- `Wednesday`: median abs error 8.2921 USD, p90 21.8106 USD, count 4439
- `Sunday`: median abs error 3.3638 USD, p90 8.6418 USD, count 569

### Volatility regime

- `low`: median abs error 4.8828 USD, p90 12.7597 USD, count 5616
- `medium`: median abs error 5.0779 USD, p90 14.1634 USD, count 5615
- `high`: median abs error 5.8666 USD, p90 15.9876 USD, count 5615

### Source availability

- `2 sources`: median abs error 5.9166 USD, p90 15.3401 USD, count 1270
- `3 sources`: median abs error 5.2261 USD, p90 14.0123 USD, count 15576

### Update proximity

- `0-5s_after_update`: median abs error 5.7928 USD, p90 17.3039 USD, count 84116
- `5-10s_after_update`: median abs error 7.6335 USD, p90 24.3543 USD, count 54593
- `10-15s_after_update`: median abs error 9.2390 USD, p90 27.7756 USD, count 6299
- `15-20s_after_update`: median abs error 10.7616 USD, p90 31.4123 USD, count 6215
- `20-25s_after_update`: median abs error 11.4765 USD, p90 29.7955 USD, count 2488
- `25-30s_after_update`: median abs error 11.7540 USD, p90 32.9182 USD, count 2255
- `0-5s_before_next_update`: median abs error 9.6086 USD, p90 31.1841 USD, count 67313
- `other`: median abs error 18.3956 USD, p90 51.3426 USD, count 15257

## Mitigations

- Keep the finalized-clean quality gate and Binance USD-M quarantine hour in force.
- Treat missing Binance spot state as a hard builder miss because the selected method is spot-anchored.
- Keep multi-source candidates available for future research, but do not promote them over the validated winner without a fresh validation run.
- Treat low-source-count seconds as missing in production.
