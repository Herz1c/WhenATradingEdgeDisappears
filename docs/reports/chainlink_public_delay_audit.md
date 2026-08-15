# Chainlink Public Delayed Page Delay Audit

This audit uses only `finalized_clean` files from `chainlink_public_delayed` and `chainlink_onchain`, excludes the `2026-04-20 07:00 UTC` quarantine hour, and uses `recv_ts_ns` as the primary clock throughout.

Exact-value matching is constrained to a generous plausibility window of `[-600s, +3600s]` to avoid falsely pairing unrelated same-price recurrences hours later.

## Summary

| metric | value |
| --- | --- |
| onchain_update_events | 114 |
| matched_events | 2 |
| unmatched_events | 112 |
| match_coverage | 1.75% |
| mean_delay_seconds | 236.009 |
| median_delay_seconds | 236.009 |
| p5_delay_seconds | 35.268 |
| p95_delay_seconds | 436.749 |
| p99_delay_seconds | 454.593 |
| min_delay_seconds | 12.964 |
| max_delay_seconds | 459.054 |
| negative_delay_matches | 0 |

## Delay Histogram (10s buckets)

| bucket | count |
| --- | --- |
| [10, 20) | 1 |
| [450, 460) | 1 |

## Hour Of Day Breakdown

| bucket | matched_events | mean_delay_seconds | median_delay_seconds | p95_delay_seconds | min_delay_seconds | max_delay_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 1 | 459.054 | 459.054 | 459.054 | 459.054 | 459.054 |
| 15 | 1 | 12.964 | 12.964 | 12.964 | 12.964 | 12.964 |

## Day Of Week Breakdown

| bucket | matched_events | mean_delay_seconds | median_delay_seconds | p95_delay_seconds | min_delay_seconds | max_delay_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Monday | 2 | 236.009 | 236.009 | 436.749 | 12.964 | 459.054 |

## Volatility Regime Breakdown

| bucket | matched_events | mean_delay_seconds | median_delay_seconds | p95_delay_seconds | min_delay_seconds | max_delay_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| high | 1 | 12.964 | 12.964 | 12.964 | 12.964 | 12.964 |
| medium | 1 | 459.054 | 459.054 | 459.054 | 459.054 | 459.054 |

## Recommendation

- Status: `insufficient_exact_match_coverage`
- Recommended causal offset: `1800.000s`
- Reason: Exact +/- 0.01 USD value matching produced too few plausible pairs to estimate a robust operational delay distribution. Keep the conservative 1800s availability floor in the dataset factory.

## Matched Pair Sample

| onchain_recv_ts_ns | onchain_recv_ts_iso | onchain_price | round_id | delayed_recv_ts_ns | delayed_recv_ts_iso | delayed_price | delayed_display_ts_ns | delayed_display_ts_iso | delivery_delay_seconds | volatility_regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1776684186972937200 | 2026-04-20T11:23:06.972937Z | 75135.890 | 129127208515966881143 | 1776684646026749400 | 2026-04-20T11:30:46.026749Z | 75135.881 | 1776684642000000000 | 2026-04-20T11:30:42Z | 459.054 | medium |
| 1776698842318571500 | 2026-04-20T15:27:22.318572Z | 75597.938 | 129127208515966881151 | 1776698855282103300 | 2026-04-20T15:27:35.282103Z | 75597.940 | 1776698849000000000 | 2026-04-20T15:27:29Z | 12.964 | high |
