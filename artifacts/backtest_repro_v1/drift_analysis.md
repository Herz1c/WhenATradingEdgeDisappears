# Drift analysis: why combined PnL differed across reports (2026-07-10)

Gate **D1: PASS**. `tools/backtest_locked_strategy.py`, running on the shared engine
`src/backtest/episode_strategy_backtester.py`, reproduces the canonical
`tcn_double_strategy_v1` lock numbers **exactly** — all eight metrics per slot and
combined, to four decimal places, at overlap 60:

| metric | lock (canonical) | reproduction |
|---|---|---|
| combined trades | 706 | 706 |
| combined PnL | +214.1461 | +214.1461 |
| positive days | 24/32 | 24/32 |
| worst day / maxDD | −18.0556 | −18.0556 |
| early trades / PnL | 417 / +119.7780 | 417 / +119.7780 |
| late trades / PnL | 289 / +94.3681 | 289 / +94.3681 |

Inputs: `data/datasets/btc_5m_episodes_v1_200ms/test.npz` (frozen split
2026-05-14 → 07-02), `artifacts/btc_5m_episode_tcn_c64_b7_cal_ttc15_90/model.pt`
(forward pass over the whole episode, batch 64, CPU, deterministic eval), and the lock
JSON. Outputs: `trades_test_cap_drop.csv`, `summary_test_cap_drop.json`,
`comparison_test_cap_drop.txt`.

## Why four different "combined" numbers were in circulation

| number | trades | source | semantics |
|---|---|---|---|
| **+214.15** | 706 | `market_vs_tcn_same_strategy_report.json` (canonical) | both slots independently, one entry per market per `strategy_id`, early slot with the UTC 02–13 filter. **This is what reproduces.** |
| +220.36 | 702 | `two_strategy_early_hour_filtered_report.json` (discovery) | same construction, but the late trade set was carried over from an older saved artifact (285 late trades against 289) — stale file reuse, which the lock itself flags in a note. |
| +238.18 | 1329 | same file, row "all hours 0-23" | early slot *without* the hour filter (1044 trades) plus late. A different strategy, not drift. |
| +165.21 | 1130 | `exact_tcn_combined_75_120_then_50_75.json` | sequential mode "75-120 then 50-75" with a global one-entry-per-market rule (no double entries) and no hour filter on early. Different portfolio semantics, not drift. |

None of the discrepancies is a calculation error. They are four different portfolio
semantics plus one case of stale-artifact reuse. From this point the canonical numbers
are defined solely as the output of `tools/backtest_locked_strategy.py` over committed
inputs.

## Notes

- The model forward pass is deterministic (eval mode, dropout 0), so repeated runs give
  identical numbers.
- One of the three raw roots named in the manifest is an external archive that was not
  attached at the time. That does not affect this reproduction — the frozen `.npz` and
  the daily shards are local — but rebuilding old shards from raw would need it.
- The engine supports `--fill-mode fill_worse` (slippage priced in rather than used as a
  filter) and a `--delay-s` override for the latency sweep. Both are used by the
  execution-realism gate.
