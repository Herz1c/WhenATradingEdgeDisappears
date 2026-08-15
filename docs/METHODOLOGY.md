# Methodology and evidence policy

## Research design

The target is the binary resolution of a Polymarket BTC five-minute Up/Down market. The
market probability is treated as the baseline forecast. A temporal convolutional network
(TCN) predicts a bounded residual over its logit using cross-venue and within-market
features at 200 ms cadence.

The private v2 episode dataset contains 7,470 training, 917 validation, and 516 test
markets. Its published split metadata fixes train through June 16, validation June 23–27,
and test June 28–July 2, 2026. Each episode has 1,500 steps and 69 numeric features plus a
validity-mask channel. The TCN has seven causal convolution blocks, 64 channels, a
101.8-second receptive field, and residual output clamped to ±0.75 logit.

Five fixed seeds (7, 11, 23, 42, 101) share the same events. They measure optimization
sensitivity, not five independent replications.

## The causal-clock failure

Three times must not be confused:

- source time: when a venue says an event occurred;
- receive time: when the recorder observed it locally; and
- decision time: when a simulated or live policy could act.

A causal join requires `right.availability_time <= decision_time`. The historical episode
builder instead inserted Polymarket and centralized-exchange observations under source
time and retained receive time mainly as an audit field. RTDS availability was similarly
constructed from its source timestamp plus configured latency.

If a payload is sourced at 12:00:00 but received at 12:02:00, source-time replay makes it
visible two minutes early. A source-age filter cannot undo that admission. Consequently,
every score and PnL result built on these episodes is a zero-latency counterfactual. The
historical builder now requires the explicit flag
`--allow-source-time-counterfactual`; this prevents accidental causal claims but does not
repair old data.

## Forecast evaluation

The public prediction arrays contain all rows satisfying the historical 15–150 second
loss band. For each seed and split, the released `logit`, `base_logit`, `delta`, label,
market, date, episode index, and step index are sufficient to recompute the scores.

Seven mappings were evaluated:

1. identity;
2. validation temperature;
3. validation Platt-L2;
4. validation isotonic;
5. validation residual shrinkage;
6. train-day-block temperature; and
7. train-day-block residual shrinkage.

The table reports mean Brier across seeds, but uncertainty is paired at the row level:
the five seed losses are averaged for each row, the market loss for the same row is
subtracted, and entire test markets are resampled with replacement. Cluster sizes are
retained. The primary 95% interval uses 20,000 bootstrap replicates over 388 test
markets with seed 20260809. A five-cluster day bootstrap is reported only as a weak
sensitivity check.

The lowest test row is labeled post hoc because all seven test scores were inspected. A
future confirmatory experiment must lock one mapping before evaluation.

## Selection-bias audit

The original daily-PnL matrix stores 846 searched strategies across 32 days. It omitted
the subsequently reported combined two-slot winner even though that winner was the
headline backtest. The public correction appends the combined daily series, producing an
847-candidate universe.

White's Reality Check tests the maximum mean daily PnL. Each candidate is recentered by
its sample mean, and days are drawn with a Politis–Romano stationary bootstrap with mean
block length three. The public audit uses 20,000 replicates, seed 20260809, and the
finite-sample correction `(exceedances + 1) / (B + 1)`. This is still a lower-bound trial
count: undocumented human choices cannot be reconstructed.

## Execution and prospective evidence

A valid execution stress must hold decisions fixed, choose books available by decision or
fill time, impose strict staleness limits, treat missing state adversely, and model queue
loss and self-impact. The stored execution audit instead selects source-time books, drops
20 missing observations, includes books up to 62.5 seconds old, and changes the decision
set under adverse-fill stress. Its values are descriptive and support no fillability or
capacity verdict.

The July 12–13 shadow sample preserves every decision count and sanitized `ENTER` rows.
It demonstrates that source-health and edge gates ran prospectively. Local timestamps and
hashes are not external timestamps, and these logs are not PnL evidence.

## Required confirmatory experiment

A defensible positive claim would require:

1. rebuilding every stream under recorded receive/availability time;
2. rejecting missing or non-monotonic availability timestamps;
3. freezing one model, calibration rule, score, and decision policy;
4. externally timestamping that lock before collecting new data;
5. evaluating paired outcomes on unseen markets with dependence-aware uncertainty;
6. logging the complete live decision denominator; and
7. applying decision-matched, conservative execution accounting.

No such confirmatory result is claimed here.
