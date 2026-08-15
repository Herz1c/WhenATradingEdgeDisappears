# Results and failure analysis

## Bottom line

**No demonstrated trading edge.** Three independent problems prevent a positive claim:
the forecast difference is statistically unresolved, the selected strategy fails a
multiple-testing correction, and the underlying replay clock is not live-causal.

| Gate | Public verdict |
|---|---|
| Forecasting | Exploratory only; clustered interval includes zero |
| Strategy selection | Fail; corrected White's Reality Check p=0.5667 |
| Causal availability | Fail; source-time counterfactual |
| Execution | No valid verdict |
| Prospective operation | Demonstrated locally; performance not demonstrated |
| Overall | **No demonstrated trading edge** |

## 1. Forecasting result

The released arrays contain 66,050 validation and 96,494 test loss-band rows. The test
rows come from 388 markets across only five days. The lowest observed test score after
inspecting seven mappings is Platt-L2:

| Quantity | Value |
|---|---:|
| Five-seed mean model Brier | 0.127732 |
| Market Brier | 0.128276 |
| Model minus market | -0.000544 |
| Market-clustered 95% interval | [-0.002730, 0.001721] |
| Seeds individually below market | 4 / 5 |

All seven primary market-cluster intervals include zero. Isotonic has the best validation
score but is worse than the market on test. The reported Platt row is therefore a post-hoc
descriptive minimum, not a confirmatory win. The small point difference, shared events,
selection on test, five-day window, and source-time clock jointly support only “not
demonstrated.”

## 2. Selected backtest

The stored historical summary reports 706 trades and +214.1461 over 32 available days.
Those values are internally consistent with the retained two-slot lock. They are not
evidence of an edge.

The older report gives White's Reality Check p=0.8355 for an 846-candidate matrix. A
fresh run of the current method does not reproduce that digit, and the matrix omits the
combined reported winner. Rather than defend the stale value, the public correction adds
that winner and reruns a deterministic 20,000-replicate audit:

| Corrected selection audit | Value |
|---|---:|
| Candidates | 847 |
| Days | 32 |
| Best mean daily PnL | 6.6921 |
| Finite-sample-corrected p-value | 0.5667 |
| 10% rejection threshold passed? | No |

The two individual locked configurations rank fifth and sixteenth in the original grid.
A distinct all-day early configuration ranks second. The combined winner's apparent
profit is ordinary relative to the searched family under the corrected null.

## 3. Why the clock dominates everything

The episode builder replays venue messages at source time. Receive time is recorded but
was not the state-transition clock. Late observations can therefore become historically
visible before the recorder knew them. The model score, strategy entries, book selection,
and offline post-lock replay inherit this issue.

The stored results are therefore best interpreted as a **zero-latency source-time
counterfactual**.

This does not prove the strategy has no information; it means the experiment cannot tell
us whether that information was available in time. The appropriate conclusion is a
failed measurement, not a claim of live-achievable profit.

## 4. Execution audit

The stored execution analysis cannot establish fillability. Twenty decisions without a
matching book are dropped, 72 of 162 inspected books are more than ten seconds old, the
maximum age is 62.497 seconds, and the worse-fill run changes the trade count from 706 to
722. Queue position, quote response, and self-impact are also absent. The correct verdict
is **unresolved / invalid audit**.

## 5. Post-lock and shadow evidence

An older five-day offline replay reports 262 trades, +23.90, and a 90% day-bootstrap
interval of [-15.95, 60.36]. It uses the same source-time episodes and its interval
contains zero. It is not a live forward-test score.

The newly released July 12–13 shadow sample is different evidence. Complete per-day
counts record 45,436 decisions: 164 `ENTER`, 26,691 `SKIP_NO_EDGE`, and 18,581
`SKIP_SOURCE`. Only sanitized entry rows are distributed. These locally dated logs show
that the guarded system operated; they do not provide fills, outcomes, or independent
timestamps.

## 6. Reproduction achieved in this release

The release improves on a summary-only repository in two concrete ways:

- Brier scores, seven calibration variants, paired losses, and clustered intervals are
  recomputed from all public validation/test prediction rows.
- The selection verdict is recomputed from all 847 daily PnL series, including the
  omitted combined winner.
- Five checkpoints rerun on a deterministic ten-market fixture and reproduce 2,586
  eligible rows per seed within 0.0001 logit.

Raw-data rebuilding, training, and receive-time correction remain private-data dependent.
The public evidence is therefore strong for the negative audit conclusion, not for a
positive market claim.
