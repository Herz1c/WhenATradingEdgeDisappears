# When a Trading Edge Disappears
Causal Clocks, Selection Bias, and a Bitcoin Prediction Market Study  
Jan Herza  
Independent high school research portfolio  
April to August 2026 | Public evidence edition

### Abstract

I investigated whether cross-venue Bitcoin microstructure could improve the probability
quoted by a five-minute prediction market. The project recorded multiple exchanges,
Polymarket order books, and oracle-related feeds; built 5 Hz episodes; trained a residual
temporal convolutional network (TCN); simulated a two-slot strategy; and operated a
guarded shadow system. Early results appeared profitable. Adversarial evaluation changed
the conclusion.

The lowest observed test Brier score after inspecting seven calibration maps is 0.127732,
versus 0.128276 for the market. The paired 388-market 95% interval for model minus market
is [-0.002730, 0.001721]. A corrected White's Reality Check appends the combined reported
winner to the original search matrix and gives p=0.5667 across 847 candidates. The
historical episode builder also replayed messages at venue source time rather than local
receive time, allowing late messages to appear before they were available. Therefore the
positive backtest is a zero-latency counterfactual, not live-achievable evidence.

The final verdict is **{{C10_OVERALL.verdict}}**. The contribution is a reproducible
negative result and a concrete account of how causal clocks, dependence, selection, and
execution assumptions can erase an apparent edge.

I conceived and directed the study, including the hypotheses, causal rules, audit design,
failure criteria, and interpretation. AI tools supported much of the software
implementation.

### Research question

Can high-frequency cross-venue Bitcoin information improve the prediction market's own
probability near resolution, and can any improvement survive realistic statistical and
execution constraints?

| Hypothesis | Operational test | Result |
|---|---|---|
| H1: TCN residual improves forecast | Five seeds, seven maps, paired cluster bootstrap | Not demonstrated |
| H2: rule survives search | 847-candidate White's Reality Check | Fail, p=0.5667 |
| H3: profit is live-achievable | Availability-clock and execution audits | Not demonstrated |

[[PAGEBREAK]]

## 1. System and data

The software includes asynchronous recorders for Polymarket and several centralized
venues; manifest and quality layers; canonical data builders; model training; replay;
shadow decisions; and statistical audits. The locally measured private inventory reports
246.7 billion compressed bytes, 119,665 files, and 76 recorded days between April 19 and
August 9, with 37 missing calendar days. This is a committed inventory claim, not a
publicly recountable corpus total.

The v2 modelling dataset contains 7,470 training, 917 validation, and 516 test markets.
Each market is represented by 1,500 time steps at 200 ms cadence, 69 numerical features,
and a validity-mask channel. Training ends June 16; validation covers June 23-27; test
covers June 28-July 2. Full tensors remain private, but their hashes, feature metadata,
normalization, and split contract are public.

The model is a seven-block causal residual TCN with 64 channels and a 101.8-second
receptive field. It predicts a residual over `logit(p_market)`, clamped to +/-0.75. Five
fixed seeds share the same events. The public release includes all five checkpoints, all
validation/test loss-band predictions, and a ten-market inference fixture.

## 2. The central methodological failure: availability time

An event has at least two relevant times: when the venue says it occurred and when the
recorder received it. A live policy cannot act on the message before receipt. Historical
state must therefore advance under availability time, while source time remains useful
for staleness and latency diagnostics.

[[CAUSAL_CLOCK_FIGURE]]

The episode builder historically inserted Polymarket and centralized-exchange messages
under venue source time. Receive timestamps were audit fields rather than the admission
clock. If a source timestamp is 12:00 but recorder receipt is 12:02, the replay can expose
the message two minutes early. A freshness check after admission cannot repair this.

The correct classification is **{{C1_DATASET_CLOCK.verdict}}**. Model scores and strategy
PnL can still describe a zero-latency counterfactual, but cannot establish information or
profit available to a live participant. A valid correction requires raw receive-time
reconstruction and retraining; that experiment has not been performed.

[[PAGEBREAK]]

## 3. Forecast evaluation

### Scoring design

The evaluation uses every valid 200 ms row with 15 to 150 seconds remaining: 66,050
validation rows and 96,494 test rows. Brier loss is paired against the market on the same
row. Seven mappings were examined: identity; validation temperature, Platt L2, isotonic,
and residual shrinkage; and train day block temperature and residual shrinkage.

For uncertainty, I average the five seed losses within each row, subtract the market loss,
and resample whole markets with replacement while retaining cluster size. The primary
20,000-replicate interval has 388 test-market clusters. A five-day cluster bootstrap is
only a sensitivity check because five clusters are too few for stable inference.

[[BRIER_FIGURE]]

| Test quantity | Value |
|---|---:|
| Lowest observed model Brier (Platt L2) | 0.127732 |
| Market Brier | 0.128276 |
| Difference | -0.000544 |
| Market-clustered 95% interval | [-0.002730, 0.001721] |
| Seeds individually below market | 4 of 5 |

The interval includes zero. Isotonic calibration wins on validation but loses to the
market on test. Selecting Platt L2 after inspecting all seven test rows is itself
selection. The canonical verdict is **{{C2_BRIER.verdict}}**. A confirmatory claim would
need one frozen mapping, unseen receive-time data, and many more independent days.

## 4. Strategy search and the disappearing profit

The stored combined backtest reports 706 trades, +214.1461, and daily Sharpe 0.4085 over
32 available days. These values are internally consistent with the retained lock, but the
strategy was chosen after extensive search.

The original anti-snooping matrix contains 846 candidate daily PnL series. Its stored
report says p=0.8355, a digit that a fresh current-method run does not reproduce. More
importantly, the matrix omits the combined strategy that became the reported winner. The
public correction appends that series, recenters all 847 candidates, and draws 20,000
stationary bootstrap day paths with mean block length three.

[[SELECTION_FIGURE]]

| Corrected audit quantity | Value |
|---|---:|
| Candidates | 847 |
| Days | 32 |
| Observed best mean daily PnL | 6.6921 |
| White's Reality Check p-value | 0.5667 |
| Reject at 10%? | No |

The individual locked early and late rules rank fifth and sixteenth in the old matrix; a
different all-day early rule ranks second. The reported combined winner is not unusual
relative to the best outcome expected from the searched family. Its verdict is
**{{C3_LOCKED_BACKTEST.verdict}}**, even before the causal-clock problem is considered.

## 5. Execution and prospective operation

The historical execution audit does not rescue the strategy. It chooses books under
source time, drops 20 missing books, includes 72 of 162 observations older than ten
seconds, and reaches a maximum age of 62.497 seconds. Its adverse-fill stress changes the
trade count from 706 to 722 instead of holding decisions fixed. Queue position,
self-impact, and quote response are not modeled. The classification is
**{{C5_EXECUTION.verdict}}**.

A separate five-day post-lock artifact reports 262 offline-replay trades and +23.90, with
a 90% day-bootstrap interval of [-15.95, 60.36]. The interval includes zero and the same
source-time episodes determine its entries. It is an
**{{C6_POST_LOCK_REPLAY.verdict}}**, not a live forward score.

The strongest prospective evidence is operational rather than financial. Hashed local
shadow logs from July 12-13 contain 45,436 decisions: 164 entries, 26,691 no-edge skips,
and 18,581 source-health skips. Sanitized entry rows and complete denominators are public.
They show that the guarded system ran and often refused to act. Local timestamps are not
independent timestamps, and no fill or PnL conclusion follows.

[[PAGEBREAK]]

## 6. Reproducibility and provenance

The public release is deliberately compact. From a fresh clone, a reader can recompute
all seven Brier rows and cluster intervals, rerun the corrected 847-candidate selection
audit, load all five checkpoints on real-shaped inputs, and verify that the claim register
and PDF match their numerical sources.

| Evidence level | Public material | Supported conclusion |
|---|---|---|
| Direct | Code, tests, predictions, checkpoints | Score and mini-inference consistency |
| Direct | 847 candidate-day returns and null draws | Negative selection verdict |
| Bound | Split and private-array SHA-256 | Named-input traceability only |
| Local | 34 commit records and hashed July logs | Iterative development and operation |
| Missing | Raw corpus and receive-time retraining | No causal performance conclusion |

The private-source history has 34 local commits from May 20 through June 29. It records
positive claims, reversals after audit, and repeated bot-safety fixes. The July/August TCN
and final audit phase was not committed there. Accordingly, chronology is
**{{C7_CHRONOLOGY.verdict}}**, not externally authenticated registration.

## 7. Conclusion

The apparent strategy profit disappears as evidence in three stages. Dependence-aware
forecast uncertainty includes zero. The selected winner is ordinary relative to 847
searched candidates. The source-time replay cannot establish what was knowable live, and
the execution audit is invalid.

The final verdict is **{{C10_OVERALL.verdict}}**. This does not prove that prediction
markets are perfectly efficient or that no causal signal could exist. It shows that this
experiment did not establish one. The next experiment must use receive-time state,
externally freeze one policy before new data, retain complete decision logs, and evaluate
decision-matched execution.

The result I would carry forward is methodological: a smaller honest claim, with public
failure analysis and reproducible uncertainty, is stronger research than a profitable
number whose clock and selection process cannot support it.

## References and artifact index

- Bailey, D. and Lopez de Prado, M. (2014), *The Deflated Sharpe Ratio*.
- Politis, D. and Romano, J. (1994), *The Stationary Bootstrap*.
- White, H. (2000), *A Reality Check for Data Snooping*.
- Forecast evidence: brier_summary.json.
- Selection evidence: selection_audit.json.
- Reproduction evidence: evaluation_repro_v2/manifest.json.
- Canonical evidence: publication_claims.json.
