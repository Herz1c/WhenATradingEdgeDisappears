# When a Trading Edge Disappears

[![tests](https://github.com/Herz1c/BitcoinMicrostructureResearch-public/actions/workflows/tests.yml/badge.svg)](https://github.com/Herz1c/BitcoinMicrostructureResearch-public/actions/workflows/tests.yml)

An independent high school research project on Bitcoin five minute prediction markets,
conducted from April through August 2026. I built a recording and modelling system across
several venues, found an apparently profitable strategy, and then investigated why that
result should not be trusted.

**Final result: no demonstrated trading edge.** The model's best observed test Brier
score is only 0.000544 below the market baseline, with a market clustered 95% interval
of **[-0.002730, 0.001721]**. The selected historical strategy fails a corrected
847 candidate White's Reality Check (**p = 0.5667**). More importantly, the historical
episodes replay venue source timestamps instead of recorder receive timestamps, so the
positive backtest is a zero latency counterfactual not live achievable evidence.

That negative conclusion is the central contribution. The project became a case study in
causal clocks, dependent observations, model selection bias, execution realism, and the
difference between running a system and validating a claim.

## My contribution

I conceived and directed the research: the question, hypotheses, source and dataset choices,
causal rules, experiment sequence, audit design, failure criteria, safety gates, and final
interpretation. I also operated the recorders and experiments. AI tools supported much of
the software, test, and report implementation.

My complete contribution statement and supporting record are in
[INTELLECTUAL_OWNERSHIP.md](INTELLECTUAL_OWNERSHIP.md). The questions I invite a reviewer
to ask me are in [TECHNICAL_DEFENSE.md](docs/TECHNICAL_DEFENSE.md).

## Where to read next

| Document | What it covers |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | Every gate, its verdict, and why the edge failed |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Dataset, splits, model, and statistical procedure |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The recording, dataset, and modelling tiers |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | What a fresh clone can and cannot check |
| [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) | What is published, what is withheld, and why |
| [docs/known_failure_modes.md](docs/known_failure_modes.md) | Failures I found and how they were caught |
| [docs/decision_log.md](docs/decision_log.md) | The contemporaneous record of research decisions |
| [paper/report.pdf](paper/report.pdf) | The written paper, built from `paper/manuscript.md` |

## What I built

- Recorders and canonical pipelines for Polymarket, Binance, Coinbase, Bybit,
  Hyperliquid, Deribit, and Chainlink related feeds.
- A 5 Hz episode representation with 69 features, validity masks, audit channels, and
  calendar day train/validation/test splits.
- A causal convolution TCN that predicts a bounded residual over the market logit.
- Backtest, shadow decision, execution, walk forward, deflated Sharpe, and
  multiple testing audit tooling.
- A compact public evidence bundle: all five checkpoints, validation/test prediction
  arrays, a ten market inference fixture, candidate day PnL, split hashes, and sanitized
  shadow decisions.

The private inventory contains 229.8 GB and 119,665 compressed raw files across 76
recorded days. Those totals are a locally measured inventory claim, the private corpus is
not distributed. The public release is about 18 MiB and publishes the evidence needed to
check the central negative conclusion.

## How the repository is laid out

| Directory | What it is |
|---|---|
| `src/` | The maintained stack: recorders, canonical pipeline, models, execution, shadow bot |
| `tests/` | 292 tests over that stack |
| `tools/` | 101 standalone runners: dataset builders, backtests, and the audits |
| `research/` | 85 earlier phase scripts, kept for provenance and not maintained |
| `artifacts/` | The public evidence bundle |
| `docs/`, `paper/` | Writeup, methodology, and the claim register |

`tools/` and `research/` are deliberately large, and that is the point rather than an
accident. Every threshold sweep, sizing scheme, and regrade in those directories is one
more specification I tried, and the multiple testing audit in
[RESULTS.md](docs/RESULTS.md) exists precisely because I tried that many. A reader who
wants to know how wide the search was should read those directories as the search record;
[research/README.md](research/README.md) and [tools/README.md](tools/README.md) index them
script by script. Publishing the search is what makes the 847 candidate correction
meaningful instead of a number I assert.

## Results that can be checked here

### Forecasting

The public arrays contain 96,494 eligible test rows from 388 markets across five days.
All seven calibration variants are recomputed from the released predictions.

| Quantity | Value |
|---|---:|
| Lowest observed test Brier (post hoc, Platt L2) | 0.127732 |
| Market Brier | 0.128276 |
| Model minus market | -0.000544 |
| Market clustered 95% interval | [-0.002730, 0.001721] |
| Calibration maps inspected | 7 |

The interval crosses zero. Five seeds share the same events, five test days are too few
for strong day level inference, and the dataset clock is noncausal. This is a small
exploratory observation, not evidence of superior forecasting.

### Strategy selection

The stored combined backtest reports 706 trades and +214.15 over 32 available days. The
original audit matrix held 846 candidates but omitted that combined reported winner. The
public reproduction appends it, tests all 847 series with a 20,000 replicate stationary
bootstrap, and obtains p=0.5667. The apparent profit does not survive selection
adjustment and remains source time counterfactual even before that correction.

### Prospective operation

Sanitized local shadow logs preserve 45,436 decisions from July 12-13: 164 `ENTER`,
26,691 `SKIP_NO_EDGE`, and 18,581 `SKIP_SOURCE`. They show that the guarded system ran
and frequently refused to act. The logs are locally dated, not externally timestamped,
and prove neither fills nor PnL.

## Reproduce the public evidence

With Python 3.11+:

```bash
py -m pip install -e ".[dev,paper]"
py tools/reproduce_public_evidence.py --check
py -m pytest tests -q
py tools/audit_public_release.py
py tools/build_publication_claims.py --check
py tools/build_public_paper.py --check
```

To rerun the released neural network checkpoints, install the optional CPU compatible
PyTorch extra and run:

```bash
py -m pip install -e ".[sequence]"
py tools/reproduce_tcn_mini.py
```

The first command recomputes Brier scores, clustered intervals, and the corrected
selection audit from public arrays. The mini check reruns five checkpoints on ten complete
markets of 1,500 steps each and compares 2,586 eligible rows per seed. Raw data reconstruction,
training, and a receive time corrected experiment remain unavailable.

## Evidence ladder

| Level | Public support | What it establishes |
|---|---|---|
| A | Source code and 292 tests | Software behavior on fixtures |
| B | Predictions, checkpoints, calibration maps | Model score and mini inference reproduction |
| C | 847 candidate day series and bootstrap output | The central selection bias verdict |
| D | Split hashes and local Git metadata | Traceability to the private source; not external attestation |
| E | Sanitized prospective decision rows | Shadow operation only; not performance |
| Missing | Raw corpus and receive time retraining | No live causal performance claim |

