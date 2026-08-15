# Canonical public claim register

> This file is generated from `artifacts/publication_claims.json`. Edit the JSON
> and run `py tools/build_publication_claims.py`; do not hand-edit this file.

Public predictions, checkpoints, candidate-day returns, split hashes, sanitized shadow rows, source code, and generated audits are directly inspectable. Raw-data reconstruction, retraining, and receive-time correction remain private-data dependent.

| ID | Claim | Verdict |
|---|---|---|
| `C1_DATASET_CLOCK` | Historical episode-model and strategy evaluations use source-time replay. | **SOURCE-TIME COUNTERFACTUAL** |
| `C2_BRIER` | The lowest observed calibration row has a slightly lower test Brier score than the market baseline. | **EXPLORATORY; IMPROVEMENT NOT DEMONSTRATED** |
| `C3_LOCKED_BACKTEST` | The stored combined strategy reports +214.15 over 706 source-time trades. | **INVALID AS EDGE EVIDENCE** |
| `C4_OUTPUT_CONSISTENCY` | The released score and selection artifacts can be recomputed from public inputs. | **PUBLIC PARTIAL REPRODUCTION: PASS** |
| `C5_EXECUTION` | The historical execution audit establishes fillability or capacity. | **UNRESOLVED / INVALID AUDIT** |
| `C6_POST_LOCK_REPLAY` | The five-day post-lock score is causal live evidence. | **INCONCLUSIVE SOURCE-TIME OFFLINE POST-LOCK REPLAY** |
| `C7_CHRONOLOGY` | Local history documents iterative development but externally authenticates the full headline chronology. | **LOCAL PROVENANCE; NOT INDEPENDENTLY TIMESTAMPED** |
| `C8_CORPUS` | The private inventory contains 229.8 GB across 76 recorded days. | **COMMITTED INVENTORY CLAIM; PUBLIC RECOUNT UNAVAILABLE** |
| `C9_SHADOW_OPERATION` | The guarded shadow system operated prospectively on July 12–13. | **LOCAL OPERATION EVIDENCE; NOT PERFORMANCE** |
| `C11_INTELLECTUAL_OWNERSHIP` | Jan Herza conceived and directed the research program. | **AUTHOR-CONCEIVED AND DIRECTED** |
| `C10_OVERALL` | This repository demonstrates a tradeable forecasting or execution edge. | **NO DEMONSTRATED TRADING EDGE** |

## C1_DATASET_CLOCK: SOURCE-TIME COUNTERFACTUAL

The historical episode dataset replays Polymarket and centralized-exchange observations at venue source time rather than recorded receive time. Model scores, backtests, and offline post-lock results derived from it are zero-latency counterfactual diagnostics, not live-achievable performance evidence.

Public evidence:

- `tools/build_btc_5m_episode_dataset.py`
- `docs/METHODOLOGY.md`

Limits:

- The raw corpus required for a receive-time rebuild is not public.
- No receive-time retraining result exists.

## C2_BRIER: EXPLORATORY; IMPROVEMENT NOT DEMONSTRATED

After inspecting seven calibration maps, Platt-L2 has the lowest observed five-seed mean test Brier: 0.127732 versus 0.128276 for the market. The paired 388-market bootstrap interval for model minus market is [-0.002730, 0.001721], which includes zero.

Public evidence:

- `artifacts/tcn_v2_eval/brier_summary.json`
- `artifacts/evaluation_repro_v2/predictions/`
- `tools/reproduce_public_evidence.py`

Limits:

- All seven test rows were inspected and the minimum is post hoc.
- The five seeds share events and are not independent replications.
- Only five test days are available.
- The underlying episodes are source-time replay.

## C3_LOCKED_BACKTEST: INVALID AS EDGE EVIDENCE

The stored total is internally consistent, but the result is selected and source-time counterfactual. A corrected public White's Reality Check appends the combined winner to the 846-candidate matrix and obtains p=0.5667 across 847 candidates, so it does not reject the no-edge null.

Public evidence:

- `artifacts/backtest_repro_v1/summary_test_cap_drop.json`
- `artifacts/audit_v1/wrc_universe_daily_pnl.npz`
- `artifacts/audit_v2/selection_audit.json`
- `tools/reproduce_public_evidence.py`

Limits:

- The 847 candidates remain a lower bound on the true number of trials.
- The original stored p=0.8355 is retained only as historical provenance.
- The historical PnL series use source-time books.

## C4_OUTPUT_CONSISTENCY: PUBLIC PARTIAL REPRODUCTION: PASS

A fresh clone can recompute seven Brier rows, paired clustered intervals, the corrected 847-candidate selection audit, and mini inference for five checkpoints. Raw-data reconstruction, training, the complete historical backtest, and receive-time correction are unavailable.

Public evidence:

- `artifacts/evaluation_repro_v2/manifest.json`
- `tools/reproduce_public_evidence.py`
- `tools/reproduce_tcn_mini.py`
- `tests/test_published_claims.py`

Limits:

- The raw corpus and full episode tensors are private.
- Checkpoint parity is a ten-market fixture, not full retraining.

## C5_EXECUTION: UNRESOLVED / INVALID AUDIT

The stored audit uses source-time books, drops missing observations, admits books up to 62.497 seconds old, and changes the decision set under adverse-fill stress. It supports no fillability or capacity verdict.

Public evidence:

- `artifacts/audit_v1/execution_realism_shadow.json`
- `artifacts/backtest_repro_v1/summary_test_fill_worse.json`
- `tools/audit_execution_realism.py`

Limits:

- Twenty missing books are dropped.
- Seventy-two of 162 inspected books are more than ten seconds old.
- Queue position, self-impact, and counterfactual quote response are absent.

## C6_POST_LOCK_REPLAY: INCONCLUSIVE SOURCE-TIME OFFLINE POST-LOCK REPLAY

The stored five-day score is an offline replay over source-time episodes: 262 trades, +23.90, with a 90% day-bootstrap interval of [-15.95, 60.36]. The interval includes zero and the score is not derived from complete prospective logs.

Public evidence:

- `artifacts/tcn_shadow_parity/shadow_pnl_summary.json`
- `artifacts/tcn_shadow_parity/shadow_pnl_2026-07-04.json`

Limits:

- The replay uses the non-causal episode clock.
- The stated duration and interval conditions were not met.

## C7_CHRONOLOGY: LOCAL PROVENANCE; NOT INDEPENDENTLY TIMESTAMPED

Thirty-four local commits document development from May 20 through June 29 and hashed July logs document shadow operation. Earlier April/May decision and audit records corroborate a student-led causal and adversarial research program. The final TCN and audit phase is uncommitted in that history, and no local metadata is treated as a signed or external timestamp.

Public evidence:

- `artifacts/provenance/source_git_history.json`
- `artifacts/live_log_sample_v1/summary.json`
- `docs/PROVENANCE.md`

Limits:

- Commit metadata was exported from a local private workspace.
- July/August headline work lacks commit-level provenance.

## C8_CORPUS: COMMITTED INVENTORY CLAIM; PUBLIC RECOUNT UNAVAILABLE

A local inventory reports 246,718,377,000 bytes, 119,665 files, and 76 recorded days out of 113 calendar days. The private roots are absent, so the counts cannot be independently recounted here.

Public evidence:

- `artifacts/data_inventory.json`

Limits:

- Thirty-seven calendar days are missing.
- The raw corpus is not public.

## C9_SHADOW_OPERATION: LOCAL OPERATION EVIDENCE; NOT PERFORMANCE

Hashed local logs record 45,436 decisions, including 164 entries, while source-health and no-edge guards rejected most snapshots. This demonstrates local shadow operation, not fills, PnL, or a trading edge.

Public evidence:

- `artifacts/live_log_sample_v1/summary.json`
- `artifacts/live_log_sample_v1/enter_decisions_2026-07-12.jsonl`
- `artifacts/live_log_sample_v1/enter_decisions_2026-07-13.jsonl`

Limits:

- Only sanitized entry rows are public; complete counts remain in the summary.
- The logs are locally dated and not independently timestamped.

## C11_INTELLECTUAL_OWNERSHIP: AUTHOR-CONCEIVED AND DIRECTED

I conceived and directed the research question, hypotheses, source and dataset choices, causal rules, experiment sequence, audit design, failure criteria, safety gates, and interpretation. AI tools supported much of the implementation.

Public evidence:

- `INTELLECTUAL_OWNERSHIP.md`
- `docs/IDEA_PROVENANCE.md`
- `docs/decision_log.md`
- `docs/reports/quant_audit_findings.md`
- `artifacts/provenance/source_git_history.json`

## C10_OVERALL: NO DEMONSTRATED TRADING EDGE

The public evidence does not demonstrate a forecasting, trading, or execution edge. The forecast interval includes zero, the selected strategy fails the corrected multiple-testing audit, the clock is non-causal, and execution remains unresolved.

Public evidence:

- `artifacts/tcn_v2_eval/brier_summary.json`
- `artifacts/audit_v2/selection_audit.json`
- `docs/RESULTS.md`
