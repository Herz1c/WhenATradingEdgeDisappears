# Decision Log

> Contemporaneous record, kept unedited. Backtick paths refer to my private workspace as it
> stood on each date; datasets and reports that are not part of the public release are listed
> in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). Conclusions recorded here were superseded
> by the August audit — [RESULTS.md](RESULTS.md) is canonical.

## 2026-04-28 - Model Factory Sprint 1 Infrastructure Contract

- Task: start Model Factory Sprint 1 as infrastructure-only work after dataset factory production readiness.
- Decision: no model training, policy optimization, paper trading, or live deployment is allowed in this sprint; the deliverables are registry/configs, leak-safe loading, provenance writing, audit, CLI, and unit tests.
- Decision: `model_registry_v1.yaml` uses the current production dataset artifact paths materialized in this repository, such as `data/datasets/market_interval_dataset_v1_preopen`, instead of prompt aliases that point at non-existent nested directories.
- Decision: the current 5-day development window (`2026-04-19` through `2026-04-23`) remains the explicit split fixture, while split configs set `auto_extend=true` so monthly data can reuse the same loader contract without architectural changes.
- Decision: model_09 and model_10 receive explicit placeholder feature configs; the loader must refuse training-mode loads for placeholder configs until their feature definitions and upstream enriched datasets exist.
- Decision: the model factory depends on Polars for the mandated loader API and is packaged as `src/model_factory` so CLI and tests can import the shared infrastructure.
- Evidence: roadmap v2 Section XIV defines the 10-model stack, dataset factory completion report declares current dataset artifacts production-ready for model-building workloads, and the schema contracts under `config/schemas` define leakage-only and training-eligible fields.
- Decision: Model registry v1 created: 10 models registered.
- Decision: LeakageSafeLoader implemented: enforces temporal split, leakage blacklist, eligibility filter, dataset hash.
- Decision: Sprint 1 audit PASS: all loader integration tests OK.
- Evidence: `py -3 -m pytest tests\test_model_factory_loader.py -v` reports `15 passed`; `py -3 -m polymarket_recorder.cli audit-model-factory --output-path .\docs\model_factory_sprint1_audit.md` reports `PASS`, `0 CRITICAL`, `0 WARNING`.

## 2026-04-26 - Chainlink Public Delayed Endpoint Migration

- Task: diagnose why `chainlink_public_delayed/public_stream_page/BTCUSD` rows after `2026-04-22 17:47:30 UTC` stopped producing usable BTC/USD delayed prices.
- Finding: the public Chainlink BTC/USD Data Streams page no longer exposes the old `/api/query-timescale?query=LIVE_STREAM_REPORTS_QUERY` path used by the recorder; that endpoint returns 404 for the current page. The current page bundle requests `/api/live-data-engine-stream-data` with `feedId`, `abiIndex=0`, `queryWindow=1m`, and `attributeName=benchmark`.
- Decision: update `chainlink_public_delayed` capture to prefer the current live data engine endpoint and keep the old query-timescale path only as a compatibility fallback.
- Evidence: `ChainlinkPublicDelayedRecorderService._fetch_record` returns `parse_status=success` against the current public page on 2026-04-26, and the full test suite reports `198 passed`.
- Limitation: stored 2026-04-23..2026-04-26 error-only raw rows do not contain recoverable live tick values. The public historical data engine exposes coarser 1-minute/1-hour/1-day candles, but not the original 2-second polling stream.

## 2026-04-22 - Synthetic BTC/USD Chainlink Proxy

- Task: reconstruct a training-safe and live-safe synthetic BTC/USD Chainlink reference without RTDS.
- Historical decision, superseded on 2026-04-25: select `spot_premium_calibrated_v1` as `synthetic_chainlink_price`.
- Evidence: median abs error 5.2582 USD, p90 14.1803 USD, coverage 98.37% on 16846 Chainlink update events.
- Alternatives considered: `median_all`, `trimmed_mean`, `volume_weighted_mid`, `binance_spot_raw`, `median_spot_only`.
- Constraints respected: finalized-clean only, Binance USD-M quarantine hour excluded, causal recv_ts_ns joins only, delayed Chainlink page used for validation only.

## 2026-04-23 - Chainlink Public Delayed Exact-Value Delay Audit

- Task: measure the delivery delay of `chainlink_public_delayed` relative to `chainlink_onchain` using exact +/- 0.01 USD value matches on finalized-clean data only.
- Decision: do not treat `chainlink_public_delayed` as a clean delayed mirror of the on-chain BTC/USD feed for exact event replay; keep the conservative `1800s` causal availability floor in the dataset factory.
- Evidence: only `2 / 114` on-chain update events produced plausible exact-value matches within the generous `[-600s, +3600s]` recv-time matching window, for just `1.75%` coverage. On the matched subset, p5/median/p95 delay was `35.268s / 236.009s / 436.749s`, min `12.964s`, max `459.054s`, and there were `0` negative-delay matches.
- Alternatives considered: unrestricted exact-value matching across the whole page stream, but that produced obvious same-price recurrence artifacts hours away from the on-chain event and was rejected as non-causal / non-identifiable.
- Recommendation: use `chainlink_public_delay_audit.md` for the detailed tables, and continue using the public page only as a delayed calibration source with a conservative `1800s` floor rather than as an exact delayed copy of the on-chain oracle.

## 2026-04-25 - Official Fake Live Chainlink and delta_to_strike Method

- Task: freeze one official live-safe BTC/USD reference for all `delta_to_strike` features and remove the older median-mids implementation from canonical Phase 2/3 data.
- Decision: `live_reference_events_v1.synthetic_raw` is now spot-only: `binance_spot_mid * 1.00029`. `synthetic_corrected` is `synthetic_raw + mean(chainlink_public_delayed_price - synthetic_raw_at_chainlink_ts)` over delayed Chainlink event timestamps in `[t-60m, t-30m]`. `delta_to_strike` everywhere is defined as `live_reference_events_v1.synthetic_corrected - price_to_beat` via the existing causal as-of join.
- Update: the 2026-04-25 carried-bias decision below preserves this official base formula, but `synthetic_corrected` now applies the last valid active-window residual when the strict residual window is empty and marks those rows with `bias_carried_forward=true`.
- Evidence: `spot_only_synthetic_chainlink_validation_report.md` matched `16846` Chainlink public-page events with `binance_spot_last_recv_ts_ns <= chainlink_display_ts_ns`; `spot * 1.00029 + mean residual [t-60m,t-30m]` achieved median absolute error `3.4989 USD`, p90 `9.2996 USD`, and p95 `12.0212 USD`. With a `spot_staleness <= 2s` cap it improved to median absolute error `3.4636 USD`, p90 `9.1436 USD`, and p95 `11.6543 USD`.
- Constraints respected: calibration observations become usable only after `chainlink_ts + 1800s` and never before local `recv_ts_ns`; the live bot can reproduce the same value online from current Binance spot plus the last 30-minute-old public Chainlink residual window. Binance USD-M and Hyperliquid remain diagnostic context only and do not enter the official price.
- Alternatives considered: using public Chainlink observations with 0-60s delay achieved lower diagnostic median error but was rejected because it violates the current 1800s public-data availability contract; keeping the old spot-vs-USD-M/Hyperliquid basis term was rejected because spot-only is simpler, has better coverage, and measured slightly better under causal tick verification.

## 2026-04-25 - Phase 3 Training Gate Requires Active Chainlink Residual Calibration

- Historical decision, superseded on 2026-04-25 by the carried-bias training contract below.
- Task: prevent Phase 3 rows built from uncalibrated base-proxy fallback from being treated as equally healthy training examples.
- Historical decision: Phase 3 still writes `live_btc_usd` and `delta_to_strike` for valid base-proxy fallback rows, but `training_feature_eligible=true` required `live_reference_events_v1.bias_active=true`. Rows with `price_valid=true` but no active `[t-60m, t-30m]` delayed Chainlink residual were preserved as audit/inference-visible and excluded with `exclusion_reason=live_reference_untrusted`.
- Historical evidence: `2026-04-23` has finalized `chainlink_onchain` verification coverage, but `chainlink_public_delayed` page rows contain no usable `btc_usd_price` values after `2026-04-22 17:47:30 UTC` due HTTP/API errors. Without an active delayed residual window, the Binance spot base proxy can still produce a deterministic price, but it is not freshly calibrated to the target Chainlink stream.
- Alternatives considered: keeping base fallback rows trainable, rejected because it overstates the health of days without active calibration; dropping rows from parquet, rejected because preserving them is useful for audit, inference simulation, and future recalibration.

## 2026-04-25 - Carried Residual Bias and 5-Minute Full-Matrix Training Gate

- Task: recover trainable Phase 2/3 coverage after the `chainlink_public_delayed` historical stream stopped producing usable BTC/USD delayed rows while keeping calibration state explicit and auditable.
- Decision: `live_reference_events_v1.synthetic_corrected` now applies the last valid active-window delayed Chainlink residual when the strict `[t-60m, t-30m]` residual window is empty. `bias_active` remains strict fresh-window truth; carried rows are marked with `bias_carried_forward=true` and `bias_mode=carried_forward`. Phase 3 now defines `live_reference_strict_trusted`, `live_reference_carried_trusted`, and keeps `live_reference_trusted` as a backward-compatible alias for the carried/training-trusted state.
- Decision: broad Phase 3 training uses `training_feature_eligible` with carried-trusted live reference rows. Full-matrix quant training uses `full_matrix_training_eligible`, which additionally requires `complete_feature_matrix_eligible=true` and `market_5m_complete_rate >= 0.80` inside the current 5-minute market and materialized view. No whole-hour completeness filter is used.
- Evidence: the health audit showed valid spot-only synthetic price continued on 2026-04-23, but strict `bias_active` dropped to 0% after the delayed public page stream failure. Simulating last-valid-bias carry-forward lifted Phase 2 trusted coverage from roughly 67.6% to roughly 96.9% and Phase 3 broad training coverage from roughly 67.8% to roughly 97.2%, while the 80% per-market complete-matrix rule targets the remaining token-mid / spot-perp completeness problem at the correct 5-minute granularity.
- Documentation: see `phase2_phase3_carried_bias_contract_2026-04-25.md` for field names, model routing, rebuild commands, and implementation touch points.

## 2026-04-23 - dataset_policy_v1 Freeze

- Task: freeze `dataset_policy_v1.yaml` for Phase 2 canonical event base.
- Decision: `dataset_policy_v1.yaml` is frozen as a role-based, quarantine-driven policy; dataset builds use only `finalized_clean` healthy inputs and preserve quarantined inputs on disk for audit.
- Evidence: [dataset_policy_v1.yaml](../config/dataset_policy_v1.yaml) defines the permanent role boundaries, allowed manifest/file states, quarantine conditions, training eligibility, and reproducibility constraint without enumerating temporary unhealthy slices in policy.
- Alternatives considered: embedding current bad slices directly into policy, rejected because the Phase 2 contract requires health-driven eligibility and explicit quarantine preservation rather than hardcoded source bans.

## 2026-04-23 - Scripture-Defined Binance USD-M Quarantine Enforcement

- Task: enforce the roadmap-defined bad Binance USD-M hour while keeping `dataset_policy_v1.yaml` in the exact structure requested for Phase 2.
- Decision: the explicit `2026-04-20 07:00 UTC` Binance USD-M quarantine is enforced as a scripture-backed source-specific integrity check in the policy loader / eligibility layer, not as an extra YAML key.
- Evidence: roadmap rule II.6 and the audit scriptures require that hour to stay out of training; the exact YAML structure I specified for `dataset_policy_v1.yaml` has no dedicated field for explicit excluded hours, so enforcement lives in `src/market_recorders/dataset_policy.py` and is surfaced through quarantine accounting in the canonical build.
- Alternatives considered: extending the YAML schema with an explicit excluded-slice list, which I rejected because I had specified that this task write the provided structure and nothing more.

## 2026-04-23 - delta_to_strike Canonical Reconnection

- Task: reconnect `delta_to_strike` to the Phase 2 canonical live reference layer.
- Decision: `delta_to_strike` now reads `synthetic_chainlink_v1` from `live_reference_events_v1` via causal as-of lookup on `ts_seconds`; dense price references no longer flow directly from RTDS, verification-only Chainlink layers, or ad-hoc exchange mids in the strike delta path.
- Evidence: [strike_delta.py](../src/polymarket_recorder/strike_delta.py) now uses [LiveReferenceEventsReader](../src/chainlink_recorder/live_reference_events.py), propagates `price_unavailable=True` and `NaN` deltas when `price_valid=False`, and the updated tests in [test_polymarket_strike_delta.py](../tests/test_polymarket_strike_delta.py) pass against the canonical parquet.
- Alternatives considered: retaining direct fallback to raw exchange mids or old best-available Chainlink proxy readers, rejected because Phase 2 requires all dense truth-like price access to flow through `live_reference_events_v1`.

## 2026-04-23 - live_reference_events_v1 Daily Shard Layout

- Task: replace the initial monolithic `live_reference_events_v1.parquet` artifact with a daily canonical layout better suited for incremental generation and quarantine-aware rebuilds.
- Decision: the canonical source of truth is now the directory `data/canonical/live_reference_events_v1` containing one parquet shard per UTC day; the previous merged single-file artifact was removed.
- Evidence: the builder in [live_reference_events.py](../src/chainlink_recorder/live_reference_events.py) now writes `YYYY-MM-DD.parquet` shards, the reader loads the directory as one logical causal source, and the regenerated shards currently cover `2026-04-19` through `2026-04-23`.
- Alternatives considered: keeping a single merged parquet as the canonical artifact, rejected because the workflow generates data incrementally and needs day-level rebuilds, day-level quarantine visibility, and easier auditability.

## 2026-04-23 - Phase 3 Sampling Policy Freeze

- Task: freeze the first supervised snapshot sampling policy for Phase 3 dataset factory.
- Decision: `resolution_snapshot_dataset_v1_coarse` uses fixed 1s snapshots from market open through `T-60s`; `resolution_snapshot_dataset_v1_dense_close` uses fixed 250ms snapshots from `T-60s` to `T-10s` and fixed 100ms snapshots from `T-10s` to strictly before close. Event-triggered expansion is deferred and raw event streams remain on disk for later augmentation.
- Evidence: the roadmap defines this exact multi-resolution progression as the recommended v1 sampling policy; the implemented factory writes both coarse and dense_close datasets using only pre-close causal timestamps.
- Alternatives considered: monolithic 1s sampling for the entire market, rejected because the roadmap explicitly warns against downsampling away close microstructure.

## 2026-04-23 - Phase 3 Canonical Label Policy

- Task: freeze deterministic label pairing for Phase 3 supervised datasets.
- Decision: the dataset factory uses the earliest `label_safe=true` finalized-clean strike row as the canonical `price_to_beat` anchor per market slug, and includes a market in supervised datasets only if a matched finalized-clean canonical resolution row exists within the allowed archive window.
- Evidence: this matches the scripture requirement to train only on `label_safe` strikes with matched canonical resolution rows. The current Phase 3 rebuild includes finalized `2026-04-23` inputs where the label and quality gates pass.
- Alternatives considered: using later duplicate strike rows or unresolved closed markets, rejected because they weaken causality and violate the training label gate.

## 2026-04-23 - Phase 3 Polymarket Microstructure Source Choice

- Task: choose the canonical Polymarket event source for Phase 3 v1 token pricing and order-flow features.
- Decision: Phase 3 v1 uses finalized-clean Polymarket `.l2` as the canonical microstructure source, because it already materializes book snapshots, reconstructed price changes, top-of-book updates, and trade execution events per token. Raw `.ws` remains preserved for audit and later event-triggered expansion.
- Evidence: the `.l2` archive exposes `l2_book_state`, `l2_top_of_book`, and `trade_execution` records with token-level market context, making it sufficient for the v1 snapshot datasets while staying inside the scripture-backed healthy input set.
- Alternatives considered: joining raw `.ws` in parallel for Phase 3 v1, rejected as unnecessary duplication for the first canonical dataset factory pass.

## 2026-04-23 - Roadmap v2 Source Stack Update

- Task: update `roadmap_btc_5m_updown.txt` before the Phase 2 / Phase 3 rebuild so the roadmap reflects the current approved stack and canonical live-reference schema.
- Decision: `roadmap_btc_5m_updown.txt` is now `v2`; the source stack was corrected to Polymarket + Binance spot + Binance USD-M + Hyperliquid with delayed Chainlink as calibration-only, and `live_reference_events_v1` was rewritten to the current synthetic Chainlink schema with `recv_ts_ns`, rolling bias fields, source mids, staleness, and `max_cross_source_spread`.
- Evidence: the roadmap header, Rule II.7, Section III source tiers, Section VI canonical table definitions, and Section XVII anti-pattern list now match the current Phase 2/3 implementation target and the updated exclusion policy.
- Alternatives considered: leaving the old roadmap wording in place and relying on decision-log drift, rejected because the rebuild must be gated by a scripture that matches the code and audit reality.

## 2026-04-23 - Roadmap Exclusion Wording Kept Machine-Verifiable

- Task: satisfy the roadmap update while also enforcing the requested grep gate that permanently excluded sources appear only in the exclusion / anti-pattern sections.
- Decision: excluded source names were confined to the dedicated `PERMANENTLY EXCLUDED` block and the explicitly allowed anti-pattern lines, while later sections use generic wording such as `approved external BTC .ws streams` and `excluded proxy path` instead of repeating excluded-source names.
- Evidence: the post-edit grep scan for `Bybit`, `FNG`, `RTDS`, `Deribit`, `Coinbase`, `btc_usd_price`, `chainlink_ts / chainlink_ts_ms`, and `live_reference_quality` now hits only the allowed exclusion / anti-pattern areas or returns zero for removed legacy schema fields.
- Alternatives considered: repeating excluded-source names inside the canonical table descriptions exactly as older text patterns did, rejected because it would make the roadmap fail its own verification gate.

## 2026-04-24 - Phase 2+3 Rebuild Source and Contract Freeze

- Task: rebuild Phase 2 and Phase 3 on the corrected roadmap / source stack and close all known audit findings.
- Historical decision: FNG stays fully excluded from Phase 2/3, Hyperliquid stays retained as diagnostic context in `live_reference_events_v1`, dead Phase 3 columns were removed, YAML schema contracts were written under `config/schemas`, and every Phase 3 row carries an explicit training eligibility contract.
- Current update: the later 2026-04-25 official method makes Binance spot the only `synthetic_raw` price input; Binance USD-M and Hyperliquid remain diagnostics only and do not enter `synthetic_corrected`.
- Evidence: `phase2_completion_report.md`, `phase3_completion_report.md`, and `phase2_phase3_health_audit_2026-04-25.md` show the current rebuilt artifacts, clean post-rebuild audit, and the carried-bias eligibility contract surface.
- Alternatives considered: dropping Hyperliquid entirely, rejected because it remains useful diagnostic context; keeping FNG as slow context, rejected because the audit found it to be non-contributive noise for the 5-minute modeling task.

## 2026-04-24 - Phase 3 Market-Level No-Live Exclusions Stored Separately

- Task: encode the hard market-level case where an entire market has no valid live reference rows, while preserving the row-level exclusion priority requested for Phase 3.
- Decision: row-level `exclusion_reason` keeps the strict priority on immediate row failures such as `live_reference_unavailable`, while the stronger market-level fact `market_fully_without_live_reference` is stored both as a row column and in phase3_market_exclusions.parquet (`data/canonical/quality_registry_v1/phase3_market_exclusions.parquet`) for downstream exclusion / audit workflows.
- Evidence: the clean post-rebuild audit still reports the fully invalid `2026-04-22 19:00 UTC` markets as present in parquet but fully excluded from the training-eligible subset, which matches the intended "flag, do not silently drop" behavior.
- Alternatives considered: overwriting the row-level exclusion priority with a market-level reason on every affected row, rejected because it weakens the more specific per-row failure ordering used by the training contract.

## 2026-04-24 - Phase 1-3 Audit Stop Condition: Binance USD-M Quarantine Representation Mismatch

- Task: run the full Phase 1-3 audit after extending Phase 1-3 outputs through `2026-04-23`.
- Decision: stop the audit with a `FAIL` verdict before Phase 4 readiness because the explicit Binance USD-M quarantine hour does not satisfy the audit contract's required proof via `binance_usdm_staleness_s > 30`.
- Evidence: in the `2026-04-20 07:00 UTC` to `07:59:59 UTC` window inside `live_reference_events_v1`, `binance_usdm_mid` is null on `3600 / 3600` rows and `source_count=2` on `3600 / 3600` rows, but `binance_usdm_staleness_s > 30` occurs on `0 / 3600` rows because the column is null rather than stale-valued. This fails the older audit brief's hard stop-condition even though the source is excluded from diagnostic context for that hour.
- Alternatives considered: treating the current null representation as implicitly equivalent to stale exclusion and continuing the audit, rejected because the audit brief explicitly names the Binance USD-M quarantine reflection as a stop gate and the canonical representation must match the contract before Phase 4 proceeds.

## 2026-04-25 - Phase 8 Microstructure Sequence Dataset Build

- Task: build the dataset factory component of Phase 8 for Model 7 and Model 8 without training models.
- Decision: `microstructure_sequence_dataset_v1` is built from Phase 3 coarse + dense_close anchors, expanded to UP/DOWN token samples through `token_registry_v1`, and materialized as tabular/event64/event128 daily parquet shards.
- Decision: tensor export is deferred until the parquet sequence contract is audited on the current five-day window.
- Decision: `sequence_length_actual` is stored as `int16` instead of signed `int8` because event128 requires the value `128`, which signed int8 cannot represent.
- Decision: Phase 8 inherits Phase 3 carried-trusted training eligibility instead of re-introducing a strict fresh-bias age gate; carried-vs-fresh state remains available through `live_bias_mode` and `live_applied_bias_age_seconds`.
- Decision: `mid_return_1s`, `mid_return_3s`, and `mid_return_5s` are absolute token-mid changes. Large jumps with `abs(value) >= 0.5` are retained as trainable BTC 5m UP/DOWN tail events and reported by audit as warnings rather than exclusions.
- Complete: Phase 8 dataset build complete. `microstructure_sequence_dataset_v1` built for 2026-04-19 to 2026-04-23. Anchor sources: Phase 3 coarse + dense_close snapshots. Sequence lengths: 64, 128. Tensor export deferred. Models 7 + 8 to be trained when sufficient data accumulated. Date: 2026-04-25.
- Evidence: see `docs/phase8_microstructure_dataset_factory_report.md` and the daily `docs/phase8_microstructure_audit_YYYY-MM-DD.md` reports.

## 2026-04-26 - Phase 8 Post-Audit Dataset Contract Fixes

- Task: fix Phase 8 `microstructure_sequence_dataset_v1` after the full audit found real target bugs and variant contract mismatches.
- Decision: MFE/MAE are clamped to non-negative excursions in both fast-index and fallback target computation paths; root cause was using raw signed extrema directly, which made all-down or all-up future windows produce negative favorable/adverse excursion values.
- Decision: target availability now requires `t_to_close_s > horizon_s` and a non-NaN `mid_move`; when unavailable, `mid_move_*` is forced to `NaN` and `mid_move_sign_*` to `0`.
- Decision: Option A chosen for variant separation. `tabular` stores only aggregated tabular/context/target/eligibility columns for LightGBM-style models, while `event64` and `event128` store those columns plus sequence arrays for deep sequence models.
- Decision: `pad` is a documented valid `events_event_type` sentinel for prepended padding positions. Sequence models must mask or learn to ignore padding positions.
- Decision: `sequence_id` is standardized across `tabular`, `event64`, and `event128` by removing sequence length from the ID hash; sequence length is represented by the variant path, not by the sample identity.
- Evidence: post-fix unit tests cover target boundary behavior, non-negative excursions, and cross-length `sequence_id` stability.

## 2026-04-26 - Phase 8 Dataset Factory Post-Fix Completion

- Task: rebuild and re-audit Phase 8 `microstructure_sequence_dataset_v1` after applying post-audit fixes.
- Decision: Phase 8 dataset factory is complete after rebuilding `tabular`, `event64`, and `event128` for 2026-04-19 through 2026-04-23 with parallel per-day workers.
- Evidence: `docs/phase8_microstructure_audit_post_fix_2026-04-19.md` through `docs/phase8_microstructure_audit_post_fix_2026-04-23.md` all report `critical_count=0`; `docs/phase8_full_audit_post_fix_2026-04-19_to_2026-04-23.md` reports `PASS_WITH_WARNINGS` with `critical_count=0`.
- Complete: Phase 8 dataset factory complete, all critical audits clean. Date: 2026-04-26.

## 2026-04-26 - Phase 5 Execution Simulator V1 Completion

- Task: implement the roadmap Phase 5 execution simulator layer without generating Phase 6 training datasets or training models.
- Decision: `execution_simulator_v1` is implemented as a causal `recv_ts_ns` replay layer over finalized-clean Polymarket L2 events, with strict as-of book state before `order_ts_ns`, post-only crossing rejection, cancel-before-fill ordering, configurable fee/rebate economics, side-signed markouts, and an inventory ledger.
- Decision: queue model v1 is configurable but defaults to zero queue-ahead in validation because the normalized L2 contract currently exposes total depth, not price-level queue position; this prevents inventing false precision and leaves queue sensitivity for Phase 6 dataset expansion.
- Decision: Phase 5 writes only simulator validation artifacts and smoke results. `execution_intent_dataset_v1`, fill probability models, adverse selection models, and policy logic remain Phase 6+ work.
- Evidence: `docs/simulator_validation_report_v1.md` reports `PASS`, synthetic scenario checks all pass, real-data smoke validation across 2026-04-19 through 2026-04-23 reports 0 CRITICAL findings, and `docs/phase5_execution_results_audit.md` reports 0 CRITICAL findings on the saved smoke result parquet.
- Complete: Phase 5 execution simulator V1 complete. Date: 2026-04-26.

## 2026-04-26 - Phase 5 Audit Interpretation of V1 Assumptions

- Task: resolve audit ambiguity before judging whether the completed Phase 5 simulator is ready to generate `execution_intent_dataset_v1`.
- Decision: the v1 assumptions documented in the Phase 5 completion entry and `simulator_validation_report_v1` are treated as implementation disclosures, not as waivers of the roadmap's Phase 5/Phase 6 readiness requirements. Missing or incomplete replace handling, fee schedule lookup, deterministic tie-breaking, live-reference decision-time context, partial-fill residual handling, or validation coverage must still be reported as audit findings when they affect causal, economic, maker-first, or downstream dataset safety.
- Evidence: roadmap Section XII requires replay orderbook, raw event timeline, as-of external BTC context, post-only quoting, cancel/replace logic, maker/taker economics, fee lookup, rebate accounting, inventory accounting, markout computation, and close-to-resolution PnL before execution-aware ML. The Phase 5 completion note explicitly says Phase 6 dataset generation has not started, so this audit must distinguish reproducible simulator smoke validation from Phase 6 readiness.
- Alternatives considered: treating the stored `PASS` report as sufficient evidence of readiness; rejected because the audit brief requires adversarial verification and the roadmap says the simulator is a mandatory precondition for execution-aware ML.

## 2026-04-26 - Phase 5 Simulator Hardening Contract

- Task: fix the Phase 5 audit findings without inventing Phase 6 policy/model behavior.
- Decision: replay event ordering is canonicalized by `recv_ts_ns` plus deterministic metadata/content tie-break keys. When raw provenance is available, normalized L2 rows carry `source_file_path`, `source_file_index`, and `book_state_seq`; when provenance is absent, a content hash is used so equivalent event sets replay identically regardless of input row order.
- Decision: replace v1 is modeled as cancel-old-then-submit-new at `replace_request_ts_ns + cancel_latency_ns + replace_latency_ns`. The replacement order is a new post-only order with its own price/size terms; old-order fills before replace effective time remain valid, and no old-order fill can occur after replace effective time.
- Decision: partial fills remain live until full size, cancel, replace, or close. The simulator accumulates multiple fill slices into one result row, records first/last fill timestamps, and exposes `fill_count`; ledger/PnL use cumulative filled size.
- Decision: Phase 5 fee lookup uses a versioned static maker-first fee schedule config (`FeeSchedule`) with maker/taker bps and rebate bps. This is not a venue API integration; it is the auditable Phase 5 lookup surface that Phase 6 can replace or enrich when a real schedule is frozen.
- Decision: Phase 5 attaches causal `live_reference_events_v1` state at `order_ts_ns` as decision-time context. Missing or invalid official reference rows make the intent policy-excluded rather than falling back to RTDS, Chainlink onchain/public page, or ad-hoc exchange mids.
- Decision: close behavior for v1 is half-open for resting-order fills: executable trades must satisfy `order_ts_ns < recv_ts_ns < market_close_ts_ns`. Settlement/PnL may use canonical resolution after close, but resting execution does not fill at the exact close timestamp.
- Evidence: these decisions directly address the 2026-04-26 Phase 5 audit findings while preserving the roadmap constraints: `recv_ts_ns` alignment, maker-first behavior, finalized/policy-compliant inputs, official dense BTC reference only, fee-adjusted/fill-adjusted evaluation, and no Phase 6 model/policy implementation.

## 2026-04-26 - Phase 5 Post-Audit Hardening Completion

- Task: complete the Phase 5 fixes requested after the full audit and rerun validation/audit.
- Decision: Phase 5 is promoted back to ready status after implementing deterministic replay tie-breaks, cumulative partial fills, replace lifecycle, causal live-reference context, static versioned fee schedule lookup, non-zero fee/rebate validation, and ledger reconciliation.
- Evidence: full test suite reports `198 passed`; `docs/simulator_validation_report_v1.md` reports `PASS`; regenerated validation report and smoke parquet artifacts match byte-for-byte in `.tmp`; `docs/phase5_execution_results_audit.md` reports `920` rows, `304` fills, `0` critical findings, and `0` warnings; `phase5_completion_audit_report.md` now reports `PASS`.
- Remaining recommendation: materialize full historical `polymarket_l2_events_v1` shards as a storage/backfill task for faster future validation, and replace the static Phase 5 fee schedule values when a real venue fee schedule is frozen.

## 2026-04-26 - Phase 6 Execution Intent Dataset V1 Factory Contract

- Task: build all Phase 6 dataset-factory artifacts for `execution_intent_dataset_v1` without training fill, markout, adverse-selection, policy, paper, or live models.
- Decision: `execution_intent_dataset_v1` is generated only by the Phase 5 simulator from `microstructure_sequence_dataset_v1_tabular` anchors, policy-filtered Polymarket L2 replay, `live_reference_events_v1`, and canonical registries. It must not introduce a learned policy or any alternative dense BTC reference.
- Decision: the v1 factory materializes deterministic daily parquet shards with explicit build caps (`max_anchors_per_day`, `max_markets_per_day`, `max_intents_per_day`) recorded in the build report. Caps are not a model-training sampling policy; they are a reproducible first materialization guard so the dataset factory can be audited before scaling to a full rebuild.
- Decision: Phase 8 future target columns are not copied into the Phase 6 decision-time feature surface. Execution outcomes produced after order submission remain present only as `leakage_only=true` schema fields.
- Decision: rows are training-eligible for Phase 6 only when the Phase 8 anchor is feature-eligible, the simulator submitted the order, trusted `live_reference_events_v1` context is present at `order_ts_ns`, the replay state is causal and non-stale, and no execution exclusion reason remains.
- Decision: the static Phase 5 fee schedule is inherited as the auditable v1 economics surface until a real venue fee schedule is frozen in a later decision log entry.

## 2026-04-26 - Phase 6 Execution Intent Dataset V1 Completion

- Task: materialize and audit `execution_intent_dataset_v1` without training Phase 6 models.
- Decision: Phase 6 dataset-factory work is complete for the current available 2026-04-19 through 2026-04-23 window. Fill probability, adverse-selection, markout, policy, paper-trading, and live work remain not started.
- Evidence: `data/datasets/execution_intent_dataset_v1/YYYY-MM-DD.parquet` contains 4510 rows across five daily shards and 4434 training-eligible rows. `docs/execution_intent_dataset_v1_build_report.md` records build caps and shard hashes. `docs/execution_intent_dataset_v1_audit_report.md` reports PASS with 0 CRITICAL and 0 WARNING. `docs/execution_intent_dataset_v1_reproducibility_report.md` reports byte-identical rebuild hashes for all five shards. Full test suite reports 198 passed.

## 2026-04-26 - Phase 5/6 Economics Bug Fixes

- Task: fix Phase 5 simulator economics bugs found by the 2026-04-26 execution audit and rebuild `execution_intent_dataset_v1`.
- Decision: Fee formula corrected: pre-2026-03-30 uses rate=0.25 exp=2 peak=1.56%, post-2026-03-30 uses rate=0.072 exp=1 peak=1.80%. All BTC 5min markets 2026-04-19..2026-04-23 use post schedule.
- Decision: Markout corrected: now size-scaled USDC PnL, not per-share. Formula: `(future_mid - fill_price) * size` for buys and `(fill_price - future_mid) * size` for sells.
- Decision: Maker fill fee bug fixed: maker `fee_paid` is now correctly 0.0 in all cases. Maker rebates are computed as 20% of the taker-fee equivalent for the same fill.
- Evidence: corrected code lives in `src/polymarket_recorder/execution_fee_schedule.py`, `src/polymarket_recorder/execution_markout.py`, and `src/polymarket_recorder/execution_simulator.py`. Phase 6 must be rebuilt after this entry so all materialized economics reflect the corrected simulator.

## 2026-04-27 - Dataset Completion Sprint Scope and Tensor Export Requirement

- Task: productionize the existing dataset factory without starting model training, policy optimization, paper trading, or live deployment.
- Decision: Dataset Completion Sprint includes required production tensor export for `microstructure_sequence_dataset_v1`.
- Decision: tensor export is a deterministic training materialization derived from audited `microstructure_sequence_dataset_v1_event64` and `microstructure_sequence_dataset_v1_event128` parquet shards; those parquet shards remain the source of truth for sequence samples, targets, sample identity, and causal event ordering.
- Decision: Phase 6 `execution_intent_dataset_v1` must expose an explicit production/uncapped build mode separate from capped debug/audit materialization. Production mode may not silently keep `max_anchors_per_day`, `max_markets_per_day`, `max_intents_per_day`, or hidden scenario caps.
- Decision: monthly orchestration, coverage matrix, reproducibility reporting, canonical-event hardening assessment, and tensor export audit/reproducibility artifacts are dataset-factory deliverables only. They must preserve `recv_ts_ns` alignment, finalized-clean policy filtering, approved dense BTC reference usage through `live_reference_events_v1`, leakage-only outcome separation, and deterministic rebuild behavior.
- Evidence target: sprint deliverables live under `docs/` and production CLI entrypoints under `src/polymarket_recorder/cli.py`; no model artifacts are produced by this sprint.

## 2026-04-27 - Phase 6 Production Build Partitioning

- Task: unblock uncapped `execution_intent_dataset_v1` production materialization after monolithic day-level replay workers exceeded available RAM and crashed during the five-day build.
- Decision: production Phase 6 materialization uses deterministic market-batch partition execution by default. Each batch loads only the canonical/raw L2 replay needed for its market subset, writes a temporary parquet part, and the builder merges parts in stable market order into the official daily shard only after all parts pass.
- Decision: this is not a row cap, market-family cap, scenario cap, or sampling policy. The same eligible anchors and scenario grid remain in scope; the change is execution layout, memory isolation, and resumability.
- Decision: canonical L2 shards should be backfilled/hardened before full Phase 6 rebuilds whenever missing, because repeatedly replaying raw zstd files is a production bottleneck and makes process-level parallelism memory-heavy.
- Evidence target: build reports must record `partition_by_market`, `market_batch_size`, requested/effective workers, part counts, runtimes, row counts, hashes, and any failed partition before claiming production readiness.

## 2026-04-28 - Infrastructure Sprint Scope and Permanent Source Policy

- Task: complete the remaining roadmap infrastructure without model training, live trading, or dataset rebuilding.
- Decision: `infrastructure_sprint` work is limited to builders, CLIs, schemas, orchestration, audits, model-training harnesses, evaluation, policy scaffolding, paper/live guardrails, and documentation. It may create small control-plane artifacts such as `execution_intent_coverage_gaps.parquet`, but it must not train models or rebuild historical modeling datasets.
- Decision: FNG excluded, RTDS excluded, Bybit excluded, Coinbase Advanced excluded, Deribit excluded, Chainlink live excluded, and Chainlink onchain excluded remain permanent modeling-source decisions unless a new reliability audit plus decision log entry reverses them. The approved dense BTC reference remains `live_reference_events_v1`; secondary Chainlink public delayed context is calibration-only with a causal floor.
- Evidence: `docs/source_policy_ledger.md` is the self-contained source ledger for this infrastructure sprint.

## 2026-04-28 - Canonical Event Base and Asof Join Contract

- Task: implement the missing roadmap Section VI canonical event base.
- Decision: `polymarket_ws_events_v1`, `external_btc_events_v1`, and `secondary_context_events_v1` are canonical event builders that read finalized-clean approved sources through the quality gate and use `recv_ts_ns` as the only event clock.
- Decision: `asof_join_lib_v1` is the only approved direct as-of join surface for new code. Forward joins are rejected because they create leakage; missing right-side rows must produce nulls, not exceptions.
- Decision: `external_btc_events_v1` excludes the Binance USD-M quarantine hour `2026-04-20 07:00 UTC` and forbids Bybit, Coinbase, Deribit, FNG, and RTDS input.
- Decision: `secondary_context_events_v1` includes only Chainlink public delayed calibration rows and applies `causal_available_ts_ns = recv_ts_ns + 1800s`.
- Evidence: implementation lives in `src/canonical/`; test coverage lives in `tests/test_asof_join_lib.py`, `tests/test_polymarket_ws_events_builder.py`, `tests/test_external_btc_events_builder.py`, and `tests/test_secondary_context_events_builder.py`.

## 2026-04-28 - Event Triggered Sampling Extension

- Task: add roadmap Section IX event-triggered sampling infrastructure for `resolution_snapshot_dataset_v1`.
- Decision: `event_triggered` rows are additive to regular coarse snapshots and carry `trigger_type` plus `trigger_magnitude`; regular rows remain valid and are not replaced.
- Decision: triggers must be causal and deduplicated by trigger type with the configured minimum gap. `training_feature_eligible` continues to be inherited from the base snapshot logic; no trigger is allowed to override leakage or quality eligibility.
- Evidence: implementation lives in `src/polymarket_recorder/event_triggered_sampler.py`; schema contract was updated in `config/schemas/resolution_snapshot_dataset_v1_coarse_schema.yaml`.

## 2026-04-28 - Fee Schedule Freeze

- Task: formalize the execution simulator fee schedule as a versioned registry.
- Decision: `fee_schedule` lookup is frozen in `config/fee_schedules/fee_schedule_registry.yaml`; BTC 5-minute markets from `2026-04-19` through `2026-04-23` use `polymarket_crypto_post_20260330`.
- Decision: fee formula tests use the documented peak behavior: post-2026-03-30 `size * 0.072 * price * (1 - price)` produces 1.80% at price 0.50.
- Evidence: loader/validator lives in `src/execution_simulator/fee_schedule_registry.py` and tests live in `tests/test_fee_schedule_registry.py`.

## 2026-04-28 - Model, Evaluation, Policy, Paper, and Live Infrastructure

- Task: complete the no-training infrastructure needed before Phase 4 model work starts.
- Decision: model trainer modules are harnesses only. They enforce `LeakageSafeLoader`, experiment manifest writing before fit, artifact writing, calibration hooks, and eval report generation, but this sprint intentionally trains no model.
- Decision: evaluation metrics and promote/kill gates exist for all 10 models. Model 10 remains blocked on `execution_intent_dataset_v2_enriched`, whose builder is a `NotImplementedError` stub until models 1-9 exist.
- Decision: policy weights are placeholders and ScoreAggregator enters placeholder mode; placeholder mode refuses trading by producing neutral scores and policy no-trade conditions.
- Decision: paper trading replay, observability, drift monitoring, live kill switch, risk caps, and daily live audit are infrastructure guardrails only. No live deployment is authorized without paper-trading and markout audit evidence.
- Evidence: implementation spans `src/model_factory/trainers/`, `src/model_factory/evaluation/`, `src/policy/`, `src/paper_trading/`, and `src/live/`.

## 2026-04-28 - Dataset Followups and Infrastructure Audit

- Task: close nonblocking followups from the dataset pipeline production readiness audit.
- Decision: one-sided/stale book rates are monitored by `monitor-book-quality` rather than silently removed from training data; row-level quality flags and `training_feature_eligible` remain the training gate.
- Decision: 51 trainable markets without execution-intent rows are accepted as-is for v1 and explicitly documented. This affects fill probability/adverse selection training coverage only; market-interval and resolution-snapshot datasets are unaffected.
- Decision: all major infrastructure decisions must be re-auditable by `audit-infrastructure`, with target verdict 0 CRITICAL and 0 WARNING.
- Evidence: gap artifact is `data/canonical/execution_intent_coverage_gaps.parquet`; audit implementation is `src/model_factory/infrastructure_audit.py`.
