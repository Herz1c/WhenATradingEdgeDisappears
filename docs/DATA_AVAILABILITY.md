# Data and release boundary

The release intentionally balances inspectability, privacy, and GitHub size. It publishes
the smallest evidence bundle that can recompute the central model-score and selection-
bias conclusions without exposing credentials, account state, or a 229.8 GB private
corpus.

## Publicly reproducible

| Artifact | Content | Check enabled |
|---|---|---|
| `artifacts/evaluation_repro_v2/predictions/` | All validation/test loss-band rows for five seeds | Seven calibration scores and paired clustered intervals |
| `artifacts/evaluation_repro_v2/checkpoints/` | Five TCN state dictionaries | Mini inference parity |
| `artifacts/evaluation_repro_v2/mini_inference/` | Ten full 1,500-step markets and expected rows | 12,930 checkpoint-row comparisons |
| `artifacts/evaluation_repro_v2/dataset_metadata/` | Feature names, normalization, split metadata, full-array hashes | Schema and source binding |
| `artifacts/tcn_v2_eval/` | Per-seed calibration maps and generated Brier summary | Report-to-prediction consistency |
| `artifacts/audit_v1/wrc_universe_daily_pnl.npz` | 846 searched candidate-day series | Original search audit input |
| `artifacts/audit_v2/` | Corrected 847-candidate result and bootstrap null draws | Deterministic negative selection verdict |
| `artifacts/live_log_sample_v1/` | Decision denominators and 164 sanitized entry rows | Prospective operation, not PnL |
| `artifacts/provenance/` | Local commit metadata and extraction-state hash | Development trace, not external attestation |

Every copied file is listed with SHA-256 and byte count in
`artifacts/evaluation_repro_v2/manifest.json`. The exporter uses an explicit allowlist and
does not walk arbitrary private files.

## Private but cryptographically bound

The full v2 arrays are not released. Their hashes and counts are:

| Split | Markets | SHA-256 |
|---|---:|---|
| Train | 7,470 | `3e19a5f2b69859e7b7a0e080f6cbe733258ca33e4a958dcf7b85e08c2e560e89` |
| Validation | 917 | `361bbe7e58d10fd2f93ba4abe9074d02703843f0d1c9f40b5c199d20091de57a` |
| Test | 516 | `1e041a2c60bf80f542099712eb13750ed6d4b9f1be0d79b551ae0506cd58ebaa` |

The hashes bind public predictions and the mini fixture to named private inputs, but a
reader cannot infer or reconstruct those tensors from a digest.

## Not distributed

- Raw compressed market and oracle feeds.
- Full canonical Parquet and train/validation/test tensors.
- Training-row prediction arrays and optimizer state.
- Complete private logs and feature vectors.
- API keys, wallet material, account state, environment files, and private Git objects.

The raw inventory reports 246,718,377,000 bytes, 119,665 files, 76 recorded days, and 37
missing calendar days from April 19 through August 9. These are locally measured inventory
claims; they cannot be independently recounted from the release.

## Consequence

A fresh clone can check scores, uncertainty, checkpoint inference, selection adjustment,
claims, and paper consistency. It cannot rebuild the recorder corpus, retrain the models,
rerun the entire historical backtest from raw events, or perform the missing receive-time
experiment. Those limitations are part of the result, not hidden footnotes.
