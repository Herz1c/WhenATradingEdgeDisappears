# Evaluation reproduction bundle

This directory contains the compact public inputs for forecasting evaluation.

- `predictions/`: every validation/test row inside the 15–150 second loss band for five
  seeds. These files are sufficient for all seven calibration scores and paired losses.
- `checkpoints/`: five PyTorch state dictionaries, approximately 0.6 MB each.
- `mini_inference/`: ten full 1,500-step test episodes plus expected loss-band outputs.
- `dataset_metadata/`: feature/audit/quote names, train-only normalization metadata,
  sanitized split metadata, and hashes of the complete private arrays.
- `manifest.json`: byte counts, SHA-256 digests, source-relative paths, and explicit
  exclusions.

Run `py tools/reproduce_public_evidence.py --check` for scores and uncertainty, or
`py tools/reproduce_tcn_mini.py` after installing the `sequence` extra for checkpoint
parity. These files do not make the historical dataset receive-time causal.
