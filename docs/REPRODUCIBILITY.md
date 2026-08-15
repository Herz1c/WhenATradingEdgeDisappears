# Public reproduction guide

## Fast path

```bash
py -m pip install -e ".[dev,paper]"
py tools/reproduce_public_evidence.py --check
py -m pytest tests -q
py tools/audit_public_release.py
py tools/build_publication_claims.py --check
py tools/build_public_paper.py --check
```

`reproduce_public_evidence.py --check` independently recalculates the seven calibration
rows, primary and sensitivity bootstrap intervals, and 847-candidate White's Reality
Check. It then compares those calculations with the committed JSON and null-distribution
artifacts.

## Checkpoint inference

PyTorch is optional because it is large:

```bash
py -m pip install -e ".[sequence]"
py tools/reproduce_tcn_mini.py
```

The script loads five state dictionaries, reconstructs the exact causal TCN from each
report, appends the validity-mask feature, and compares output logits with the published
expected arrays for ten test markets. Cross-build CPU kernels are accepted only within
0.0001 logit.

## Regenerate outputs

```bash
py tools/reproduce_public_evidence.py
py tools/build_publication_claims.py
py tools/build_public_paper.py
```

The first command updates `brier_summary.json`, `selection_audit.json`, and the compressed
bootstrap null distribution. The paper embeds a SHA-256 over its manuscript, claim
register, and quantitative source artifacts; `--check` detects stale builds.

## Release-only extraction

The public evidence was copied from the author's private workspace with:

```bash
py tools/build_public_evidence_bundle.py --private-root PATH_TO_PRIVATE_WORKSPACE
```

This is not needed by a reader. The script refuses to use the public root as its source,
names every allowed file explicitly, strips private paths from split metadata, publishes
only sanitized `ENTER` records, and copies no Git objects or secrets.

## Reproduction boundary

| Question | Fresh clone answer |
|---|---|
| Can I recompute the Brier table and uncertainty? | Yes |
| Can I rerun the released checkpoints on real-shaped inputs? | Yes |
| Can I recompute the corrected multiple-testing verdict? | Yes |
| Can I verify the claim register and PDF are current? | Yes |
| Can I retrain from the private corpus? | No |
| Can I rebuild the historical backtest from raw messages? | No |
| Can I test a receive-time correction? | No |

Passing checks establish public computational consistency for the released evidence. They
do not convert a source-time counterfactual into live-causal evidence.
