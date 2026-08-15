# How I preserved provenance and integrity

## My source workspace snapshot

I created this public release from a separate private workspace, and the release process did
not write to that workspace. At extraction, my local Git history contained 34 commits from
May 20 through June 29, 2026. The private HEAD was
`5ba8fc999a0e3f87766a5ae1763b50f8e9ed23dc`, and the local `origin/main` tracking ref was
`378465b41a80d34f1edd01985a8c71e0d3cbd660`. My private worktree was dirty, so I retained
only a SHA-256 of the exact porcelain-v2 status byte stream in the public provenance file.

I did not copy `.git` objects, diffs, ignored files, credentials, or raw private state into
this release. The commit subjects and dates are local metadata, not signed timestamps. The
TCN, July shadow, and final audit phase happened after the last commit, so I describe that
phase as artifact-level rather than commit-level provenance.

## How I bound the public artifacts

The extraction manifest records the public path, source-relative path, byte count, and
SHA-256 for 26 copied or derived model and dataset files. The split metadata retains SHA-256
values for my complete private train, validation, and test arrays. The sanitized shadow
summaries retain the size and SHA-256 of each original full log.

These hashes let me demonstrate later that a private file matches the input evaluated here.
They do not let a public reader inspect omitted contents or establish when those contents
first existed.

## How I describe the evidence

| Evidence | What it supports | What it does not establish |
|---|---|---|
| Public code and tests | Direct inspection of implemented behavior | Validation on my private data |
| Public predictions and checkpoints | Reproduction of scoring and mini-inference | Independent validation of training data |
| Public candidate-day matrix | Reproduction of the central selection result | Proof that the search universe is complete |
| My local Git metadata | Evidence of iterative development | Signed, hosted, or complete history through August |
| My local lock and log timestamps | A recorded local chronology | Independent timestamping |
| Private-array hashes | Binding to named private files | Disclosure or validation of their contents |

For this reason, I use the phrase "local provenance." I do not describe the headline phase
as "pre-registered," "externally authenticated," or "independently timestamped."

## Relationship to my research record

I document my intellectual contribution in
[INTELLECTUAL_OWNERSHIP.md](../INTELLECTUAL_OWNERSHIP.md) and map individual decisions to
earlier records in [IDEA_PROVENANCE.md](IDEA_PROVENANCE.md). My April and May records contain
receive-time contracts, audit stop conditions, rejected alternatives, execution requirements,
and no-deployment gates. The July/August phase is represented by artifact metadata rather
than commit history.
