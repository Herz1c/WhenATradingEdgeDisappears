"""Extract a small, allowlisted public evidence bundle from the private workspace.

This script is for the repository author's release process.  It never writes to
the private source tree, never copies Git objects, and never walks arbitrary
files looking for material to publish.  Every copied or derived input is named
explicitly below so credentials, raw account state, and unrelated logs cannot
enter the public release accidentally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 11, 23, 42, 101)
SPLITS = ("val", "test")
METADATA_FILES = (
    "normalization.json",
    "feature_names.json",
    "audit_names.json",
    "quote_names.json",
)
LIVE_LOGS = (
    "logs/tcn_shadow_bot_direct_capture_v6/tcn_decisions_2026-07-12.jsonl",
    "logs/tcn_shadow_bot_direct_capture_v6/tcn_decisions_2026-07-13.jsonl",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy(
    source: Path,
    destination: Path,
    manifest: list[dict[str, Any]],
    private_root: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    manifest.append(
        {
            "public_path": destination.relative_to(PUBLIC_ROOT).as_posix(),
            "source_relative_path": source.relative_to(private_root).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    )


def _git(*args: str, cwd: Path, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=cwd, text=text)


def extract_git_provenance(private_root: Path, out: Path) -> dict[str, Any]:
    fmt = "%H%x1f%aI%x1f%an%x1f%s"
    rows = []
    for line in str(_git("log", "--reverse", f"--pretty=format:{fmt}", cwd=private_root)).splitlines():
        commit, authored_at, author, subject = line.split("\x1f", 3)
        rows.append(
            {
                "commit": commit,
                "authored_at": authored_at,
                "author": author,
                "subject": subject,
            }
        )
    status = _git("status", "--porcelain=v2", "-z", cwd=private_root, text=False)
    head = str(_git("rev-parse", "HEAD", cwd=private_root)).strip()
    try:
        origin_main = str(_git("rev-parse", "origin/main", cwd=private_root)).strip()
    except subprocess.CalledProcessError:
        origin_main = None
    try:
        remote = str(_git("config", "--get", "remote.origin.url", cwd=private_root)).strip()
    except subprocess.CalledProcessError:
        remote = None
    payload = {
        "classification": "LOCAL GIT PROVENANCE; FINAL JULY/AUGUST PHASE NOT COMMITTED",
        "head": head,
        "origin_main_local_tracking_ref": origin_main,
        "origin_url_recorded_locally": remote,
        "commit_count": len(rows),
        "first_authored_at": rows[0]["authored_at"] if rows else None,
        "last_authored_at": rows[-1]["authored_at"] if rows else None,
        "private_worktree_was_dirty_at_extraction": bool(status),
        "private_status_sha256": hashlib.sha256(status).hexdigest(),
        "limits": [
            "Commit metadata was read from the local private workspace; this export is not a signed attestation.",
            "The local origin/main tracking ref ends before the July TCN locks and final audit.",
            "No Git objects, diffs, ignored files, credentials, or private repository state are included.",
        ],
        "commits": rows,
    }
    _write_json(out, payload)
    return payload


def extract_dataset_metadata(
    private_root: Path, out_dir: Path, copied: list[dict[str, Any]]
) -> dict[str, Any]:
    source_dir = private_root / "data/datasets/btc_5m_episodes_v2_200ms"
    for name in METADATA_FILES:
        _copy(source_dir / name, out_dir / name, copied, private_root)

    source_splits = source_dir / "splits_v2.json"
    splits = json.loads(source_splits.read_text(encoding="utf-8"))
    splits["daily_shards_dir"] = "private daily shards; not distributed"
    splits["public_release_note"] = (
        "The full arrays are private. Their source hashes are retained below, while a ten-market "
        "test subset and all loss-band validation/test predictions are public."
    )
    splits["source_splits_v2_sha256"] = sha256(source_splits)
    public_splits = out_dir / "splits_v2_public.json"
    _write_json(public_splits, splits)
    copied.append(
        {
            "public_path": public_splits.relative_to(PUBLIC_ROOT).as_posix(),
            "source_relative_path": "data/datasets/btc_5m_episodes_v2_200ms/splits_v2.json (path redacted)",
            "bytes": public_splits.stat().st_size,
            "sha256": sha256(public_splits),
        }
    )
    return splits


def extract_model_evidence(
    private_root: Path, out_dir: Path, copied: list[dict[str, Any]]
) -> dict[str, Any]:
    reports: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        source_dir = private_root / f"artifacts/tcn_v2_c64_b7_ttc15_150_seed{seed}"
        report_path = source_dir / "tcn_report.json"
        reports[seed] = json.loads(report_path.read_text(encoding="utf-8"))
        for split in SPLITS:
            _copy(
                source_dir / f"predictions_{split}.npz",
                out_dir / "predictions" / f"seed{seed}_{split}.npz",
                copied,
                private_root,
            )
        _copy(
            source_dir / "model.pt",
            out_dir / "checkpoints" / f"seed{seed}.pt",
            copied,
            private_root,
        )

    dataset_path = private_root / "data/datasets/btc_5m_episodes_v2_200ms/test.npz"
    dataset = np.load(dataset_path, allow_pickle=False, mmap_mode="r")
    reference_path = out_dir / "predictions" / "seed11_test.npz"
    reference = np.load(reference_path, allow_pickle=False)
    eligible = {int(x) for x in np.unique(reference["ep_idx"]).tolist()}

    selected: list[int] = []
    dates = dataset["date"]
    labels = dataset["y"]
    slugs = dataset["market_slug"]
    for day in sorted({str(x) for x in dates.tolist()}):
        day_indices = [i for i in range(len(dates)) if str(dates[i]) == day and i in eligible]
        for label in (0, 1):
            choices = [i for i in day_indices if int(labels[i]) == label]
            if choices:
                selected.append(min(choices, key=lambda i: str(slugs[i])))
    selected = sorted(set(selected))
    if len(selected) != 10:
        raise RuntimeError(f"expected two eligible markets for each of five test days; got {selected}")

    subset_path = out_dir / "mini_inference" / "test_subset_10_markets.npz"
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subset_path,
        X=np.asarray(dataset["X"][selected], dtype=np.float32),
        valid_mask=np.asarray(dataset["valid_mask"][selected], dtype=bool),
        y=np.asarray(dataset["y"][selected], dtype=np.int8),
        p_market=np.asarray(dataset["p_market"][selected], dtype=np.float32),
        date=np.asarray(dataset["date"][selected]),
        market_slug=np.asarray(dataset["market_slug"][selected]),
        open_s=np.asarray(dataset["open_s"][selected], dtype=np.int64),
        original_ep_idx=np.asarray(selected, dtype=np.int32),
    )
    copied.append(
        {
            "public_path": subset_path.relative_to(PUBLIC_ROOT).as_posix(),
            "source_relative_path": "derived from data/datasets/btc_5m_episodes_v2_200ms/test.npz",
            "bytes": subset_path.stat().st_size,
            "sha256": sha256(subset_path),
        }
    )

    selected_array = np.asarray(selected, dtype=np.int32)
    for seed in SEEDS:
        pred_path = out_dir / "predictions" / f"seed{seed}_test.npz"
        pred = np.load(pred_path, allow_pickle=False)
        mask = np.isin(pred["ep_idx"], selected_array)
        expected_path = out_dir / "mini_inference" / f"expected_seed{seed}.npz"
        np.savez_compressed(expected_path, **{key: pred[key][mask] for key in pred.files})
        copied.append(
            {
                "public_path": expected_path.relative_to(PUBLIC_ROOT).as_posix(),
                "source_relative_path": f"derived from artifacts/tcn_v2_c64_b7_ttc15_150_seed{seed}/predictions_test.npz",
                "bytes": expected_path.stat().st_size,
                "sha256": sha256(expected_path),
            }
        )

    return {
        "seeds": list(SEEDS),
        "splits": list(SPLITS),
        "mini_original_ep_indices": selected,
        "mini_markets": [str(x) for x in dataset["market_slug"][selected].tolist()],
        "mini_dates": [str(x) for x in dataset["date"][selected].tolist()],
        "full_private_test_sha256": sha256(dataset_path),
        "reports_sha256": {
            str(seed): sha256(
                private_root / f"artifacts/tcn_v2_c64_b7_ttc15_150_seed{seed}/tcn_report.json"
            )
            for seed in SEEDS
        },
    }


def _sanitize_enter(record: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "iso",
        "logged_at_ns",
        "snapshot_ts_ns",
        "market_slug",
        "strategy_id",
        "decision",
        "ttc_s",
        "lock_id",
        "beta",
        "p_market",
        "tcn_delta",
        "p_strategy",
        "ev",
        "ev_up",
        "ev_down",
        "ev_gate",
        "side",
        "up_best_ask",
        "down_best_ask",
        "fill_quote",
    )
    out = {key: record.get(key) for key in allowed if key in record}
    audit = record.get("audit") or {}
    out["availability_audit"] = {
        key: audit.get(key)
        for key in (
            "source_valid",
            "pm_lag_s",
            "pm_recv_lag_s",
            "cex_age_s",
            "rtds_source_age_s",
            "history_ready",
            "history_missing_steps",
            "source_replay_pending",
        )
        if key in audit
    }
    hashes = record.get("artifact_hashes") or {}
    out["artifact_hashes"] = {
        "lock_sha256": hashes.get("lock_sha256"),
        "model_pt_sha256": hashes.get("model_pt_sha256"),
        "normalization_sha256": hashes.get("normalization_sha256"),
    }
    return out


def extract_live_log_sample(private_root: Path, out_dir: Path) -> dict[str, Any]:
    days: dict[str, Any] = {}
    all_enters: list[dict[str, Any]] = []
    for relative in LIVE_LOGS:
        source = private_root / relative
        counts: Counter[str] = Counter()
        enters: list[dict[str, Any]] = []
        first_iso = None
        last_iso = None
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    counts["MALFORMED_JSON"] += 1
                    continue
                decision = str(row.get("decision", "UNKNOWN"))
                counts[decision] += 1
                iso = row.get("iso")
                first_iso = first_iso or iso
                last_iso = iso or last_iso
                if decision == "ENTER":
                    sanitized = _sanitize_enter(row)
                    enters.append(sanitized)
                    all_enters.append(sanitized)
        day = source.stem.removeprefix("tcn_decisions_")
        days[day] = {
            "source_relative_path": relative,
            "source_bytes": source.stat().st_size,
            "source_sha256": sha256(source),
            "first_iso": first_iso,
            "last_iso": last_iso,
            "decision_counts": dict(sorted(counts.items())),
            "published_enter_records": len(enters),
        }
        target = out_dir / f"enter_decisions_{day}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in enters),
            encoding="utf-8",
        )
    summary = {
        "classification": "SANITIZED PROSPECTIVE SHADOW-DECISION EVIDENCE; NOT A PERFORMANCE SCORE",
        "days": days,
        "total_enter_records": len(all_enters),
        "limits": [
            "Only ENTER rows are published; per-day counts preserve the complete decision denominator.",
            "Local paths, session identifiers, full feature vectors, and private raw data were removed.",
            "The logs are locally dated and hashed but are not independently timestamped.",
            "These records demonstrate shadow-system operation, not fillability, PnL, or a trading edge.",
        ],
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def build(private_root: Path) -> None:
    private_root = private_root.resolve()
    if private_root == PUBLIC_ROOT.resolve():
        raise SystemExit("private and public roots must be different")
    if not (private_root / ".git").is_dir():
        raise SystemExit(f"private source has no .git directory: {private_root}")
    if not (private_root / "data/datasets/btc_5m_episodes_v2_200ms/test.npz").is_file():
        raise SystemExit("private v2 test dataset is missing")

    out = PUBLIC_ROOT / "artifacts/evaluation_repro_v2"
    copied: list[dict[str, Any]] = []
    split_info = extract_dataset_metadata(private_root, out / "dataset_metadata", copied)
    model_info = extract_model_evidence(private_root, out, copied)
    git_info = extract_git_provenance(
        private_root, PUBLIC_ROOT / "artifacts/provenance/source_git_history.json"
    )
    live_info = extract_live_log_sample(
        private_root, PUBLIC_ROOT / "artifacts/live_log_sample_v1"
    )
    manifest = {
        "bundle_version": "evaluation_repro_v2",
        "created_at": datetime.now(UTC).isoformat(),
        "policy": "explicit allowlist; private source read-only",
        "dataset": {
            "private_counts": split_info.get("counts"),
            "private_split_hashes": split_info.get("hashes"),
            **model_info,
        },
        "git": {
            "head": git_info["head"],
            "origin_main_local_tracking_ref": git_info["origin_main_local_tracking_ref"],
            "commit_count": git_info["commit_count"],
            "private_status_sha256": git_info["private_status_sha256"],
        },
        "live_logs": {
            "days": sorted(live_info["days"]),
            "total_enter_records": live_info["total_enter_records"],
        },
        "files": sorted(copied, key=lambda row: row["public_path"]),
        "exclusions": [
            "API keys, wallets, account state, and secrets",
            "private Git objects and diffs",
            "raw market corpus and full episode tensors",
            "complete private logs and full feature vectors",
            "training prediction arrays, which are unnecessary for public evaluation checks",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "bundle": str(out.relative_to(PUBLIC_ROOT)),
                "files": len(copied),
                "bytes": sum(int(row["bytes"]) for row in copied),
                "mini_markets": len(model_info["mini_markets"]),
                "live_enters": live_info["total_enter_records"],
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.private_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
