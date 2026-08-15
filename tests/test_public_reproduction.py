"""Tests for the compact public reproduction bundle."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def test_public_evidence_recomputes_identically() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/reproduce_public_evidence.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_public_release_audit_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/audit_public_release.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_bundle_has_five_checkpoints_and_ten_balanced_markets() -> None:
    bundle = ARTIFACTS / "evaluation_repro_v2"
    assert len(list((bundle / "checkpoints").glob("seed*.pt"))) == 5
    subset = np.load(bundle / "mini_inference/test_subset_10_markets.npz", allow_pickle=False)
    assert subset["X"].shape == (10, 1500, 69)
    assert subset["valid_mask"].shape == (10, 1500)
    assert set(subset["y"].tolist()) == {0, 1}
    for day in np.unique(subset["date"]):
        assert sorted(subset["y"][subset["date"] == day].tolist()) == [0, 1]


def test_manifest_contains_no_absolute_private_paths() -> None:
    manifest_path = ARTIFACTS / "evaluation_repro_v2/manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in text
    manifest = json.loads(text)
    assert manifest["policy"] == "explicit allowlist; private source read-only"
    assert len(manifest["files"]) == 26
    assert all(not Path(row["source_relative_path"]).is_absolute() for row in manifest["files"])


def test_sanitized_shadow_counts_reconcile() -> None:
    summary = json.loads(
        (ARTIFACTS / "live_log_sample_v1/summary.json").read_text(encoding="utf-8")
    )
    assert sum(day["published_enter_records"] for day in summary["days"].values()) == 164
    assert sum(sum(day["decision_counts"].values()) for day in summary["days"].values()) == 45436


def test_intellectual_ownership_declaration_and_project_role_are_present() -> None:
    ownership = (ROOT / "INTELLECTUAL_OWNERSHIP.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # Anchored on hyphen-free wording: hyphenation gets adjusted for house style and
    # must not be able to break the guard on the declaration itself.
    assert "ideas in this project originated with me" in ownership
    assert "This is my personal declaration, not independently notarized proof" in ownership
    # The assistance disclosure must stay on both documents a reader sees first.
    assert "unaided authorship of every line" in ownership
    assert "AI tools supported much of" in readme


def test_tcn_mini_inference_when_torch_is_installed() -> None:
    pytest.importorskip("torch")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/reproduce_tcn_mini.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"status": "PASS"' in result.stdout
