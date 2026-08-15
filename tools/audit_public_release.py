"""Audit the public checkout for size, manifest integrity, and obvious secret leaks."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".ruff_cache", ".mypy_cache"}
TEXT_SUFFIXES = {
    ".cfg", ".csv", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"
}
FORBIDDEN_NAMES = {"API_Keys", ".env", "id_rsa", "id_ed25519"}
FORBIDDEN_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
CONTENT_PATTERNS = {
    "private host path": re.compile(r"C:[\\/]Users[\\/]HONZA", re.IGNORECASE),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-style token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_RELEASE_BYTES = 60 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)
    )


def main() -> int:
    files = _files()
    failures: list[str] = []
    suspicious_names = []
    content_hits = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            suspicious_names.append(relative)
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"oversized file: {relative} ({path.stat().st_size} bytes)")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in CONTENT_PATTERNS.items():
                if pattern.search(text):
                    content_hits.append({"path": relative, "pattern": label})

    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > MAX_RELEASE_BYTES:
        failures.append(f"release exceeds {MAX_RELEASE_BYTES} bytes: {total_bytes}")
    failures.extend(f"forbidden file: {path}" for path in suspicious_names)
    failures.extend(f"content leak ({hit['pattern']}): {hit['path']}" for hit in content_hits)

    manifest_path = ROOT / "artifacts/evaluation_repro_v2/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked_manifest_files = 0
    for row in manifest["files"]:
        path = ROOT / row["public_path"]
        if not path.is_file():
            failures.append(f"manifest file missing: {row['public_path']}")
            continue
        checked_manifest_files += 1
        if path.stat().st_size != row["bytes"]:
            failures.append(f"manifest size mismatch: {row['public_path']}")
        if _sha256(path) != row["sha256"]:
            failures.append(f"manifest hash mismatch: {row['public_path']}")

    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    if "!artifacts/evaluation_repro_v2/checkpoints/*.pt" not in ignore_text:
        failures.append("public checkpoint exception missing from .gitignore")

    largest = sorted(files, key=lambda path: path.stat().st_size, reverse=True)[:5]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "files": len(files),
        "bytes": total_bytes,
        "mib": round(total_bytes / (1024 * 1024), 2),
        "largest_files": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size}
            for path in largest
        ],
        "manifest_files_verified": checked_manifest_files,
        "secret_or_private_path_hits": content_hits,
        "forbidden_files": suspicious_names,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
