from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from binance_recorder.utils import ns_to_iso

from .time_utils import utc_now_ns

_SENSITIVE_FIELD_MARKERS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "credential",
    "auth",
    "passphrase",
)


def _is_sensitive_field(name: str | None) -> bool:
    if not name:
        return False
    normalized = str(name).strip().lower()
    return any(marker in normalized for marker in _SENSITIVE_FIELD_MARKERS)


def _sanitize_snapshot(value: Any, *, field_name: str | None = None) -> Any:
    if _is_sensitive_field(field_name):
        return "<redacted>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _sanitize_snapshot(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _sanitize_snapshot(nested, field_name=str(key))
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [_sanitize_snapshot(item) for item in value]
    if isinstance(value, list):
        return [_sanitize_snapshot(item) for item in value]
    if isinstance(value, set):
        return [_sanitize_snapshot(item) for item in sorted(value, key=repr)]
    return repr(value)


@lru_cache(maxsize=1)
def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _workspace_git_sha() -> str | None:
    root = _workspace_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return None
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None


@dataclass(slots=True, frozen=True)
class RecorderRuntimeMetadata:
    recorder_service: str
    runtime_session_id: str
    runtime_started_at_ns: int
    runtime_started_at_iso: str
    workspace_git_sha: str | None
    workspace_version: str
    config_fingerprint: str
    config_snapshot: dict[str, Any]

    def static_fields(self) -> dict[str, Any]:
        return {
            "recorder_service": self.recorder_service,
            "runtime_session_id": self.runtime_session_id,
            "runtime_started_at_ns": self.runtime_started_at_ns,
            "runtime_started_at_iso": self.runtime_started_at_iso,
            "workspace_git_sha": self.workspace_git_sha,
            "workspace_version": self.workspace_version,
            "config_fingerprint": self.config_fingerprint,
            "config_snapshot_redacted": True,
            "config_snapshot": self.config_snapshot,
        }

    def manifest_fields(self, base_fields: dict[str, Any]) -> dict[str, Any]:
        return {**base_fields, **self.static_fields()}

    def lifecycle_fields(
        self,
        *,
        include_config_snapshot: bool = False,
        extra_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "recorder_service": self.recorder_service,
            "runtime_session_id": self.runtime_session_id,
            "runtime_started_at_ns": self.runtime_started_at_ns,
            "runtime_started_at_iso": self.runtime_started_at_iso,
            "workspace_git_sha": self.workspace_git_sha,
            "workspace_version": self.workspace_version,
            "config_fingerprint": self.config_fingerprint,
        }
        if include_config_snapshot:
            payload["config_snapshot_redacted"] = True
            payload["config_snapshot"] = self.config_snapshot
        if extra_fields:
            payload.update(extra_fields)
        return payload


def build_runtime_metadata(*, recorder_service: str, config: Any) -> RecorderRuntimeMetadata:
    runtime_started_at_ns = utc_now_ns()
    config_snapshot = _sanitize_snapshot(config)
    config_bytes = json.dumps(config_snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    workspace_git_sha = _workspace_git_sha()
    return RecorderRuntimeMetadata(
        recorder_service=recorder_service,
        runtime_session_id=uuid.uuid4().hex,
        runtime_started_at_ns=runtime_started_at_ns,
        runtime_started_at_iso=ns_to_iso(runtime_started_at_ns),
        workspace_git_sha=workspace_git_sha,
        workspace_version=f"git:{workspace_git_sha}" if workspace_git_sha else "workspace_no_git",
        config_fingerprint=hashlib.sha256(config_bytes).hexdigest(),
        config_snapshot=config_snapshot if isinstance(config_snapshot, dict) else {"value": config_snapshot},
    )
