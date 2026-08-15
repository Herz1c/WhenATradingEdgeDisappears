from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .source_registry import DEBUG_ONLY, VERIFICATION_ONLY


DEFAULT_DATASET_POLICY_PATH = Path("config/dataset_policy_v1.yaml")

# Scripture-defined explicit quarantine from roadmap rule II.6:
# "Binance USD-M has an explicitly quarantined bad hour that must not go into training."
SCRIPTURE_EXPLICIT_SOURCE_HOUR_QUARANTINES = {
    ("binance_usdm", "2026-04-20", "07"),
}


@dataclass(slots=True, frozen=True)
class DatasetPolicy:
    source_roles: dict[str, Any]
    health_policy: dict[str, Any]
    training_data_policy: dict[str, Any]
    quarantine_policy: dict[str, Any]

    @property
    def live_reference_primary_inputs(self) -> tuple[str, ...]:
        values = self.source_roles.get("live_reference_primary_inputs", [])
        return tuple(str(value) for value in values)

    @property
    def live_reference_calibration_inputs(self) -> dict[str, dict[str, Any]]:
        values = self.source_roles.get("live_reference_calibration_inputs", {})
        return {
            str(source_name): dict(source_config)
            for source_name, source_config in values.items()
            if isinstance(source_config, dict)
        }

    @property
    def verification_only_inputs(self) -> tuple[str, ...]:
        values = self.source_roles.get("verification_only_inputs", [])
        return tuple(str(value) for value in values)

    @property
    def forbidden_for_live_reference(self) -> tuple[str, ...]:
        values = self.source_roles.get("forbidden_for_live_reference", [])
        return tuple(str(value) for value in values)

    @property
    def allowed_manifest_quality_classes(self) -> set[str]:
        values = self.health_policy.get("allowed_manifest_quality_classes", [])
        return {str(value) for value in values}

    @property
    def quarantined_manifest_quality_classes(self) -> set[str]:
        values = self.health_policy.get("quarantined_manifest_quality_classes", [])
        return {str(value) for value in values}

    @property
    def allowed_file_states(self) -> set[str]:
        values = self.health_policy.get("allowed_file_states", [])
        return {str(value) for value in values}

    @property
    def forbidden_file_states(self) -> set[str]:
        values = self.health_policy.get("forbidden_file_states", [])
        return {str(value) for value in values}

    def source_is_live_reference_primary(self, source_name: str) -> bool:
        return source_name in self.live_reference_primary_inputs

    def source_is_live_reference_calibration(self, source_name: str) -> bool:
        return source_name in self.live_reference_calibration_inputs

    def source_is_verification_only(self, source_name: str) -> bool:
        return source_name in self.verification_only_inputs


def load_dataset_policy(path: Path | None = None) -> DatasetPolicy:
    policy_path = Path(path or DEFAULT_DATASET_POLICY_PATH)
    payload = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset policy must load as a mapping: {policy_path}")
    required_sections = {
        "source_roles",
        "health_policy",
        "training_data_policy",
        "quarantine_policy",
    }
    missing = sorted(required_sections - set(payload))
    if missing:
        raise ValueError(f"Dataset policy is missing required sections: {missing}")
    return DatasetPolicy(
        source_roles=dict(payload["source_roles"]),
        health_policy=dict(payload["health_policy"]),
        training_data_policy=dict(payload["training_data_policy"]),
        quarantine_policy=dict(payload["quarantine_policy"]),
    )


def training_metadata_eligibility(
    *,
    policy: DatasetPolicy,
    metadata: dict[str, Any],
    source_name: str | None = None,
    date_str: str | None = None,
    hour_str: str | None = None,
) -> tuple[bool, str | None]:
    if source_name is not None and date_str is not None and hour_str is not None:
        if (source_name, date_str, hour_str) in SCRIPTURE_EXPLICIT_SOURCE_HOUR_QUARANTINES:
            return False, "source_specific_integrity_check"

    quality_class = str(metadata.get("quality_class") or "unknown")
    if quality_class not in policy.allowed_manifest_quality_classes:
        return False, f"manifest_quality_class:{quality_class}"

    file_state = str(metadata.get("file_state") or "unknown")
    if file_state not in policy.training_data_policy.get("allowed_file_states", []):
        return False, f"file_state:{file_state}"

    source_tier = str(metadata.get("source_tier") or "unknown")
    if source_tier in {VERIFICATION_ONLY, DEBUG_ONLY}:
        return False, f"source_tier:{source_tier}"

    return True, None
