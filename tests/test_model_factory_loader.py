from __future__ import annotations

import calendar
import json
from datetime import date, datetime, time, timezone
from pathlib import Path

import polars as pl
import pytest
import yaml

from model_factory.artifact_writer import ArtifactWriter
from model_factory.dataset_loader import DatasetSplit, LeakageSafeLoader


def ns(day: str, seconds: int = 0) -> int:
    parsed = date.fromisoformat(day)
    dt = datetime.combine(parsed, time.min, tzinfo=timezone.utc)
    return calendar.timegm(dt.utctimetuple()) * 1_000_000_000 + seconds * 1_000_000_000


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def default_rows(rows_per_day: int = 3) -> list[dict]:
    rows = []
    for day_index, day in enumerate(["2026-04-19", "2026-04-20", "2026-04-21", "2026-04-22", "2026-04-23"]):
        for offset in range(rows_per_day):
            rows.append(
                {
                    "ts": ns(day, offset),
                    "feature_a": float(day_index + offset),
                    "feature_b": float(offset),
                    "target": int((day_index + offset) % 2),
                    "leak": "future",
                    "training_feature_eligible": True,
                }
            )
    return rows


def make_loader(
    tmp_path: Path,
    *,
    rows: list[dict] | None = None,
    feature_columns: list[str] | None = None,
    leakage_columns: list[str] | None = None,
    target_column: str = "target",
    ordering_column: str = "ts",
    feature_status: str | None = None,
    split_overrides: dict | None = None,
    dataset_filter: str | None = None,
) -> LeakageSafeLoader:
    rows = rows or default_rows()
    feature_columns = feature_columns if feature_columns is not None else ["feature_a", "feature_b"]
    leakage_columns = leakage_columns if leakage_columns is not None else ["leak"]

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(dataset_dir / "part.parquet")

    feature_dir = tmp_path / "config" / "feature_sets"
    split_dir = tmp_path / "config" / "splits"
    registry_path = tmp_path / "config" / "model_registry_v1.yaml"
    feature_path = feature_dir / "test_model_features_v1.yaml"
    split_path = split_dir / "test_model_split_v1.yaml"

    dataset_config = {
        "family": "synthetic",
        "version": "v1",
        "paths": {"default": str(dataset_dir)},
        "dataset_hash_required": True,
    }
    if dataset_filter:
        dataset_config["filter"] = dataset_filter

    write_yaml(
        registry_path,
        {
            "version": 1,
            "last_updated": "2026-04-28",
            "models": {
                "test_model": {
                    "display_name": "Synthetic test model",
                    "model_id": "test_model",
                    "phase": 0,
                    "status": "not_started",
                    "purpose": "unit test",
                    "dataset": dataset_config,
                    "feature_set": {"version": "v1", "config": str(feature_path)},
                    "target": {"primary": target_column, "leakage_only_cols": leakage_columns},
                    "split": {
                        "method": "anchored_walkforward",
                        "config": str(split_path),
                        "unit": "row",
                        "ordering_col": ordering_column,
                        "purge_days": 0,
                        "embargo_days": 0,
                    },
                    "algorithms": {"baseline": "none"},
                    "evaluation": {"metrics": ["none"], "gate": "none"},
                    "artifacts": {"base_path": "artifacts/test_model/"},
                }
            },
        },
    )

    feature_config = {
        "feature_set_id": "test_model_features_v1",
        "model_id": "test_model",
        "version": "v1",
        "created": "2026-04-28",
        "feature_columns": feature_columns,
        "leakage_only_columns": leakage_columns,
        "target_columns": {"primary": target_column, "auxiliary": []},
    }
    if feature_status:
        feature_config["status"] = feature_status
    write_yaml(feature_path, feature_config)

    split_config = {
        "split_id": "test_model_split_v1",
        "model_id": "test_model",
        "version": "v1",
        "method": "anchored_walkforward",
        "ordering_column": ordering_column,
        "ordering_unit": "row",
        "split_strategy": "day_based",
        "train": {"start_day": "2026-04-19", "end_day": "2026-04-21"},
        "validation": {"start_day": "2026-04-22", "end_day": "2026-04-22"},
        "test": {"start_day": "2026-04-23", "end_day": "2026-04-23"},
        "purge_days": 0,
        "embargo_days": 0,
        "min_train_rows": 1,
        "min_val_rows": 1,
        "min_test_rows": 1,
        "auto_extend": True,
        "auto_extend_val_days": 1,
        "auto_extend_test_days": 1,
        "preserve_validation_rows_on_short_window": True,
    }
    if split_overrides:
        split_config.update(split_overrides)
    write_yaml(split_path, split_config)

    return LeakageSafeLoader(
        model_registry_path=registry_path,
        feature_sets_dir=feature_dir,
        splits_dir=split_dir,
    )


def test_temporal_ordering_enforced(tmp_path: Path) -> None:
    loader = make_loader(tmp_path)
    with pytest.raises(ValueError, match="decreases"):
        loader._verify_temporal_ordering(pl.DataFrame({"ts": [2, 1]}), "ts", "test_model")


def test_no_future_data_in_splits(tmp_path: Path) -> None:
    loader = make_loader(tmp_path)
    with pytest.raises(ValueError, match="Future data violation"):
        loader._verify_no_future_data(
            pl.DataFrame({"ts": [3]}),
            pl.DataFrame({"ts": [2]}),
            pl.DataFrame({"ts": [4]}),
            "ts",
            "test_model",
        )


def test_leakage_column_in_features_raises(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, feature_columns=["feature_a", "leak"], leakage_columns=["leak"])
    with pytest.raises(ValueError, match="LEAKAGE"):
        loader.load("test_model", verbose=False)


def test_target_in_features_raises(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, feature_columns=["feature_a", "target"])
    with pytest.raises(ValueError, match="target column"):
        loader.load("test_model", verbose=False)


def test_eligibility_filter_applied(tmp_path: Path) -> None:
    rows = default_rows(rows_per_day=2)
    rows[0]["training_feature_eligible"] = False
    rows[0]["feature_a"] = 999.0
    loader = make_loader(tmp_path, rows=rows)
    split = loader.load("test_model", verbose=False)
    all_features = pl.concat([split.train, split.val, split.test])
    assert 999.0 not in all_features["feature_a"].to_list()


def test_purge_removes_boundary_rows(tmp_path: Path) -> None:
    loader = make_loader(
        tmp_path,
        rows=default_rows(rows_per_day=2),
        split_overrides={"purge_days": 1, "min_train_rows": 1},
    )
    split = loader.load("test_model", verbose=False)
    assert split.n_train == 4
    assert split.metadata["train_val_purged_rows"] == 2


def test_dataset_hash_is_stable(tmp_path: Path) -> None:
    loader = make_loader(tmp_path)
    first = loader.load("test_model", verbose=False)
    second = loader.load("test_model", verbose=False)
    assert first.dataset_hash == second.dataset_hash
    assert len(first.dataset_hash) == 64


def test_dataset_hash_recorded_in_manifest(tmp_path: Path) -> None:
    split = DatasetSplit(
        train=pl.DataFrame({"feature_a": [1.0], "target": [1]}),
        val=pl.DataFrame({"feature_a": [2.0], "target": [0]}),
        test=pl.DataFrame({"feature_a": [3.0], "target": [1]}),
        feature_columns=["feature_a"],
        target_column="target",
        dataset_hash="a" * 64,
        split_id="split_v1",
        model_id="test_model",
        n_train=1,
        n_val=1,
        n_test=1,
        metadata={"dataset_variant": "default"},
    )
    artifact_path = tmp_path / "artifacts"
    ArtifactWriter().write_experiment_manifest(
        model_id="test_model",
        dataset_split=split,
        algorithm="unit",
        hyperparams={},
        artifact_path=artifact_path,
    )
    manifest = json.loads((artifact_path / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_provenance"]["dataset_hash"] == "a" * 64


def test_placeholder_feature_config_raises(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, feature_columns=[], feature_status="placeholder")
    with pytest.raises(ValueError, match="placeholder"):
        loader.load("test_model", verbose=False)


def test_split_temporal_monotone(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, rows=list(reversed(default_rows(rows_per_day=2))))
    split = loader.load("test_model", verbose=False)
    assert split.train["feature_a"].len() > 0
    assert split.n_train < split.n_train + split.n_val + split.n_test


def test_min_row_count_gate(tmp_path: Path) -> None:
    loader = make_loader(tmp_path, split_overrides={"min_train_rows": 999})
    with pytest.raises(ValueError, match="Insufficient training rows"):
        loader.load("test_model", verbose=False)


def test_missing_ordering_col_raises(tmp_path: Path) -> None:
    rows = default_rows()
    for row in rows:
        row.pop("ts")
    loader = make_loader(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="Ordering column"):
        loader.load("test_model", verbose=False)


def test_core_10_models_in_registry() -> None:
    """The original 10-model stack must stay registered.

    Later phases added derived models (04b markout-to-close, 08b big-move,
    08c maker-defense), so this is a subset check rather than an exact match.
    """
    registry = yaml.safe_load(Path("config/model_registry_v1.yaml").read_text(encoding="utf-8"))
    core = {
        "model_01_bias",
        "model_02_fair_resolution",
        "model_03_fill_probability",
        "model_04_adverse_selection",
        "model_05_closing_flip",
        "model_06_mispricing",
        "model_07_microstructure_direction",
        "model_08_move_size",
        "model_09_regime_encoder",
        "model_10_action_policy",
    }
    registered = set(registry["models"])
    assert core <= registered, core - registered


def test_no_overlap_feature_and_leakage_cols() -> None:
    for path in sorted(Path("config/feature_sets").glob("*_features_v1.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        features = set(config.get("feature_columns") or [])
        leakage = set(config.get("leakage_only_columns") or [])
        targets = {config["target_columns"]["primary"], *(config["target_columns"].get("auxiliary") or [])}
        targets.discard(None)
        assert not features & leakage, path
        assert not features & targets, path


def test_loader_runs_on_all_available_datasets() -> None:
    """Integration check against the materialised datasets.

    The datasets live under `data/`, which is not distributed with the
    repository (see docs/DATA_AVAILABILITY.md), so each model is skipped when
    its dataset is absent and the whole test is skipped when none are present.
    """
    loader = LeakageSafeLoader()
    loadable = [
        ("model_01_bias", "preopen"),
        ("model_02_fair_resolution", "coarse"),
        ("model_03_fill_probability", None),
        ("model_04_adverse_selection", None),
        ("model_05_closing_flip", "dense_close"),
        ("model_06_mispricing", "dense_close"),
        ("model_07_microstructure_direction", "tabular"),
        ("model_08_move_size", "tabular"),
    ]
    checked = 0
    for model_id, variant in loadable:
        try:
            split = loader.load(model_id, dataset_variant=variant, verbose=False)
        except ValueError as exc:
            if "does not exist" in str(exc):
                continue
            raise
        assert split.n_train > 0, model_id
        assert split.n_val > 0, model_id
        assert split.n_test > 0, model_id
        assert split.feature_columns, model_id
        checked += 1
    if checked == 0:
        pytest.skip("no materialised datasets available under data/datasets")
