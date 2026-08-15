from __future__ import annotations

import calendar
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

DAY_NS = 24 * 3600 * 1_000_000_000


@dataclass
class DatasetSplit:
    """Holds train/val/test DataFrames with metadata."""

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    feature_columns: list[str]
    target_column: str
    dataset_hash: str
    split_id: str
    model_id: str
    n_train: int
    n_val: int
    n_test: int
    metadata: dict[str, Any]


class LeakageSafeLoader:
    """
    Enforces leak-free dataset loading for all model training.

    Guarantees:
    1. No future data: train rows always precede val rows by ordering_col.
    2. No leakage columns in feature matrix; violations raise ValueError.
    3. Only whitelisted feature columns are returned to the model layer.
    4. Dataset hash is recorded for every load.
    5. Purge/embargo gaps are applied around temporal split boundaries.
    6. Training eligibility filters are applied before split.
    7. Minimum row count gates are enforced.
    """

    def __init__(
        self,
        model_registry_path: str | Path = "config/model_registry_v1.yaml",
        feature_sets_dir: str | Path = "config/feature_sets/",
        splits_dir: str | Path = "config/splits/",
    ) -> None:
        self.registry_path = Path(model_registry_path)
        self.registry = self._load_registry(self.registry_path)
        self.feature_sets_dir = Path(feature_sets_dir)
        self.splits_dir = Path(splits_dir)
        self._hash_cache: dict[Path, str] = {}

    def load(
        self,
        model_id: str,
        dataset_variant: str | None = None,
        target_override: str | None = None,
        verbose: bool = True,
    ) -> DatasetSplit:
        """
        Load a dataset for a model_id and return a leak-safe split.

        Args:
            model_id: must match a key in model_registry_v1.yaml.
            dataset_variant: optional variant such as "coarse", "preopen", or "tabular".
            target_override: override the feature config primary target.
            verbose: print summary statistics.

        Raises:
            ValueError: on leakage, policy, temporal split, or data sufficiency violations.
        """
        model_config = self._get_model_config(model_id)
        feature_config = self._load_feature_config(model_config)
        split_config = self._load_split_config(model_config)

        if feature_config.get("status") == "placeholder":
            raise ValueError(
                f"Model {model_id} has placeholder feature config. "
                f"Define feature_columns before training."
            )

        feature_cols = list(feature_config["feature_columns"])
        leakage_cols = list(feature_config["leakage_only_columns"])
        target_columns = feature_config["target_columns"]
        target_col = target_override or target_columns["primary"]

        self._validate_feature_contract(model_id, feature_cols, leakage_cols, target_columns)

        ordering_col = split_config["ordering_column"]
        dataset_path = self._resolve_dataset_path(model_config, dataset_variant)
        effective_variant = self._effective_dataset_variant(model_config, dataset_variant)
        dataset_hash = self._compute_dataset_hash(dataset_path)

        required_columns = self._required_columns(
            model_config=model_config,
            feature_columns=feature_cols,
            target_column=target_col,
            ordering_column=ordering_col,
        )
        df = self._load_parquet(dataset_path, required_columns, split_config=split_config)

        eligibility_column = self._apply_eligibility_filter(df)
        if eligibility_column:
            n_before = len(df)
            df = df.filter(pl.col(eligibility_column).fill_null(False))
            if verbose:
                print(
                    f"Eligibility filter ({eligibility_column}): "
                    f"{n_before} -> {len(df)} rows ({n_before - len(df)} dropped)"
                )

        df = self._apply_dataset_filter(df, model_config)

        if ordering_col not in df.columns:
            raise ValueError(
                f"Ordering column '{ordering_col}' not found in dataset. "
                f"Cannot ensure temporal ordering."
            )
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in dataset for model {model_id}. "
                f"Available columns: {df.columns[:30]}"
            )

        df = df.sort(ordering_col)
        self._verify_temporal_ordering(df, ordering_col, model_id)

        train_df, val_df, test_df, split_metadata = self._apply_split(
            df=df,
            split_config=split_config,
            ordering_col=ordering_col,
        )
        train_df, val_df, test_df, purge_metadata = self._apply_purge_embargo(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            split_config=split_config,
            ordering_col=ordering_col,
        )
        self._validate_row_counts(train_df, val_df, test_df, split_config, model_id)

        available_features = [column for column in feature_cols if column in df.columns]
        missing_features = [column for column in feature_cols if column not in df.columns]
        if not available_features:
            raise ValueError(f"No configured feature columns are available for model {model_id}.")

        available_leakage = set(available_features) & set(leakage_cols)
        if available_leakage:
            raise ValueError(
                f"CRITICAL LEAKAGE: leakage columns are present in the selected "
                f"feature matrix for {model_id}: {sorted(available_leakage)}"
            )

        self._verify_no_future_data(train_df, val_df, test_df, ordering_col, model_id)

        if verbose and missing_features:
            print(
                f"WARNING: {len(missing_features)} configured feature columns are "
                f"missing from this dataset variant: {missing_features[:10]}"
            )
        if verbose:
            self._print_summary(
                model_id=model_id,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                features=available_features,
                target=target_col,
                dataset_hash=dataset_hash,
            )

        # === FEATURE-CLEANUP HOOK ===
        # When FEATURE_CLEANUP_ENABLED=1 is set in the environment, apply the
        # trojan-feature cleanup before returning the split.  Drops/transforms
        # `live_btc_usd`, `price_to_beat`, `live_applied_bias_age_seconds`.
        # See `src/feature_cleanup.py` and `docs/feature_cleanup_runbook.md`.
        import os
        if os.environ.get("FEATURE_CLEANUP_ENABLED") == "1":
            from feature_cleanup import clean_features, cleaned_feature_list
            train_df = clean_features(train_df)
            val_df = clean_features(val_df)
            test_df = clean_features(test_df)
            available_features = cleaned_feature_list(available_features)
            if verbose:
                print(f"  [FEATURE-CLEANUP] applied; {len(available_features)} features after cleanup")
        # === END HOOK ===

        select_columns = available_features + [target_col]
        return DatasetSplit(
            train=train_df.select(select_columns),
            val=val_df.select(select_columns),
            test=test_df.select(select_columns),
            feature_columns=available_features,
            target_column=target_col,
            dataset_hash=dataset_hash,
            split_id=split_config["split_id"],
            model_id=model_id,
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            metadata={
                "dataset_variant": effective_variant,
                "dataset_path": str(dataset_path),
                "ordering_column": ordering_col,
                "purge_days": split_config.get("purge_days", 0),
                "embargo_days": split_config.get("embargo_days", 0),
                "eligibility_column": eligibility_column,
                "dataset_filter": model_config.get("dataset", {}).get("filter"),
                "missing_feature_columns": missing_features,
                "feature_config_version": feature_config["version"],
                "split_config_version": split_config["version"],
                **split_metadata,
                **purge_metadata,
            },
        )

    def _load_registry(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ValueError(f"Model registry not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            registry = yaml.safe_load(handle) or {}
        if "models" not in registry or not isinstance(registry["models"], dict):
            raise ValueError(f"Model registry {path} must contain a mapping named 'models'.")
        return registry

    def _get_model_config(self, model_id: str) -> dict[str, Any]:
        try:
            return self.registry["models"][model_id]
        except KeyError as exc:
            raise ValueError(f"Unknown model_id: {model_id}") from exc

    def _load_feature_config(self, model_config: dict[str, Any]) -> dict[str, Any]:
        configured = model_config["feature_set"]["config"]
        path = Path(configured)
        if not path.is_absolute():
            path = Path(configured)
        return self._load_feature_config_from_path(path)

    def _load_feature_config_from_path(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Feature config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        required = [
            "feature_set_id",
            "model_id",
            "version",
            "feature_columns",
            "leakage_only_columns",
            "target_columns",
        ]
        missing = [field for field in required if field not in config]
        if missing:
            raise ValueError(f"Feature config {path} missing required fields: {missing}")

        self._validate_feature_contract(
            model_id=str(config["model_id"]),
            feature_columns=list(config.get("feature_columns") or []),
            leakage_columns=list(config.get("leakage_only_columns") or []),
            target_columns=dict(config.get("target_columns") or {}),
        )
        return config

    def _load_split_config(self, model_config: dict[str, Any]) -> dict[str, Any]:
        configured = model_config["split"]["config"]
        path = Path(configured)
        if not path.exists():
            fallback = self.splits_dir / Path(configured).name
            path = fallback if fallback.exists() else path
        if not path.exists():
            raise ValueError(f"Split config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _validate_feature_contract(
        self,
        model_id: str,
        feature_columns: list[str],
        leakage_columns: list[str],
        target_columns: dict[str, Any],
    ) -> None:
        leakage_in_features = set(feature_columns) & set(leakage_columns)
        if leakage_in_features:
            raise ValueError(
                f"CRITICAL LEAKAGE: columns are in BOTH feature_columns and "
                f"leakage_only_columns for model {model_id}: {sorted(leakage_in_features)}"
            )

        targets = [target_columns.get("primary")]
        targets.extend(target_columns.get("auxiliary") or [])
        target_in_features = set(filter(None, targets)) & set(feature_columns)
        if target_in_features:
            raise ValueError(
                f"CRITICAL LEAKAGE: target column(s) are in feature_columns for "
                f"model {model_id}: {sorted(target_in_features)}"
            )

    def _resolve_dataset_path(
        self,
        model_config: dict[str, Any],
        dataset_variant: str | None,
    ) -> Path:
        dataset = model_config["dataset"]
        paths = dataset.get("paths") or {}
        if not paths:
            raise ValueError(f"Model {model_config['model_id']} has no dataset paths configured.")
        if dataset_variant:
            if dataset_variant not in paths:
                raise ValueError(
                    f"Dataset variant '{dataset_variant}' not configured for "
                    f"{model_config['model_id']}. Available: {sorted(paths)}"
                )
            target_path = Path(paths[dataset_variant])
        elif "default" in paths:
            target_path = Path(paths["default"])
        else:
            variants = dataset.get("variants") or list(paths)
            target_path = Path(paths[variants[0]])
        if not target_path.exists():
            raise ValueError(f"Dataset path does not exist: {target_path}")
        return target_path

    def _effective_dataset_variant(
        self,
        model_config: dict[str, Any],
        dataset_variant: str | None,
    ) -> str:
        if dataset_variant:
            return dataset_variant
        paths = model_config["dataset"].get("paths") or {}
        if "default" in paths:
            return "default"
        variants = model_config["dataset"].get("variants") or list(paths)
        return variants[0]

    def _parquet_files(self, dataset_path: Path) -> list[Path]:
        if dataset_path.is_file():
            files = [dataset_path]
        else:
            files = sorted(dataset_path.rglob("*.parquet"))
        if not files:
            raise ValueError(f"No parquet files found under {dataset_path}")
        return files

    def _compute_dataset_hash(self, dataset_path: Path) -> str:
        resolved = dataset_path.resolve()
        if resolved in self._hash_cache:
            return self._hash_cache[resolved]

        hasher = hashlib.sha256()
        for parquet_file in self._parquet_files(dataset_path):
            relative = parquet_file.relative_to(dataset_path) if dataset_path.is_dir() else parquet_file.name
            hasher.update(str(relative).replace("\\", "/").encode("utf-8"))
            with parquet_file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        digest = hasher.hexdigest()
        self._hash_cache[resolved] = digest
        return digest

    def _required_columns(
        self,
        model_config: dict[str, Any],
        feature_columns: list[str],
        target_column: str,
        ordering_column: str,
    ) -> list[str]:
        required = set(feature_columns)
        required.add(target_column)
        required.add(ordering_column)
        required.update(
            [
                "training_feature_eligible",
                "sequence_feature_eligible",
                "execution_training_eligible",
                "full_matrix_training_eligible",
            ]
        )
        dataset_filter = model_config.get("dataset", {}).get("filter")
        if dataset_filter:
            filter_col = self._parse_simple_filter(dataset_filter)[0]
            required.add(filter_col)
        return sorted(required)

    def _load_parquet(self, dataset_path: Path, required_columns: list[str],
                      split_config: dict[str, Any] | None = None) -> pl.DataFrame:
        files = self._parquet_files(dataset_path)

        # === MEMORY FIX (OOM on fill_probability / adverse_selection) ===
        # Two safety valves:
        #
        # 1. MAX_LOAD_DAYS_FROM_END env var: cap the number of most-recent
        #    daily shards loaded.  Use when auto_extend would otherwise
        #    consume all available history (currently 28 shards x ~3M rows
        #    for execution_intent_dataset_v1 = 84M rows, > available RAM).
        #
        # 2. Static-split date filtering: when split_config has explicit
        #    start_day/end_day for train/val/test, only load shards within
        #    the union of those date ranges.  Skipped when auto_extend=True
        #    because the dynamic split intentionally consumes all data.
        import os as _os
        max_days = _os.environ.get("MAX_LOAD_DAYS_FROM_END")
        if max_days:
            try:
                n = int(max_days)
                date_stem_files = []
                for f in files:
                    try:
                        from datetime import date as _date
                        _date.fromisoformat(f.stem)
                        date_stem_files.append(f)
                    except ValueError:
                        pass
                if date_stem_files and n > 0:
                    files = sorted(date_stem_files)[-n:]
            except (ValueError, TypeError):
                pass

        if split_config is not None and not split_config.get("auto_extend", False):
            try:
                from datetime import date as _date
                def _parse(s):
                    return _date.fromisoformat(str(s)) if s else None
                bounds = []
                for section in ("train", "validation", "test"):
                    cfg = split_config.get(section, {})
                    d0, d1 = _parse(cfg.get("start_day")), _parse(cfg.get("end_day"))
                    if d0 and d1:
                        bounds.append((d0, d1))
                if bounds:
                    lo = min(b[0] for b in bounds)
                    hi = max(b[1] for b in bounds)
                    filtered = []
                    for f in files:
                        stem = f.stem
                        try:
                            d = _date.fromisoformat(stem)
                        except ValueError:
                            filtered.append(f)  # unparseable stem — keep
                            continue
                        if lo <= d <= hi:
                            filtered.append(f)
                    if filtered:
                        files = filtered
            except Exception:
                pass  # any failure → fall back to loading all files
        # === END MEMORY FIX ===

        schema = pl.scan_parquet([str(path) for path in files]).collect_schema()
        available = set(schema.names())
        columns = [column for column in required_columns if column in available]
        if not columns:
            raise ValueError(f"None of the requested columns exist in {dataset_path}")
        return pl.read_parquet([str(path) for path in files], columns=columns)

    def _apply_eligibility_filter(self, df: pl.DataFrame) -> str | None:
        for column in [
            "training_feature_eligible",
            "sequence_feature_eligible",
            "execution_training_eligible",
            "full_matrix_training_eligible",
        ]:
            if column in df.columns:
                return column
        return None

    def _apply_dataset_filter(self, df: pl.DataFrame, model_config: dict[str, Any]) -> pl.DataFrame:
        dataset_filter = model_config.get("dataset", {}).get("filter")
        if not dataset_filter:
            return df
        column, op, value = self._parse_simple_filter(dataset_filter)
        if column not in df.columns:
            raise ValueError(f"Dataset filter column '{column}' not found for {model_config['model_id']}")
        expr = pl.col(column) == value if op == "==" else pl.col(column) != value
        return df.filter(expr)

    def _parse_simple_filter(self, expression: str) -> tuple[str, str, Any]:
        match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=)\s*(True|False|null|[-0-9.]+|'[^']*'|\"[^\"]*\")\s*", expression)
        if not match:
            raise ValueError(f"Unsupported dataset filter expression: {expression}")
        column, op, raw_value = match.groups()
        if raw_value == "True":
            value: Any = True
        elif raw_value == "False":
            value = False
        elif raw_value == "null":
            value = None
        elif raw_value.startswith(("'", '"')):
            value = raw_value[1:-1]
        elif "." in raw_value:
            value = float(raw_value)
        else:
            value = int(raw_value)
        return column, op, value

    def _apply_split(
        self,
        df: pl.DataFrame,
        split_config: dict[str, Any],
        ordering_col: str,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
        strategy = split_config.get("split_strategy", "day_based")
        if strategy != "day_based":
            raise ValueError(f"Unknown split strategy: {strategy}")

        train_days, val_days, test_days = self._effective_split_days(df, split_config, ordering_col)
        train_df = self._filter_day_range(df, ordering_col, train_days[0], train_days[-1])
        val_df = self._filter_day_range(df, ordering_col, val_days[0], val_days[-1])
        test_df = self._filter_day_range(df, ordering_col, test_days[0], test_days[-1])
        return train_df, val_df, test_df, {
            "train_days": train_days,
            "validation_days": val_days,
            "test_days": test_days,
        }

    def _effective_split_days(
        self,
        df: pl.DataFrame,
        split_config: dict[str, Any],
        ordering_col: str,
    ) -> tuple[list[str], list[str], list[str]]:
        if split_config.get("auto_extend", False):
            available_days = self._available_days(df, ordering_col)
            val_count = int(split_config.get("auto_extend_val_days", 1))
            test_count = int(split_config.get("auto_extend_test_days", 1))
            if len(available_days) > val_count + test_count:
                train_days = available_days[: -(val_count + test_count)]
                val_days = available_days[-(val_count + test_count) : -test_count]
                test_days = available_days[-test_count:]
                return train_days, val_days, test_days

        return (
            self._inclusive_days(split_config["train"]["start_day"], split_config["train"]["end_day"]),
            self._inclusive_days(
                split_config["validation"]["start_day"],
                split_config["validation"]["end_day"],
            ),
            self._inclusive_days(split_config["test"]["start_day"], split_config["test"]["end_day"]),
        )

    def _available_days(self, df: pl.DataFrame, ordering_col: str) -> list[str]:
        day_numbers = (
            df.select((pl.col(ordering_col) // DAY_NS).unique().sort())
            .to_series()
            .to_list()
        )
        return [
            datetime.fromtimestamp(int(day_number) * 24 * 3600, tz=timezone.utc).date().isoformat()
            for day_number in day_numbers
        ]

    def _filter_day_range(
        self,
        df: pl.DataFrame,
        ordering_col: str,
        start_day: str,
        end_day: str,
    ) -> pl.DataFrame:
        return df.filter(
            (pl.col(ordering_col) >= self._day_to_start_ns(start_day))
            & (pl.col(ordering_col) <= self._day_to_end_ns(end_day))
        )

    def _apply_purge_embargo(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        split_config: dict[str, Any],
        ordering_col: str,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, Any]]:
        purge_days = int(split_config.get("purge_days", 0))
        embargo_days = int(split_config.get("embargo_days", 0))
        gap_days = purge_days + embargo_days
        metadata: dict[str, Any] = {"skipped_purge_boundaries": []}
        if gap_days <= 0:
            return train_df, val_df, test_df, metadata

        gap_ns = gap_days * DAY_NS
        if len(val_df) > 0:
            val_start = val_df[ordering_col].min()
            before = len(train_df)
            train_df = train_df.filter(pl.col(ordering_col) < val_start - gap_ns)
            metadata["train_val_purged_rows"] = before - len(train_df)

        if len(test_df) > 0 and len(val_df) > 0:
            test_start = test_df[ordering_col].min()
            candidate = val_df.filter(pl.col(ordering_col) < test_start - gap_ns)
            min_val = int(split_config.get("min_val_rows", 0))
            preserve_short = bool(split_config.get("preserve_validation_rows_on_short_window", False))
            if preserve_short and len(candidate) < min_val <= len(val_df):
                metadata["skipped_purge_boundaries"].append("validation_test")
                metadata["validation_test_purged_rows"] = 0
            else:
                metadata["validation_test_purged_rows"] = len(val_df) - len(candidate)
                val_df = candidate

        return train_df, val_df, test_df, metadata

    def _validate_row_counts(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        split_config: dict[str, Any],
        model_id: str,
    ) -> None:
        min_train = int(split_config.get("min_train_rows", 50))
        min_val = int(split_config.get("min_val_rows", 10))
        min_test = int(split_config.get("min_test_rows", 10))
        if len(train_df) < min_train:
            raise ValueError(f"Insufficient training rows for {model_id}: {len(train_df)} < {min_train}")
        if len(val_df) < min_val:
            raise ValueError(f"Insufficient validation rows for {model_id}: {len(val_df)} < {min_val}")
        if len(test_df) < min_test:
            raise ValueError(f"Insufficient test rows for {model_id}: {len(test_df)} < {min_test}")

    def _verify_temporal_ordering(self, df: pl.DataFrame, ordering_col: str, model_id: str) -> None:
        if len(df) <= 1:
            return
        violations = df.select((pl.col(ordering_col).diff() < 0).sum()).item()
        if violations:
            raise ValueError(
                f"CRITICAL: {violations} rows where {ordering_col} decreases for "
                f"model {model_id}. Dataset is not temporally ordered."
            )

    def _verify_no_future_data(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        ordering_col: str,
        model_id: str,
    ) -> None:
        if len(train_df) > 0 and len(val_df) > 0:
            max_train = train_df[ordering_col].max()
            min_val = val_df[ordering_col].min()
            if max_train >= min_val:
                raise ValueError(
                    f"CRITICAL: Future data violation for {model_id}. "
                    f"max(train.{ordering_col})={max_train} >= min(val.{ordering_col})={min_val}."
                )
        if len(val_df) > 0 and len(test_df) > 0:
            max_val = val_df[ordering_col].max()
            min_test = test_df[ordering_col].min()
            if max_val >= min_test:
                raise ValueError(
                    f"CRITICAL: Future data violation for {model_id}. "
                    f"max(val.{ordering_col})={max_val} >= min(test.{ordering_col})={min_test}."
                )

    def _print_summary(
        self,
        model_id: str,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: pl.DataFrame,
        features: list[str],
        target: str,
        dataset_hash: str,
    ) -> None:
        print(f"\n{'=' * 60}")
        print(f"LeakageSafeLoader: {model_id}")
        print(f"{'=' * 60}")
        print(f"Dataset hash:  {dataset_hash[:16]}...")
        print(f"Features:      {len(features)} columns")
        print(f"Target:        {target}")
        print(f"Train rows:    {len(train_df)}")
        print(f"Val rows:      {len(val_df)}")
        print(f"Test rows:     {len(test_df)}")
        if target in train_df.columns and len(train_df) > 0:
            try:
                train_mean = train_df.select(pl.col(target).cast(pl.Float64).mean()).item()
                val_mean = val_df.select(pl.col(target).cast(pl.Float64).mean()).item() if len(val_df) else None
                print(f"Train target mean: {train_mean:.6f}")
                if val_mean is not None:
                    print(f"Val target mean:   {val_mean:.6f}")
            except Exception:
                pass
        print(f"{'=' * 60}\n")

    def _day_to_start_ns(self, day: str) -> int:
        parsed = date.fromisoformat(day)
        dt = datetime.combine(parsed, time.min, tzinfo=timezone.utc)
        return calendar.timegm(dt.utctimetuple()) * 1_000_000_000

    def _day_to_end_ns(self, day: str) -> int:
        return self._day_to_start_ns(day) + DAY_NS - 1

    def _inclusive_days(self, start_day: str, end_day: str) -> list[str]:
        start_num = self._day_to_start_ns(start_day) // DAY_NS
        end_num = self._day_to_start_ns(end_day) // DAY_NS
        return [
            datetime.fromtimestamp(day_num * 24 * 3600, tz=timezone.utc).date().isoformat()
            for day_num in range(int(start_num), int(end_num) + 1)
        ]
