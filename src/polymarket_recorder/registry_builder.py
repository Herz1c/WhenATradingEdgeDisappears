from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from binance_recorder.utils import ns_to_iso
from market_recorders.dataset_policy import SCRIPTURE_EXPLICIT_SOURCE_HOUR_QUARANTINES, load_dataset_policy
from market_recorders.source_registry import infer_kind_from_path
from market_recorders.unified_reader import UnifiedRawReader

SECOND_NS = 1_000_000_000
DEFAULT_DATE_FROM = date(2026, 4, 19)
DEFAULT_DATE_TO = date(2026, 4, 22)


SOURCE_DESCRIPTORS: dict[str, dict[str, Any]] = {
    "polymarket_resolution": {
        "role": "primary",
        "default_include": True,
        "source_tier": "primary_training",
        "remarks": "Authoritative supervised label layer.",
    },
    "polymarket_strike": {
        "role": "primary",
        "default_include": True,
        "source_tier": "primary_training",
        "remarks": "Price-to-beat strike layer; only label_safe rows may define labels.",
    },
    "polymarket_l2": {
        "role": "primary",
        "default_include": True,
        "source_tier": "primary_training",
        "remarks": "Canonical Polymarket microstructure layer.",
    },
    "polymarket_ws": {
        "role": "primary",
        "default_include": True,
        "source_tier": "primary_training",
        "remarks": "Raw market websocket layer preserved for audit and later sequence expansion.",
    },
    "polymarket_rest": {
        "role": "context",
        "default_include": True,
        "source_tier": "secondary_context",
        "remarks": "Secondary context snapshots; recv_ts_ns remains the only shared join clock.",
    },
    "polymarket_discovery": {
        "role": "debug",
        "default_include": False,
        "source_tier": "debug_only",
        "remarks": "Discovery telemetry only.",
    },
    "polymarket_lifecycle": {
        "role": "debug",
        "default_include": False,
        "source_tier": "debug_only",
        "remarks": "Recorder lifecycle telemetry only.",
    },
    "polymarket_history": {
        "role": "debug",
        "default_include": False,
        "source_tier": "debug_only",
        "remarks": "Resolution history audit layer only.",
    },
    "polymarket_rtds": {
        "role": "debug",
        "default_include": False,
        "source_tier": "debug_only",
        "remarks": "Permanently excluded; structurally unreliable and affected by silent-freeze behavior.",
    },
    "binance_spot": {
        "role": "primary",
        "default_include": True,
        "source_tier": "primary_training",
        "remarks": "Primary BTC price source.",
    },
    "binance_usdm": {
        "role": "primary",
        "default_include": True,
        "source_tier": "primary_training",
        "remarks": "Secondary BTC price source with scripture-defined quarantine on 2026-04-20 07:00 UTC.",
    },
    "hyperliquid": {
        "role": "primary",
        "default_include": True,
        "source_tier": "primary_training",
        "remarks": "Secondary robustness source for the synthetic Chainlink median.",
    },
    "chainlink_public_delayed": {
        "role": "verification",
        "default_include": True,
        "source_tier": "verification_only",
        "remarks": "Calibration-only delayed input with a strict 1800-second causal floor.",
    },
    "chainlink_onchain": {
        "role": "verification",
        "default_include": False,
        "source_tier": "verification_only",
        "remarks": "Verification-only cadence; not a live-equivalent source.",
    },
    "chainlink_live": {
        "role": "debug",
        "default_include": False,
        "source_tier": "debug_only",
        "remarks": "Excluded from the public stack; not public and not part of the approved source set.",
    },
    "bybit": {
        "role": "debug",
        "default_include": False,
        "source_tier": "debug_only",
        "remarks": "Permanently excluded due to continuity issues.",
    },
    "coinbase_advanced": {
        "role": "debug",
        "default_include": False,
        "source_tier": "debug_only",
        "remarks": "Permanently excluded because continuity_compromised behavior remains structural.",
    },
    "deribit": {
        "role": "debug",
        "default_include": False,
        "source_tier": "debug_only",
        "remarks": "Permanently excluded; complexity is not justified in the current phase.",
    },
    "fng": {
        "role": "debug",
        "default_include": False,
        "source_tier": "debug_only",
        "remarks": "Permanently excluded; slow prior with -0.0079 correlation to 5-minute resolution outcome.",
    },
}


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _date_range(date_from: date, date_to: date) -> list[date]:
    values: list[date] = []
    current = date_from
    while current <= date_to:
        values.append(current)
        current += timedelta(days=1)
    return values


def _available_archive_dates(*, root: Path, relative_prefix: Path) -> list[date]:
    folder = root / "raw" / relative_prefix
    if not folder.exists():
        return []
    values: list[date] = []
    for child in folder.iterdir():
        if not child.is_dir():
            continue
        try:
            values.append(date.fromisoformat(child.name))
        except ValueError:
            continue
    return sorted(values)


def _file_hour(path: Path) -> str:
    return path.name.split(".")[0].split("__", 1)[0]


def _kind_for_relative_path(relative: Path) -> str:
    return infer_kind_from_path(relative) or "unknown"


def _normalize_source_name(relative: Path) -> str:
    parts = relative.parts
    kind = _kind_for_relative_path(relative)
    if not parts:
        return "unknown"
    root = parts[0]
    if root == "binance":
        venue = parts[1] if len(parts) > 1 else "unknown"
        return f"binance_{venue}"
    if root == "hyperliquid":
        return "hyperliquid"
    if root == "polymarket":
        if len(parts) > 1 and parts[1] == "resolution":
            return "polymarket_resolution" if kind == "resolution" else f"polymarket_{kind}"
        if len(parts) > 1 and parts[1] == "rtds":
            return "polymarket_rtds"
        return {
            "ws": "polymarket_ws",
            "l2": "polymarket_l2",
            "rest": "polymarket_rest",
            "strike": "polymarket_strike",
            "discovery": "polymarket_discovery",
            "lifecycle": "polymarket_lifecycle",
            "history": "polymarket_history",
        }.get(kind, f"polymarket_{kind}")
    if root == "chainlink_public_delayed":
        return "chainlink_public_delayed"
    if root == "chainlink":
        backend = parts[1] if len(parts) > 1 else ""
        if backend == "onchain":
            return "chainlink_onchain"
        if backend == "live":
            return "chainlink_live"
        return f"chainlink_{backend or kind}"
    if root == "bybit":
        return "bybit"
    if root == "coinbase":
        return "coinbase_advanced"
    if root == "deribit":
        return "deribit"
    if root == "fng":
        return "fng"
    return root


def _descriptor_for_source(source_name: str) -> dict[str, Any]:
    return dict(
        SOURCE_DESCRIPTORS.get(
            source_name,
            {
                "role": "debug",
                "default_include": False,
                "source_tier": "debug_only",
                "remarks": "Unclassified source; treated as excluded until explicitly approved.",
            },
        )
    )


def _relative_raw_path_from_manifest_file_path(file_path: str) -> Path:
    normalized = file_path.replace("\\", "/")
    marker = "/raw/"
    if marker in normalized:
        return Path(normalized.split(marker, 1)[1])
    if normalized.startswith("data/raw/"):
        return Path(normalized.removeprefix("data/raw/"))
    return Path(normalized)


def _manifest_path_for_raw_path(root: Path, raw_path: Path) -> Path:
    relative = raw_path.relative_to(root / "raw")
    return (root / "manifests" / relative).with_suffix("").with_suffix(".manifest.json")


def _normalize_quality_class(value: str | None) -> str:
    if value == "finalized_clean":
        return "clean"
    return value or "unknown"


def _canonical_asset_mapping(record: dict[str, Any]) -> tuple[str | None, str | None]:
    asset_ids = [str(value) for value in record.get("selected_asset_ids", []) if value is not None]
    outcomes = [str(value).lower() for value in record.get("selected_outcomes", []) if value is not None]
    mapping = {
        outcome: asset_id
        for asset_id, outcome in zip(asset_ids, outcomes, strict=False)
    }
    return mapping.get("up"), mapping.get("down")


def _scan_strike_records(
    *,
    root: Path,
    policy_path: Path,
    date_from: date,
    date_to: date,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    policy = load_dataset_policy(policy_path)
    reader = UnifiedRawReader(root=root)
    strike_prefix = Path("polymarket") / "btc_updown_5m"
    earliest_strikes: dict[str, dict[str, Any]] = {}
    exclusions: dict[str, Counter[str]] = defaultdict(Counter)
    scan_start = date_from - timedelta(days=1)
    scan_dates = [
        shard_date
        for shard_date in _available_archive_dates(root=root, relative_prefix=strike_prefix)
        if scan_start <= shard_date <= date_to
    ] or _date_range(scan_start, date_to)

    for target_date in scan_dates:
        for path in reader.list_files_for_date(
            relative_prefix=strike_prefix,
            target_date=target_date,
            kind="strike",
            finalized_only=False,
        ):
            metadata = reader.file_metadata(path)
            eligible, reason = True, None
            if metadata.get("quality_class") != "finalized_clean":
                eligible, reason = False, f"manifest_quality_class:{metadata.get('quality_class') or 'unknown'}"
            elif metadata.get("file_state") != "finalized":
                eligible, reason = False, f"file_state:{metadata.get('file_state') or 'unknown'}"
            if not eligible:
                exclusions["polymarket_strike"][reason or "unknown"] += 1
                continue
            for record in reader.iter_file(path, finalized_only=True, tail_tolerant=False):
                if record.get("record_type") != "strike_observation":
                    continue
                if record.get("label_safe") is not True:
                    continue
                market_slug = str(record.get("selected_market_slug") or "").strip()
                market_open_s = _to_int(record.get("market_open_s"))
                market_close_s = _to_int(record.get("market_close_s"))
                recv_ts_ns = _to_int(record.get("recv_ts_ns"))
                strike_price = _to_float(record.get("price_to_beat"))
                if (
                    not market_slug
                    or market_open_s is None
                    or market_close_s is None
                    or recv_ts_ns is None
                    or strike_price is None
                ):
                    continue
                current = earliest_strikes.get(market_slug)
                if current is not None and recv_ts_ns >= int(current["strike_recv_ts_ns"]):
                    continue
                up_asset_id, down_asset_id = _canonical_asset_mapping(record)
                earliest_strikes[market_slug] = {
                    "market_slug": market_slug,
                    "market_id": str(record.get("selected_market_id") or ""),
                    "condition_id": str(record.get("selected_condition_id") or ""),
                    "selected_event_id": str(record.get("selected_event_id") or ""),
                    "selected_event_slug": str(record.get("selected_event_slug") or ""),
                    "market_open_s": int(market_open_s),
                    "market_close_s": int(market_close_s),
                    "market_interval_seconds": int(market_close_s - market_open_s),
                    "price_to_beat": float(strike_price),
                    "strike_recv_ts_ns": int(recv_ts_ns),
                    "strike_label_safe": True,
                    "up_token_asset_id": up_asset_id,
                    "down_token_asset_id": down_asset_id,
                    "asset_ids": [str(value) for value in record.get("selected_asset_ids", []) if value is not None],
                    "outcomes": [str(value) for value in record.get("selected_outcomes", []) if value is not None],
                }
    return earliest_strikes, {key: dict(value) for key, value in exclusions.items()}


def _scan_resolution_records(
    *,
    root: Path,
    policy_path: Path,
    date_from: date,
    date_to: date,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    reader = UnifiedRawReader(root=root)
    resolution_prefix = Path("polymarket") / "resolution" / "btc_updown_5m"
    by_slug: dict[str, dict[str, Any]] = {}
    by_condition_id: dict[str, dict[str, Any]] = {}
    exclusions: dict[str, Counter[str]] = defaultdict(Counter)
    scan_dates = [
        shard_date
        for shard_date in _available_archive_dates(root=root, relative_prefix=resolution_prefix)
        if date_from <= shard_date <= date_to + timedelta(days=1)
    ] or _date_range(date_from, date_to + timedelta(days=1))

    for target_date in scan_dates:
        for path in reader.list_files_for_date(
            relative_prefix=resolution_prefix,
            target_date=target_date,
            kind="resolution",
            finalized_only=False,
        ):
            metadata = reader.file_metadata(path)
            eligible, reason = True, None
            if metadata.get("quality_class") != "finalized_clean":
                eligible, reason = False, f"manifest_quality_class:{metadata.get('quality_class') or 'unknown'}"
            elif metadata.get("file_state") != "finalized":
                eligible, reason = False, f"file_state:{metadata.get('file_state') or 'unknown'}"
            if not eligible:
                exclusions["polymarket_resolution"][reason or "unknown"] += 1
                continue
            for record in reader.iter_file(path, finalized_only=True, tail_tolerant=False):
                if record.get("record_type") != "market_resolution":
                    continue
                market_slug = str(record.get("market_slug") or "").strip()
                condition_id = str(record.get("condition_id") or "").strip()
                resolved_side = str(record.get("resolved_side") or "").strip().lower()
                recv_ts_ns = _to_int(record.get("recv_ts_ns"))
                resolved_ts_ns = _to_int(record.get("resolved_ts_ns"))
                if resolved_side not in {"up", "down"} or recv_ts_ns is None or resolved_ts_ns is None:
                    continue
                candidate = {
                    "market_slug": market_slug,
                    "condition_id": condition_id,
                    "resolved_side": resolved_side,
                    "resolution_recv_ts_ns": int(recv_ts_ns),
                    "resolved_ts_ns": int(resolved_ts_ns),
                }
                rank = (candidate["resolved_ts_ns"], candidate["resolution_recv_ts_ns"])
                if market_slug:
                    current = by_slug.get(market_slug)
                    current_rank = (
                        (int(current["resolved_ts_ns"]), int(current["resolution_recv_ts_ns"]))
                        if current is not None
                        else None
                    )
                    if current_rank is None or rank < current_rank:
                        by_slug[market_slug] = dict(candidate)
                if condition_id:
                    current = by_condition_id.get(condition_id)
                    current_rank = (
                        (int(current["resolved_ts_ns"]), int(current["resolution_recv_ts_ns"]))
                        if current is not None
                        else None
                    )
                    if current_rank is None or rank < current_rank:
                        by_condition_id[condition_id] = dict(candidate)
    return by_slug, by_condition_id, {key: dict(value) for key, value in exclusions.items()}


def _build_market_rows(
    *,
    earliest_strikes: dict[str, dict[str, Any]],
    resolution_by_slug: dict[str, dict[str, Any]],
    resolution_by_condition_id: dict[str, dict[str, Any]],
    date_from: date,
    date_to: date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    market_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []

    for market_slug, strike in sorted(
        earliest_strikes.items(),
        key=lambda item: (int(item[1]["market_open_s"]), item[0]),
    ):
        market_day = datetime.fromtimestamp(int(strike["market_open_s"]), tz=UTC).date()
        if market_day < date_from or market_day > date_to:
            continue

        resolution = resolution_by_slug.get(market_slug)
        label_join_quality = ""
        if resolution is not None:
            label_join_quality = "market_slug_exact"
        elif strike["condition_id"]:
            resolution = resolution_by_condition_id.get(strike["condition_id"])
            if resolution is not None:
                label_join_quality = "condition_id_fallback"

        has_canonical_resolution = resolution is not None
        resolved_side = str(resolution["resolved_side"]).title() if resolution is not None else None
        winning_asset_id = None
        if resolved_side == "Up":
            winning_asset_id = strike["up_token_asset_id"]
        elif resolved_side == "Down":
            winning_asset_id = strike["down_token_asset_id"]

        exclusion_reason = ""
        if not has_canonical_resolution:
            exclusion_reason = "missing_canonical_resolution"
        elif not strike["up_token_asset_id"] or not strike["down_token_asset_id"]:
            exclusion_reason = "missing_token_mapping"
        elif int(strike["market_interval_seconds"]) != 300:
            exclusion_reason = "non_5min_market"

        is_trainable = exclusion_reason == ""
        market_open_ts_ns = int(strike["market_open_s"]) * SECOND_NS
        market_close_ts_ns = int(strike["market_close_s"]) * SECOND_NS
        market_row = {
            "market_slug": market_slug,
            "market_id": strike["market_id"],
            "condition_id": strike["condition_id"],
            "selected_event_id": strike["selected_event_id"],
            "selected_event_slug": strike["selected_event_slug"],
            "market_open_ts_ns": market_open_ts_ns,
            "market_open_iso": ns_to_iso(market_open_ts_ns),
            "market_close_ts_ns": market_close_ts_ns,
            "market_close_iso": ns_to_iso(market_close_ts_ns),
            "market_interval_seconds": int(strike["market_interval_seconds"]),
            "asset_ids": json.dumps(strike["asset_ids"], separators=(",", ":")),
            "outcomes": json.dumps(strike["outcomes"], separators=(",", ":")),
            "up_token_asset_id": strike["up_token_asset_id"] or "",
            "down_token_asset_id": strike["down_token_asset_id"] or "",
            "price_to_beat": float(strike["price_to_beat"]),
            "strike_recv_ts_ns": int(strike["strike_recv_ts_ns"]),
            "strike_label_safe": True,
            "resolved_side": resolved_side or "",
            "winning_asset_id": winning_asset_id or "",
            "resolved_ts_ns": int(resolution["resolved_ts_ns"]) if resolution is not None else pd.NA,
            "has_canonical_resolution": bool(has_canonical_resolution),
            "is_trainable": bool(is_trainable),
            "exclusion_reason": exclusion_reason,
        }
        market_rows.append(market_row)

        if strike["up_token_asset_id"]:
            token_rows.append(
                {
                    "market_slug": market_slug,
                    "token_asset_id": strike["up_token_asset_id"],
                    "token_outcome": "Up",
                    "token_side_normalized": "UP",
                    "active_from_ts_ns": market_open_ts_ns,
                    "active_to_ts_ns": market_close_ts_ns,
                }
            )
        if strike["down_token_asset_id"]:
            token_rows.append(
                {
                    "market_slug": market_slug,
                    "token_asset_id": strike["down_token_asset_id"],
                    "token_outcome": "Down",
                    "token_side_normalized": "DOWN",
                    "active_from_ts_ns": market_open_ts_ns,
                    "active_to_ts_ns": market_close_ts_ns,
                }
            )

        if has_canonical_resolution:
            label_rows.append(
                {
                    "market_slug": market_slug,
                    "condition_id": strike["condition_id"],
                    "price_to_beat": float(strike["price_to_beat"]),
                    "strike_source": "polymarket.strike",
                    "strike_recv_ts_ns": int(strike["strike_recv_ts_ns"]),
                    "strike_label_safe": True,
                    "resolved_side": str(resolution["resolved_side"]).title(),
                    "resolution_source": "polymarket.resolution",
                    "resolved_side_authoritative": True,
                    "resolved_ts_ns": int(resolution["resolved_ts_ns"]),
                    "label_join_quality": label_join_quality or "market_slug_exact",
                }
            )

    return market_rows, token_rows, label_rows


def _iter_manifest_files(root: Path, *, date_from: date, date_to: date) -> list[Path]:
    manifests_root = root / "manifests"
    manifest_paths: list[Path] = []
    for path in manifests_root.rglob("*.manifest.json"):
        target_date = None
        for part in path.parts:
            try:
                target_date = date.fromisoformat(part)
                break
            except ValueError:
                continue
        if target_date is None or target_date < date_from or target_date > date_to:
            continue
        manifest_paths.append(path)
    return sorted(manifest_paths)


def _build_quality_rows(
    *,
    root: Path,
    date_from: date,
    date_to: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in _iter_manifest_files(root, date_from=date_from, date_to=date_to):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        file_path = str(payload.get("file_path") or "")
        relative_path = _relative_raw_path_from_manifest_file_path(file_path)
        source_name = _normalize_source_name(relative_path)
        descriptor = _descriptor_for_source(source_name)
        manifest_quality_class = str(payload.get("quality_class") or "unknown")
        file_state = str(payload.get("file_state") or "unknown")
        first_recv_ts_ns = _to_int(payload.get("first_recv_ts_ns"))
        last_recv_ts_ns = _to_int(payload.get("last_recv_ts_ns"))
        target_day = None
        for part in manifest_path.parts:
            try:
                target_day = date.fromisoformat(part)
                break
            except ValueError:
                continue
        hour_str = Path(file_path).name.split(".")[0].split("__", 1)[0]

        quarantine_flag = False
        quarantine_reason = ""
        if not bool(descriptor["default_include"]):
            quarantine_flag = True
            quarantine_reason = "source_excluded"
        elif descriptor["source_tier"] in {"verification_only", "debug_only"}:
            quarantine_flag = True
            quarantine_reason = f"source_tier:{descriptor['source_tier']}"
        elif manifest_quality_class != "finalized_clean":
            quarantine_flag = True
            quarantine_reason = f"manifest_quality_class:{manifest_quality_class}"
        elif file_state != "finalized":
            quarantine_flag = True
            quarantine_reason = f"file_state:{file_state}"
        elif target_day is not None and (source_name, target_day.isoformat(), hour_str) in SCRIPTURE_EXPLICIT_SOURCE_HOUR_QUARANTINES:
            quarantine_flag = True
            quarantine_reason = "source_specific_integrity_check"

        rows.append(
            {
                "file_path": file_path.replace("\\", "/"),
                "source_name": source_name,
                "source_tier": descriptor["source_tier"],
                "quality_class": _normalize_quality_class(manifest_quality_class),
                "manifest_quality_class": manifest_quality_class,
                "finalized": bool(payload.get("finalized") is True),
                "file_state": file_state,
                "tail_status": str(payload.get("tail_status") or ""),
                "first_recv_ts_ns": int(first_recv_ts_ns) if first_recv_ts_ns is not None else pd.NA,
                "last_recv_ts_ns": int(last_recv_ts_ns) if last_recv_ts_ns is not None else pd.NA,
                "quarantine_flag": bool(quarantine_flag),
                "quarantine_reason": quarantine_reason,
            }
        )
    rows.sort(key=lambda item: (item["file_path"], item["source_name"]))
    return rows


def _build_source_rows() -> list[dict[str, Any]]:
    rows = []
    for source_name in sorted(SOURCE_DESCRIPTORS):
        descriptor = _descriptor_for_source(source_name)
        rows.append(
            {
                "source_name": source_name,
                "role": descriptor["role"],
                "default_include": bool(descriptor["default_include"]),
                "join_clock_policy": "recv_ts_ns",
                "remarks": descriptor["remarks"],
            }
        )
    return rows


def _coerce_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].where(frame[column].notna(), None)
    return frame


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    _coerce_frame(frame.copy()).to_parquet(path, index=False, compression="zstd")


def build_registries(
    *,
    root: Path,
    policy_path: Path,
    date_from: date = DEFAULT_DATE_FROM,
    date_to: date = DEFAULT_DATE_TO,
) -> dict[str, Any]:
    root = Path(root)
    policy_path = Path(policy_path)
    if date_to < date_from:
        raise ValueError(f"date_to must be on or after date_from: {date_from} > {date_to}")
    if not root.exists():
        raise FileNotFoundError(root)
    load_dataset_policy(policy_path)

    earliest_strikes, strike_exclusions = _scan_strike_records(
        root=root,
        policy_path=policy_path,
        date_from=date_from,
        date_to=date_to,
    )
    resolution_by_slug, resolution_by_condition_id, resolution_exclusions = _scan_resolution_records(
        root=root,
        policy_path=policy_path,
        date_from=date_from,
        date_to=date_to,
    )
    market_rows, token_rows, label_rows = _build_market_rows(
        earliest_strikes=earliest_strikes,
        resolution_by_slug=resolution_by_slug,
        resolution_by_condition_id=resolution_by_condition_id,
        date_from=date_from,
        date_to=date_to,
    )
    quality_rows = _build_quality_rows(root=root, date_from=date_from, date_to=date_to)
    source_rows = _build_source_rows()

    canonical_root = root / "canonical"
    market_path = canonical_root / "market_registry_v1.parquet"
    token_path = canonical_root / "token_registry_v1.parquet"
    label_path = canonical_root / "label_registry_v1.parquet"
    quality_path = canonical_root / "quality_registry_v1" / "quality_registry_v1.parquet"
    source_path = canonical_root / "source_registry_v1.parquet"

    market_frame = pd.DataFrame(market_rows).sort_values(["market_open_ts_ns", "market_slug"], kind="stable").reset_index(drop=True)
    token_frame = pd.DataFrame(token_rows).sort_values(["market_slug", "token_side_normalized"], kind="stable").reset_index(drop=True)
    label_frame = pd.DataFrame(label_rows).sort_values(["resolved_ts_ns", "market_slug"], kind="stable").reset_index(drop=True)
    quality_frame = pd.DataFrame(quality_rows).sort_values(["source_name", "file_path"], kind="stable").reset_index(drop=True)
    source_frame = pd.DataFrame(source_rows).sort_values(["source_name"], kind="stable").reset_index(drop=True)

    _write_parquet(market_path, market_frame)
    _write_parquet(token_path, token_frame)
    _write_parquet(label_path, label_frame)
    _write_parquet(quality_path, quality_frame)
    _write_parquet(source_path, source_frame)

    return {
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "market_registry_path": str(market_path),
        "token_registry_path": str(token_path),
        "label_registry_path": str(label_path),
        "quality_registry_path": str(quality_path),
        "source_registry_path": str(source_path),
        "total_markets": int(len(market_frame)),
        "trainable_markets": int(market_frame["is_trainable"].sum()) if not market_frame.empty else 0,
        "markets_missing_resolution": int((~market_frame["has_canonical_resolution"]).sum()) if not market_frame.empty else 0,
        "label_rows": int(len(label_frame)),
        "token_rows": int(len(token_frame)),
        "quality_rows": int(len(quality_frame)),
        "source_rows": int(len(source_frame)),
        "strike_exclusions": strike_exclusions,
        "resolution_exclusions": resolution_exclusions,
    }
