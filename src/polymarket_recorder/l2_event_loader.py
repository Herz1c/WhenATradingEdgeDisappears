from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import re
from pathlib import Path
from typing import Any

import pandas as pd

from market_recorders.dataset_policy import DEFAULT_DATASET_POLICY_PATH, load_dataset_policy, training_metadata_eligibility
from market_recorders.unified_reader import UnifiedRawReader

SECOND_NS = 1_000_000_000
DEFAULT_QUALITY_REGISTRY_PATH = Path("data/canonical/quality_registry_v1/quality_registry_v1.parquet")
DEFAULT_CANONICAL_L2_EVENTS_PATH = Path("canonical/polymarket_l2_events_v1")
DEFAULT_CANONICAL_L2_MARKET_EVENTS_PATH = Path("canonical/polymarket_l2_events_v1_by_market")


@dataclass(slots=True, frozen=True)
class L2EventLoadResult:
    events: pd.DataFrame
    exclusion_summary: dict[str, Counter]


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


def _file_hour(path: Path) -> str:
    return path.name.split(".", 1)[0].split("__", 1)[0][:2]


def _normalize_path(value: str | Path) -> str:
    return Path(value).as_posix()


def _safe_market_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned[:160] or "market"


@lru_cache(maxsize=8)
def _load_quality_registry_allowed_l2_paths_cached(root_value: str, quality_registry_value: str) -> frozenset[str] | None:
    """Return canonical finalized-clean, non-quarantined L2 raw file paths.

    If the registry is missing, callers can fall back to manifest/policy checks. Returning
    None makes that fallback explicit.
    """

    root = Path(root_value)
    registry_path = Path(quality_registry_value)
    if not registry_path.exists():
        return None
    frame = pd.read_parquet(registry_path)
    required = {"file_path", "source_name", "manifest_quality_class", "finalized", "file_state", "quarantine_flag"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"quality_registry_v1 missing required columns for Phase 8: {sorted(missing)}")
    mask = (
        frame["source_name"].astype(str).eq("polymarket_l2")
        & frame["manifest_quality_class"].astype(str).eq("finalized_clean")
        & frame["finalized"].astype(bool)
        & frame["file_state"].astype(str).eq("finalized")
        & ~frame["quarantine_flag"].astype(bool)
    )
    values: set[str] = set()
    for raw_path in frame.loc[mask, "file_path"].astype(str):
        normalized = _normalize_path(raw_path)
        values.add(normalized)
        try:
            values.add(_normalize_path(Path(normalized).relative_to(root)))
        except Exception:
            pass
        if normalized.startswith("data/"):
            values.add(normalized.removeprefix("data/"))
    return frozenset(values)


def load_quality_registry_allowed_l2_paths(
    *,
    root: Path,
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
) -> set[str] | None:
    values = _load_quality_registry_allowed_l2_paths_cached(str(root), str(quality_registry_path))
    return set(values) if values is not None else None


def normalize_event_type(record: dict[str, Any]) -> str:
    record_type = str(record.get("record_type") or "")
    event_type = str(record.get("event_type") or "")
    if record_type == "l2_book_state":
        return "book_state"
    if record_type == "l2_top_of_book" or event_type == "best_bid_ask":
        return "top_of_book"
    if record_type == "trade_execution" or event_type == "last_trade_price":
        return "trade"
    if record_type == "tick_size_update" or event_type == "tick_size_change":
        return "tick_update"
    return "other"


def normalize_trade_side(value: Any, *, is_trade: bool) -> str:
    if not is_trade:
        return "none"
    side = str(value or "").strip().lower()
    if side in {"buy", "bid"}:
        return "buy"
    if side in {"sell", "ask"}:
        return "sell"
    return "unknown"


def _token_side(record: dict[str, Any]) -> str | None:
    outcome = str(record.get("token_outcome") or "").strip().upper()
    if outcome in {"UP", "DOWN"}:
        return outcome
    asset_id = str(record.get("token_asset_id") or "")
    asset_ids = [str(value) for value in record.get("selected_asset_ids", []) if value is not None]
    outcomes = [str(value).strip().upper() for value in record.get("selected_outcomes", []) if value is not None]
    for candidate_asset_id, candidate_outcome in zip(asset_ids, outcomes, strict=False):
        if candidate_asset_id == asset_id and candidate_outcome in {"UP", "DOWN"}:
            return candidate_outcome
    return None


def normalize_l2_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("token_modeling_safe") is not True:
        return None
    market_slug = str(record.get("selected_market_slug") or "").strip()
    token_asset_id = str(record.get("token_asset_id") or "").strip()
    token_side = _token_side(record)
    recv_ts_ns = _to_int(record.get("recv_ts_ns"))
    if not market_slug or not token_asset_id or token_side not in {"UP", "DOWN"} or recv_ts_ns is None:
        return None

    event_type = normalize_event_type(record)
    is_trade = event_type == "trade"
    best_bid = _to_float(record.get("best_bid"))
    best_ask = _to_float(record.get("best_ask"))
    spread = _to_float(record.get("spread"))
    mid = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None

    depth_totals = record.get("depth_totals") if isinstance(record.get("depth_totals"), dict) else {}
    bid_depth_total = _to_float(depth_totals.get("bid_size_total"))
    ask_depth_total = _to_float(depth_totals.get("ask_size_total"))
    total_depth = (bid_depth_total or 0.0) + (ask_depth_total or 0.0)
    if best_bid is not None and best_ask is not None:
        if total_depth > 0:
            microprice = ((best_bid * (ask_depth_total or 0.0)) + (best_ask * (bid_depth_total or 0.0))) / total_depth
        else:
            microprice = (best_bid + best_ask) / 2.0
    else:
        microprice = None
    imbalance = ((bid_depth_total or 0.0) - (ask_depth_total or 0.0)) / total_depth if total_depth > 0 else 0.0

    trade_price = _to_float(record.get("price")) if is_trade else None
    trade_size = _to_float(record.get("size")) if is_trade else 0.0
    return {
        "market_slug": market_slug,
        "token_asset_id": token_asset_id,
        "token_side_normalized": token_side,
        "recv_ts_ns": int(recv_ts_ns),
        "book_state_seq": _to_int(record.get("book_state_seq")),
        "event_type": event_type,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread if spread is not None else (best_ask - best_bid if best_bid is not None and best_ask is not None else None),
        "microprice": microprice,
        "bid_depth_total": bid_depth_total,
        "ask_depth_total": ask_depth_total,
        "imbalance": imbalance,
        "last_trade_price": trade_price if is_trade else _to_float(record.get("last_trade_price")),
        "trade_size": trade_size or 0.0,
        "trade_side": normalize_trade_side(record.get("side"), is_trade=is_trade),
        "tick_size": _to_float(record.get("tick_size") or record.get("new_tick_size")),
    }


def iter_l2_events(
    *,
    root: Path,
    target_date: date,
    policy_path: Path = DEFAULT_DATASET_POLICY_PATH,
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
    market_slug: str | None = None,
    market_slugs: set[str] | None = None,
    exclusion_summary: dict[str, Counter] | None = None,
) -> Iterator[dict[str, Any]]:
    policy = load_dataset_policy(policy_path)
    reader = UnifiedRawReader(root=root)
    allowed_paths = load_quality_registry_allowed_l2_paths(root=root, quality_registry_path=quality_registry_path)
    exclusions = exclusion_summary if exclusion_summary is not None else defaultdict(Counter)
    relative_prefix = Path("polymarket") / "btc_updown_5m"
    wanted_markets = set(market_slugs or set())
    if market_slug:
        wanted_markets.add(market_slug)

    for path in reader.list_files_for_date(
        relative_prefix=relative_prefix,
        target_date=target_date,
        kind="l2",
        finalized_only=False,
    ):
        normalized = _normalize_path(path)
        relative_raw = _normalize_path(path.relative_to(root / "raw"))
        data_prefixed = f"data/raw/{relative_raw}"
        if wanted_markets and not any(slug in path.name for slug in wanted_markets):
            continue
        if allowed_paths is not None and normalized not in allowed_paths and data_prefixed not in allowed_paths and relative_raw not in allowed_paths:
            exclusions["polymarket_l2"]["quality_registry_excluded"] += 1
            continue

        metadata = reader.file_metadata(path)
        eligible, reason = training_metadata_eligibility(
            policy=policy,
            metadata=metadata,
            source_name="polymarket",
            date_str=target_date.isoformat(),
            hour_str=_file_hour(path),
        )
        if not eligible:
            exclusions["polymarket_l2"][reason or "unknown"] += 1
            continue
        for source_file_index, record in enumerate(reader.iter_file(path, finalized_only=True, tail_tolerant=False)):
            normalized_record = normalize_l2_record(record)
            if normalized_record is None:
                continue
            if wanted_markets and normalized_record["market_slug"] not in wanted_markets:
                continue
            normalized_record["source_file_path"] = relative_raw
            normalized_record["source_file_index"] = int(source_file_index)
            yield normalized_record


def load_l2_events(
    *,
    root: Path,
    target_date: date,
    policy_path: Path = DEFAULT_DATASET_POLICY_PATH,
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
    market_slug: str | None = None,
    market_slugs: set[str] | None = None,
) -> L2EventLoadResult:
    exclusion_summary: dict[str, Counter] = defaultdict(Counter)
    rows = list(
        iter_l2_events(
            root=root,
            target_date=target_date,
            policy_path=policy_path,
            quality_registry_path=quality_registry_path,
            market_slug=market_slug,
            market_slugs=market_slugs,
            exclusion_summary=exclusion_summary,
        )
    )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_columns = [
            column
            for column in [
                "market_slug",
                "token_asset_id",
                "recv_ts_ns",
                "source_file_path",
                "source_file_index",
            ]
            if column in frame.columns
        ]
        frame = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return L2EventLoadResult(events=frame, exclusion_summary={key: Counter(value) for key, value in exclusion_summary.items()})


def _canonical_l2_path(root: Path, target_date: date, output_path: Path = DEFAULT_CANONICAL_L2_EVENTS_PATH) -> Path:
    resolved = output_path if output_path.is_absolute() else root / output_path
    return resolved / f"{target_date.isoformat()}.parquet"


def _canonical_l2_market_dir(
    root: Path,
    target_date: date,
    output_path: Path = DEFAULT_CANONICAL_L2_MARKET_EVENTS_PATH,
) -> Path:
    resolved = output_path if output_path.is_absolute() else root / output_path
    return resolved / target_date.isoformat()


def _canonical_l2_market_path(root: Path, target_date: date, market_slug: str) -> Path:
    return _canonical_l2_market_dir(root, target_date) / f"{_safe_market_slug(market_slug)}.parquet"


def write_canonical_l2_events_for_date(
    *,
    root: Path,
    target_date: date,
    output_path: Path = DEFAULT_CANONICAL_L2_EVENTS_PATH,
    policy_path: Path = DEFAULT_DATASET_POLICY_PATH,
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
    overwrite: bool = False,
) -> Path:
    """Materialize policy-filtered Polymarket L2 events as the Phase 2 canonical table shard."""

    path = _canonical_l2_path(root, target_date, output_path)
    if path.exists() and not overwrite:
        return path
    result = load_l2_events(
        root=root,
        target_date=target_date,
        policy_path=policy_path,
        quality_registry_path=quality_registry_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    result.events.to_parquet(path, index=False)
    return path


def write_canonical_l2_market_shards_for_date(
    *,
    root: Path,
    target_date: date,
    output_path: Path = DEFAULT_CANONICAL_L2_MARKET_EVENTS_PATH,
    daily_output_path: Path = DEFAULT_CANONICAL_L2_EVENTS_PATH,
    policy_path: Path = DEFAULT_DATASET_POLICY_PATH,
    quality_registry_path: Path = DEFAULT_QUALITY_REGISTRY_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize deterministic market-level canonical L2 shards for replay-heavy builders."""

    daily_path = write_canonical_l2_events_for_date(
        root=root,
        target_date=target_date,
        output_path=daily_output_path,
        policy_path=policy_path,
        quality_registry_path=quality_registry_path,
        overwrite=False,
    )
    market_dir = _canonical_l2_market_dir(root, target_date, output_path)
    if market_dir.exists() and not overwrite and any(market_dir.glob("*.parquet")):
        paths = sorted(market_dir.glob("*.parquet"))
        return {
            "date": target_date.isoformat(),
            "daily_path": str(daily_path),
            "market_dir": str(market_dir),
            "market_shards": len(paths),
            "rows": sum(int(pd.read_parquet(path, columns=["recv_ts_ns"]).shape[0]) for path in paths),
            "preexisting": True,
        }
    if market_dir.exists() and overwrite:
        for old_path in market_dir.glob("*.parquet"):
            old_path.unlink()
    market_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(daily_path)
    if frame.empty:
        return {
            "date": target_date.isoformat(),
            "daily_path": str(daily_path),
            "market_dir": str(market_dir),
            "market_shards": 0,
            "rows": 0,
            "preexisting": False,
        }
    frame = frame.sort_values(
        [
            column
            for column in [
                "market_slug",
                "token_asset_id",
                "recv_ts_ns",
                "source_file_path",
                "source_file_index",
            ]
            if column in frame.columns
        ],
        kind="stable",
    ).reset_index(drop=True)
    rows = 0
    shard_count = 0
    for market_slug, group in frame.groupby("market_slug", sort=False):
        path = _canonical_l2_market_path(root, target_date, str(market_slug))
        group.to_parquet(path, index=False)
        rows += int(len(group))
        shard_count += 1
    return {
        "date": target_date.isoformat(),
        "daily_path": str(daily_path),
        "market_dir": str(market_dir),
        "market_shards": int(shard_count),
        "rows": int(rows),
        "preexisting": False,
    }


def load_canonical_l2_events_for_date(
    *,
    root: Path,
    target_date: date,
    canonical_path: Path = DEFAULT_CANONICAL_L2_EVENTS_PATH,
    market_slug: str | None = None,
    market_slugs: set[str] | None = None,
) -> pd.DataFrame | None:
    path = _canonical_l2_path(root, target_date, canonical_path)
    if not path.exists():
        return None
    wanted_markets = set(market_slugs or set())
    if market_slug:
        wanted_markets.add(market_slug)
    if wanted_markets:
        market_dir = _canonical_l2_market_dir(root, target_date)
        market_paths = [_canonical_l2_market_path(root, target_date, str(market)) for market in sorted(wanted_markets)]
        if market_dir.exists() and all(path.exists() for path in market_paths):
            frames = [pd.read_parquet(path) for path in market_paths]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, axis=0, ignore_index=True).reset_index(drop=True)
    filters = None
    if wanted_markets:
        filters = [("market_slug", "in", sorted(str(value) for value in wanted_markets))]
    try:
        frame = pd.read_parquet(path, filters=filters)
    except Exception:
        frame = pd.read_parquet(path)
    if wanted_markets and not frame.empty:
        frame = frame.loc[frame["market_slug"].astype(str).isin(wanted_markets)].copy()
    return frame.reset_index(drop=True)
