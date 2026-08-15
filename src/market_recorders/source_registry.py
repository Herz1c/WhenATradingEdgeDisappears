from __future__ import annotations

from pathlib import Path

PRIMARY_TRAINING = "primary_training"
SECONDARY_CONTEXT = "secondary_context"
VERIFICATION_ONLY = "verification_only"
DEBUG_ONLY = "debug_only"


def infer_kind_from_path(path: Path) -> str | None:
    if path.name.endswith(".jsonl.zst"):
        return path.name.removesuffix(".jsonl.zst").rsplit(".", 1)[-1]
    if path.name.endswith(".manifest.json"):
        return path.name.removesuffix(".manifest.json").rsplit(".", 1)[-1]
    return None


def source_tier_for_relative_path(relative: Path, *, kind: str | None = None) -> str:
    parts = relative.parts
    if not parts:
        return DEBUG_ONLY
    resolved_kind = kind or infer_kind_from_path(relative)
    root = parts[0]
    if root == "binance":
        market = parts[1] if len(parts) > 1 else ""
        if resolved_kind == "lifecycle":
            return DEBUG_ONLY
        if market == "spot":
            return PRIMARY_TRAINING if resolved_kind in {"ws", "snapshot"} else SECONDARY_CONTEXT
        if market == "usdm":
            return PRIMARY_TRAINING if resolved_kind == "ws" else SECONDARY_CONTEXT
    if root == "polymarket":
        if len(parts) > 1 and parts[1] == "rtds":
            return DEBUG_ONLY if resolved_kind == "lifecycle" else SECONDARY_CONTEXT
        if resolved_kind in {"lifecycle", "discovery"}:
            return DEBUG_ONLY
        if resolved_kind in {"ws", "l2", "rest", "strike", "resolution"}:
            return PRIMARY_TRAINING
        if resolved_kind in {"history", "state"}:
            return SECONDARY_CONTEXT
        return SECONDARY_CONTEXT
    if root == "bybit":
        return DEBUG_ONLY
    if root == "coinbase":
        return DEBUG_ONLY
    if root == "hyperliquid":
        if resolved_kind == "lifecycle":
            return DEBUG_ONLY
        return PRIMARY_TRAINING if resolved_kind == "ws" else SECONDARY_CONTEXT
    if root == "deribit":
        return DEBUG_ONLY
    if root == "chainlink_public_delayed":
        return DEBUG_ONLY if resolved_kind == "lifecycle" else VERIFICATION_ONLY
    if root == "chainlink":
        backend = parts[1] if len(parts) > 1 else ""
        if resolved_kind == "lifecycle":
            return DEBUG_ONLY
        if backend == "onchain":
            return VERIFICATION_ONLY
        if backend in {"public_page", "public_stream_page"}:
            return VERIFICATION_ONLY
        return SECONDARY_CONTEXT
    if root == "fng":
        return DEBUG_ONLY
    return DEBUG_ONLY


def source_registry_metadata(relative: Path, *, kind: str | None = None) -> dict[str, str]:
    resolved_kind = kind or infer_kind_from_path(relative) or "unknown"
    return {
        "archive_kind": resolved_kind,
        "source_tier": source_tier_for_relative_path(relative, kind=resolved_kind),
    }
