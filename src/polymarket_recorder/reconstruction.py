from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .reader import PolymarketRawReader


def _token_outcome_map(record: dict[str, Any]) -> dict[str, str | None]:
    asset_ids = [str(value) for value in record.get("selected_asset_ids", []) if value is not None]
    outcomes = [str(value) for value in record.get("selected_outcomes", []) if value is not None]
    mapping = {
        asset_id: outcome
        for asset_id, outcome in zip(asset_ids, outcomes, strict=False)
    }
    for asset_id, outcome in zip(record.get("matched_asset_ids", []), record.get("matched_outcomes", []), strict=False):
        mapping[str(asset_id)] = None if outcome is None else str(outcome)
    return mapping


def _extract_payload_asset_ids(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    if payload.get("asset_id") is not None:
        values.append(str(payload["asset_id"]))
    for key in ("asset_ids", "assets_ids"):
        if isinstance(payload.get(key), list):
            values.extend(str(item) for item in payload[key] if item is not None)
    for key in ("price_changes", "orders"):
        if isinstance(payload.get(key), list):
            values.extend(
                str(item.get("asset_id"))
                for item in payload[key]
                if isinstance(item, dict) and item.get("asset_id") is not None
            )
    return list(dict.fromkeys(value for value in values if value))


def _filter_payload_for_token(payload: dict[str, Any], token_asset_id: str) -> dict[str, Any]:
    filtered = dict(payload)
    if filtered.get("asset_id") is not None and str(filtered["asset_id"]) != token_asset_id:
        filtered.pop("asset_id", None)
    for key in ("asset_ids", "assets_ids"):
        if isinstance(filtered.get(key), list):
            filtered[key] = [item for item in filtered[key] if str(item) == token_asset_id]
    for key in ("price_changes", "orders"):
        if isinstance(filtered.get(key), list):
            filtered[key] = [
                item
                for item in filtered[key]
                if isinstance(item, dict) and str(item.get("asset_id")) == token_asset_id
            ]
    return filtered


def reconstruct_token_ws_records(record: dict[str, Any]) -> list[dict[str, Any]]:
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return []
    outcome_map = _token_outcome_map(record)
    token_ids = _extract_payload_asset_ids(payload)
    if not token_ids:
        token_ids = [str(value) for value in record.get("matched_asset_ids", []) if value is not None]
    reconstructed: list[dict[str, Any]] = []
    for token_asset_id in token_ids:
        token_payload = _filter_payload_for_token(payload, token_asset_id)
        token_outcome = outcome_map.get(token_asset_id)
        reconstructed.append(
            {
                **record,
                "record_type": "reconstructed_token_ws_event",
                "reconstruction_source": "market_level_ws",
                "token_asset_id": token_asset_id,
                "token_outcome": token_outcome,
                "matched_asset_ids": [token_asset_id],
                "matched_outcomes": [token_outcome] if token_outcome is not None else [],
                "payload": token_payload,
            }
        )
    return reconstructed


def reconstruct_token_rest_records(
    record: dict[str, Any],
    *,
    include_shared_context: bool = False,
) -> list[dict[str, Any]]:
    outcome_map = _token_outcome_map(record)
    token_asset_id = record.get("token_asset_id")
    if token_asset_id is None:
        extra = record.get("extra")
        if isinstance(extra, dict):
            token_asset_id = extra.get("asset_id")
    token_ids: list[str] = []
    if token_asset_id is not None:
        token_ids = [str(token_asset_id)]
    elif include_shared_context:
        extra = record.get("extra")
        if isinstance(extra, dict) and isinstance(extra.get("asset_ids"), list):
            token_ids = [str(value) for value in extra["asset_ids"] if value is not None]
    reconstructed: list[dict[str, Any]] = []
    for asset_id in token_ids:
        token_outcome = outcome_map.get(asset_id)
        reconstructed.append(
            {
                **record,
                "record_type": "reconstructed_token_rest_observation",
                "reconstruction_source": "market_level_rest",
                "token_asset_id": asset_id,
                "token_outcome": token_outcome,
                "shared_context": token_asset_id is None,
            }
        )
    return reconstructed


class PolymarketTrainingViewReader(PolymarketRawReader):
    def __init__(self, *, root: Path) -> None:
        super().__init__(root=root)

    def iter_token_l2(
        self,
        *,
        target_date: str,
        token_outcome: str | None = None,
        token_asset_id: str | None = None,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict[str, Any]]:
        for record in self.iter_kind(
            target_date=target_date,
            kind="l2",
            token_outcome=token_outcome,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            if token_asset_id is not None and str(record.get("token_asset_id")) != token_asset_id:
                continue
            yield record

    def iter_token_ws(
        self,
        *,
        target_date: str,
        token_outcome: str | None = None,
        token_asset_id: str | None = None,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict[str, Any]]:
        for record in self.iter_kind(
            target_date=target_date,
            kind="ws",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            for reconstructed in reconstruct_token_ws_records(record):
                if token_outcome is not None and str(reconstructed.get("token_outcome")) != token_outcome:
                    continue
                if token_asset_id is not None and str(reconstructed.get("token_asset_id")) != token_asset_id:
                    continue
                yield reconstructed

    def iter_token_rest(
        self,
        *,
        target_date: str,
        token_outcome: str | None = None,
        token_asset_id: str | None = None,
        include_shared_context: bool = False,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ) -> Iterator[dict[str, Any]]:
        for record in self.iter_kind(
            target_date=target_date,
            kind="rest",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            for reconstructed in reconstruct_token_rest_records(
                record,
                include_shared_context=include_shared_context,
            ):
                if token_outcome is not None and str(reconstructed.get("token_outcome")) != token_outcome:
                    continue
                if token_asset_id is not None and str(reconstructed.get("token_asset_id")) != token_asset_id:
                    continue
                yield reconstructed
