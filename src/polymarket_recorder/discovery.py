from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from market_recorders.http_utils import request_with_retries

from .config import PolymarketRecorderConfig
from .utils import (
    SelectedMarket,
    candidate_target_markets,
    generate_target_slugs,
    newest_market,
    select_active_markets,
    select_target_market,
)


class GammaDiscoveryClient:
    def __init__(self, client: httpx.AsyncClient, config: PolymarketRecorderConfig) -> None:
        self.client = client
        self.config = config

    async def fetch_markets_for_slugs(
        self,
        target_slugs: list[str],
        *,
        closed: bool,
    ) -> list[dict[str, Any]]:
        if not target_slugs:
            return []
        unique_slugs = list(dict.fromkeys(target_slugs))
        chunk_size = min(self.config.gamma_page_limit, 100)
        items: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index in range(0, len(unique_slugs), chunk_size):
            chunk = unique_slugs[index : index + chunk_size]
            params: list[tuple[str, str]] = [
                ("limit", str(len(chunk))),
                ("closed", "true" if closed else "false"),
            ]
            params.extend(("slug", slug) for slug in chunk)
            outcome = await request_with_retries(
                self.client,
                "GET",
                "/markets",
                max_retries=self.config.http_max_retries,
                retry_backoff_seconds=self.config.http_retry_backoff_seconds,
                params=params,
            )
            response = outcome.response
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                continue
            for item in payload:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id") or item.get("conditionId") or item.get("slug") or "")
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                items.append(item)
        return items

    async def fetch_target_market_pages(self) -> tuple[list[dict[str, Any]], list[str]]:
        now = datetime.now(tz=UTC)
        target_slugs = generate_target_slugs(
            now=now,
            prefix=self.config.market_slug_prefix,
            interval_seconds=self.config.market_interval_seconds,
            lookback_intervals=self.config.market_discovery_lookback_intervals,
            lookahead_intervals=self.config.market_discovery_lookahead_intervals,
        )
        return await self.fetch_markets_for_slugs(target_slugs, closed=False), target_slugs

    async def fetch_market_by_id(self, market_id: str) -> dict[str, Any] | None:
        outcome = await request_with_retries(
            self.client,
            "GET",
            f"/markets/{market_id}",
            max_retries=self.config.http_max_retries,
            retry_backoff_seconds=self.config.http_retry_backoff_seconds,
        )
        response = outcome.response
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None

    async def fetch_active_event_pages(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for page_index in range(self.config.gamma_max_pages):
            outcome = await request_with_retries(
                self.client,
                "GET",
                "/events",
                max_retries=self.config.http_max_retries,
                retry_backoff_seconds=self.config.http_retry_backoff_seconds,
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": self.config.gamma_page_limit,
                    "offset": page_index * self.config.gamma_page_limit,
                },
            )
            response = outcome.response
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break
            items.extend(item for item in payload if isinstance(item, dict))
        return items

    async def discover(self) -> tuple[SelectedMarket | None, list[dict[str, Any]], list[str]]:
        items, target_slugs = await self.fetch_target_market_pages()
        now = datetime.now(tz=UTC)
        selected = select_target_market(
            [item for item in items if isinstance(item, dict)],
            target_slugs=target_slugs,
            now=now,
            stale_grace_seconds=self.config.market_stale_grace_seconds,
        )
        return selected, items, target_slugs

    async def discover_market_set(
        self,
    ) -> tuple[SelectedMarket | None, list[SelectedMarket], list[SelectedMarket], list[dict[str, Any]], list[str]]:
        items, target_slugs = await self.fetch_target_market_pages()
        market_items = [item for item in items if isinstance(item, dict)]
        now = datetime.now(tz=UTC)
        candidates = candidate_target_markets(
            market_items,
            target_slugs=target_slugs,
            now=now,
            stale_grace_seconds=self.config.market_stale_grace_seconds,
        )
        active_markets = select_active_markets(
            market_items,
            target_slugs=target_slugs,
            now=now,
            stale_grace_seconds=self.config.market_stale_grace_seconds,
            interval_seconds=self.config.market_interval_seconds,
            preopen_seconds=self.config.market_active_preopen_seconds,
        )
        selected = newest_market(active_markets)
        if selected is None:
            selected = select_target_market(
                market_items,
                target_slugs=target_slugs,
                now=now,
                stale_grace_seconds=self.config.market_stale_grace_seconds,
            )
        return selected, candidates, active_markets, items, target_slugs
