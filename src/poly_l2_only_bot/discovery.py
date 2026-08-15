"""Lightweight market discovery for BTC up/down 5-minute markets.

Pattern: the slug is always `btc-updown-5m-<unix_close_ts>` (close ts is
5-min boundary). We generate the next N slug candidates on a fixed
schedule and fetch each via the Gamma `/markets?slug=...` endpoint to
get the asset_ids needed for the WS subscription.

Polled every REFRESH_S seconds. New markets are pushed to the bot via
callback as soon as they're discovered.

NOTE: Each market closes at its `closeTime` field (unix seconds at the
5-min boundary). Markets open ~5 minutes BEFORE close. We want to be
subscribed and building book state from open.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional

import httpx
import orjson


GAMMA_BASE = "https://gamma-api.polymarket.com"
SLUG_PREFIX = "btc-updown-5m-"

# How often to poll for new markets.
REFRESH_S = 30.0

# How far in the future to pre-fetch markets (covers a few cycles + bot startup).
LOOKAHEAD_SECONDS = 30 * 60   # 30 minutes


@dataclass(slots=True)
class DiscoveredMarket:
    market_slug: str
    market_id: str
    condition_id: str
    up_asset_id: str
    down_asset_id: str
    market_open_s: int     # unix seconds
    market_close_s: int    # unix seconds


def _next_5min_boundary(now_s: int) -> int:
    """Round up to the next 5-min boundary."""
    return ((now_s + 299) // 300) * 300


def _generate_target_close_ts(now_s: int, lookahead_s: int) -> List[int]:
    """Generate all 5-min close boundaries within the next lookahead window."""
    first = _next_5min_boundary(now_s)
    return list(range(first, first + lookahead_s + 1, 300))


class BTCUpDownDiscovery:
    """Polls Gamma every REFRESH_S seconds, calls on_new_market(DiscoveredMarket)
    for each previously-unseen slug."""

    def __init__(
        self,
        *,
        on_new_market: Callable[[DiscoveredMarket], Awaitable[None] | None],
        logger: logging.Logger | None = None,
        refresh_s: float = REFRESH_S,
        lookahead_s: int = LOOKAHEAD_SECONDS,
    ) -> None:
        self._on_new_market = on_new_market
        self.logger = logger or logging.getLogger("poly_l2_only_bot.discovery")
        self._refresh_s = refresh_s
        self._lookahead_s = lookahead_s
        self._seen_slugs: set[str] = set()
        self._shutdown = asyncio.Event()

    def stop(self) -> None:
        self._shutdown.set()

    async def run(self) -> None:
        """Outer loop: refresh every refresh_s seconds until shutdown."""
        async with httpx.AsyncClient(
            base_url=GAMMA_BASE,
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=10.0),
            headers={"Accept": "application/json"},
        ) as client:
            while not self._shutdown.is_set():
                try:
                    await self._refresh_once(client)
                except Exception as exc:
                    self.logger.warning("discovery refresh failed: %r", exc)
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=self._refresh_s)
                except asyncio.TimeoutError:
                    pass

    async def _refresh_once(self, client: httpx.AsyncClient) -> None:
        now_s = int(time.time())
        target_close_ts = _generate_target_close_ts(now_s, self._lookahead_s)
        target_slugs = [f"{SLUG_PREFIX}{t}" for t in target_close_ts]
        # Filter out slugs we've already seen.
        new_slugs = [s for s in target_slugs if s not in self._seen_slugs]
        if not new_slugs:
            return
        # Gamma's /markets endpoint accepts repeated ?slug= params.
        # Chunk into batches of 50 to keep URLs reasonable.
        for i in range(0, len(new_slugs), 50):
            batch = new_slugs[i:i+50]
            params = [("limit", str(len(batch))), ("closed", "false")]
            params.extend(("slug", s) for s in batch)
            r = await client.get("/markets", params=params)
            if r.status_code != 200:
                continue
            try:
                items = r.json()
            except Exception:
                continue
            if not isinstance(items, list):
                continue
            for item in items:
                dm = self._parse(item)
                if dm is None:
                    continue
                if dm.market_slug in self._seen_slugs:
                    continue
                self._seen_slugs.add(dm.market_slug)
                self.logger.info(
                    "discovered market %s (close=%s, ids=%s/%s)",
                    dm.market_slug,
                    datetime.fromtimestamp(dm.market_close_s, tz=timezone.utc)
                            .strftime("%H:%M:%S"),
                    (dm.up_asset_id or "")[:12],
                    (dm.down_asset_id or "")[:12],
                )
                ret = self._on_new_market(dm)
                if asyncio.iscoroutine(ret):
                    try:
                        await ret
                    except Exception as exc:
                        self.logger.warning("on_new_market raised: %r", exc)

    @staticmethod
    def _parse(item: Dict) -> Optional[DiscoveredMarket]:
        try:
            slug = str(item.get("slug") or "")
            if not slug.startswith(SLUG_PREFIX):
                return None
            market_id = str(item.get("id") or "")
            condition_id = str(item.get("conditionId") or "")
            # clobTokenIds is a JSON string array; outcomes is also a JSON
            # string array of ["Up", "Down"] (order matches).
            tokens_raw = item.get("clobTokenIds")
            outcomes_raw = item.get("outcomes")
            tokens = orjson.loads(tokens_raw) if isinstance(tokens_raw, str) else (tokens_raw or [])
            outcomes = orjson.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
            if not (isinstance(tokens, list) and isinstance(outcomes, list)):
                return None
            if len(tokens) != 2 or len(outcomes) != 2:
                return None
            up_asset_id = down_asset_id = ""
            for tk, oc in zip(tokens, outcomes):
                oc_s = str(oc).strip().lower()
                if oc_s == "up":
                    up_asset_id = str(tk)
                elif oc_s == "down":
                    down_asset_id = str(tk)
            if not up_asset_id or not down_asset_id:
                return None
            # The slug suffix is the market OPEN/start time; the market RESOLVES
            # 300s later. Verified against Gamma endDate (slug 1781302200 =
            # 22:10 open, endDate 22:15 close) and matches the dataset builder
            # (build_poly_l2_last60s.py), which also treats the slug unix as open
            # and close = open + 300. Treating the slug as close made the live
            # bot trade ~5 min too early (at the market open instead of the last
            # 10-50s before resolution).
            try:
                market_open_s = int(slug.removeprefix(SLUG_PREFIX))
            except ValueError:
                return None
            market_close_s = market_open_s + 300
            return DiscoveredMarket(
                market_slug=slug,
                market_id=market_id,
                condition_id=condition_id,
                up_asset_id=up_asset_id,
                down_asset_id=down_asset_id,
                market_open_s=market_open_s,
                market_close_s=market_close_s,
            )
        except Exception:
            return None
