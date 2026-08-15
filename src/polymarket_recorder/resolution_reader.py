from __future__ import annotations

from pathlib import Path

from market_recorders.unified_reader import UnifiedRawReader


class PolymarketResolutionReader(UnifiedRawReader):
    def __init__(self, *, root: Path) -> None:
        super().__init__(root=root)
        self.relative_prefix = Path("polymarket") / "resolution" / "btc_updown_5m"

    def iter_resolutions(
        self,
        *,
        target_date: str,
        market_id: str | None = None,
        market_slug: str | None = None,
        resolved_side: str | None = None,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ):
        for record in self.iter_date(
            relative_prefix=self.relative_prefix,
            target_date=target_date,
            kind="resolution",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            if market_id is not None and str(record.get("market_id")) != market_id:
                continue
            if market_slug is not None and str(record.get("market_slug")) != market_slug:
                continue
            if resolved_side is not None and str(record.get("resolved_side")) != resolved_side:
                continue
            yield record

    def iter_history(
        self,
        *,
        target_date: str,
        market_id: str | None = None,
        market_slug: str | None = None,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ):
        for record in self.iter_date(
            relative_prefix=self.relative_prefix,
            target_date=target_date,
            kind="history",
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        ):
            if market_id is not None and str(record.get("market_id") or record.get("matched_market_id")) != market_id:
                continue
            if market_slug is not None and str(record.get("market_slug") or record.get("matched_market_slug")) != market_slug:
                continue
            yield record
