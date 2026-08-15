from __future__ import annotations

from pathlib import Path

from market_recorders.unified_reader import UnifiedRawReader


class PolymarketRawReader(UnifiedRawReader):
    def __init__(self, *, root: Path) -> None:
        super().__init__(root=root)
        self.relative_prefix = Path("polymarket") / "btc_updown_5m"

    def iter_kind(
        self,
        *,
        target_date: str,
        kind: str,
        token_outcome: str | None = None,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ):
        yield from self.iter_date(
            relative_prefix=self.relative_prefix,
            target_date=target_date,
            kind=kind,
            token_outcome=token_outcome,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        )
