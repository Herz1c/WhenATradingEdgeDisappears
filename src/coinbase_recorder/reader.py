from __future__ import annotations

from pathlib import Path

from market_recorders.unified_reader import UnifiedRawReader


class CoinbaseAdvancedRawReader(UnifiedRawReader):
    def __init__(self, *, root: Path, product_id: str = "BTC-USD") -> None:
        super().__init__(root=root)
        self.relative_prefix = Path("coinbase") / "advanced" / product_id

    def iter_kind(
        self,
        *,
        target_date: str,
        kind: str,
        finalized_only: bool = True,
        tail_tolerant: bool = False,
    ):
        yield from self.iter_date(
            relative_prefix=self.relative_prefix,
            target_date=target_date,
            kind=kind,
            finalized_only=finalized_only,
            tail_tolerant=tail_tolerant,
        )
