from __future__ import annotations

from pathlib import Path

from market_recorders.unified_reader import UnifiedRawReader


class RawArchiveReader(UnifiedRawReader):
    def __init__(self, *, root: Path, market: str = "spot", symbol: str = "BTCUSDT") -> None:
        super().__init__(root=root)
        self.relative_prefix = Path("binance") / market / symbol

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
