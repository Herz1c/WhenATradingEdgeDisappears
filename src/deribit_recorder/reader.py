from __future__ import annotations

from pathlib import Path

from market_recorders.unified_reader import UnifiedRawReader


class DeribitRawReader(UnifiedRawReader):
    def __init__(self, *, root: Path, currency: str = "BTC") -> None:
        super().__init__(root=root)
        self.relative_prefix = Path("deribit") / "options" / currency

    def iter_kind(self, *, target_date: str, kind: str):
        yield from self.iter_date(relative_prefix=self.relative_prefix, target_date=target_date, kind=kind)
