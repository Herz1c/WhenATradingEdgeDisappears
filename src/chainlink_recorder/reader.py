from __future__ import annotations

from pathlib import Path

from market_recorders.unified_reader import UnifiedRawReader


class ChainlinkRawReader(UnifiedRawReader):
    def __init__(self, *, root: Path, backend: str = "datastreams_ws", symbol: str = "BTCUSD") -> None:
        super().__init__(root=root)
        self.relative_prefix = Path("chainlink") / backend / symbol

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


class ChainlinkPublicDelayedRawReader(UnifiedRawReader):
    def __init__(self, *, root: Path, symbol: str = "BTCUSD") -> None:
        super().__init__(root=root)
        self.relative_prefix = Path("chainlink_public_delayed") / "public_stream_page" / symbol

    def iter_kind(
        self,
        *,
        target_date: str,
        kind: str = "page",
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
