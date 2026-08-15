from datetime import date
from pathlib import Path

from binance_recorder.archive import archive_output_path, archive_url


def test_binance_archive_url_and_output_path() -> None:
    day = date(2026, 4, 16)
    url = archive_url(market="spot", dataset="trades", symbol="BTCUSDT", target_date=day)
    assert url.endswith("/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2026-04-16.zip")
    path = archive_output_path(
        out=Path("data"),
        market="spot",
        dataset="trades",
        symbol="BTCUSDT",
        target_date=day,
    )
    assert str(path).replace("\\", "/").endswith(
        "data/archives/binance/spot/trades/BTCUSDT/2026-04-16/BTCUSDT-trades-2026-04-16.zip"
    )
