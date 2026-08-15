from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import orjson

from binance_recorder.utils import ensure_directory, isoformat_utc


def archive_url(*, market: str, dataset: str, symbol: str, target_date: date) -> str:
    date_text = target_date.isoformat()
    if market == "spot":
        prefix = "spot"
    elif market == "usdm":
        prefix = "futures/um"
    else:
        raise ValueError(f"Unsupported market: {market}")
    filename = f"{symbol}-{dataset}-{date_text}.zip"
    return f"https://data.binance.vision/data/{prefix}/daily/{dataset}/{symbol}/{filename}"


def archive_output_path(*, out: Path, market: str, dataset: str, symbol: str, target_date: date) -> Path:
    folder = ensure_directory(out / "archives" / "binance" / market / dataset / symbol / target_date.isoformat())
    return folder / f"{symbol}-{dataset}-{target_date.isoformat()}.zip"


def download_archive_day(
    *,
    out: Path,
    market: str,
    dataset: str,
    symbol: str,
    target_date: date,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    url = archive_url(market=market, dataset=dataset, symbol=symbol, target_date=target_date)
    output_path = archive_output_path(
        out=out,
        market=market,
        dataset=dataset,
        symbol=symbol,
        target_date=target_date,
    )
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        output_path.write_bytes(response.content)
    metadata = {
        "downloaded_at": isoformat_utc(),
        "url": url,
        "market": market,
        "dataset": dataset,
        "symbol": symbol,
        "date": target_date.isoformat(),
        "file_path": str(output_path).replace("\\", "/"),
        "file_size": output_path.stat().st_size,
        "http_status": 200,
    }
    output_path.with_suffix(".download.json").write_bytes(
        orjson.dumps(metadata, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )
    return metadata
