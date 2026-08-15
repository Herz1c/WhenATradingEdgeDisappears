from __future__ import annotations

from binance_recorder.compression import ZstdJsonlAppender
from chainlink_recorder.alignment import (
    AlignmentPoint,
    build_chainlink_alignment_report,
    compute_alignment_metrics,
)
from binance_recorder.utils import ns_to_iso


def test_compute_chainlink_alignment_metrics_detects_stable_offset() -> None:
    rtds_points = [
        AlignmentPoint(recv_ts_ns=(index + 1) * 1_000_000_000, price=74_100 + index, source_ts_ns=None)
        for index in range(12)
    ]
    public_points = [
        AlignmentPoint(
            recv_ts_ns=((index + 61) * 1_000_000_000),
            price=74_100 + index,
            source_ts_ns=None,
        )
        for index in range(12)
    ]
    summary = compute_alignment_metrics(rtds_points, public_points)
    assert summary["status"] == "aligned"
    assert summary["best_offset_ms"] == 60_000
    assert summary["median_lag_ms"] == 60_000.0
    assert summary["near_match_rate"] == 1.0
    assert summary["lag_relationship"] == "stable"


def test_build_chainlink_alignment_report_reads_rtds_and_public_delayed_files(tmp_path) -> None:
    rtds_path = (
        tmp_path
        / "raw"
        / "polymarket"
        / "rtds"
        / "crypto_prices_chainlink"
        / "btc_usd"
        / "2026-04-18"
        / "17.ws.jsonl.zst"
    )
    public_path = (
        tmp_path
        / "raw"
        / "chainlink_public_delayed"
        / "public_stream_page"
        / "BTCUSD"
        / "2026-04-18"
        / "17.page.jsonl.zst"
    )
    rtds_appender = ZstdJsonlAppender(rtds_path)
    public_appender = ZstdJsonlAppender(public_path)
    for index in range(8):
        rtds_recv_ts_ns = (1_776_538_000 + index) * 1_000_000_000
        public_recv_ts_ns = (1_776_538_060 + index) * 1_000_000_000
        price = f"{74100 + index}.00"
        rtds_appender.write_json(
            {
                "record_type": "chainlink_live_reference",
                "recv_ts_ns": rtds_recv_ts_ns,
                "recv_ts_iso": ns_to_iso(rtds_recv_ts_ns),
                "recv_ts_source": "time.time_ns",
                "chainlink_ts_ms": 1_776_538_000_000 + (index * 1_000),
                "chainlink_ts_iso": ns_to_iso(1_776_538_000_000_000_000 + (index * 1_000_000_000)),
                "btc_usd_price": price,
                "source": "polymarket_rtds_chainlink",
            }
        )
        public_appender.write_json(
            {
                "record_type": "chainlink_public_delayed_observation",
                "recv_ts_ns": public_recv_ts_ns,
                "recv_ts_iso": ns_to_iso(public_recv_ts_ns),
                "recv_ts_source": "time.time_ns",
                "source": "chainlink_public_delayed",
                "backend": "public_stream_page",
                "symbol": "BTCUSD",
                "page_url": "https://data.chain.link/streams/btc-usd-cexprice-streams",
                "chainlink_display_ts": 1_776_538_000 + index,
                "chainlink_display_ts_iso": ns_to_iso(1_776_538_000_000_000_000 + (index * 1_000_000_000)),
                "btc_usd_price": price,
                "capture_timing": {},
                "parse_status": "success",
                "parse_error": None,
            }
        )
    rtds_appender.close()
    public_appender.close()

    report = build_chainlink_alignment_report(
        root=tmp_path,
        date_from="2026-04-18",
        date_to="2026-04-18",
        finalized_only=False,
        by_hour=True,
    )
    assert report["rtds_point_count"] == 8
    assert report["public_delayed_point_count"] == 8
    assert report["summary"]["status"] == "aligned"
    assert report["summary"]["best_offset_ms"] == 60_000
    assert report["hourly"]
