from coinbase_recorder.continuity import CoinbaseSequenceTracker
from coinbase_recorder.recorder import _normalized, _source_timestamps


def test_coinbase_ticker_normalized() -> None:
    payload = {
        "channel": "ticker",
        "events": [
            {
                "tickers": [
                    {
                        "best_bid": "74100.1",
                        "best_bid_quantity": "0.1",
                        "best_ask": "74100.2",
                        "best_ask_quantity": "0.2",
                        "price": "74100.15",
                    }
                ]
            }
        ],
    }
    result = _normalized(payload)
    assert result["best_bid"] == "74100.1"
    assert result["best_ask"] == "74100.2"
    assert result["event_name"] == "ticker"


def test_coinbase_market_trades_normalized_summary() -> None:
    payload = {
        "channel": "market_trades",
        "events": [
            {
                "trades": [
                    {"product_id": "BTC-USD", "price": "74100.1", "side": "BUY"},
                    {"product_id": "BTC-USD", "price": "74101.2", "side": "SELL"},
                ]
            }
        ],
    }
    result = _normalized(payload)
    assert result["trade_count"] == 2
    assert result["first_price"] == "74100.1"
    assert result["last_side"] == "SELL"


def test_coinbase_level2_normalized_summary() -> None:
    payload = {
        "channel": "level2",
        "events": [
            {
                "product_id": "BTC-USD",
                "type": "update",
                "updates": [
                    {"side": "bid", "price_level": "74100.0", "new_quantity": "0.1"},
                    {"side": "offer", "price_level": "74100.1", "new_quantity": "0.2"},
                ],
            }
        ],
    }
    result = _normalized(payload)
    assert result["type"] == "update"
    assert result["update_count"] == 2


def test_coinbase_l2_data_snapshot_is_normalized() -> None:
    payload = {
        "channel": "l2_data",
        "timestamp": "2026-04-16T17:04:57.060264484Z",
        "events": [
            {
                "product_id": "BTC-USD",
                "type": "snapshot",
                "updates": [
                    {
                        "side": "bid",
                        "price_level": "74100.0",
                        "new_quantity": "0.1",
                        "event_time": "2026-04-16T17:04:56.965744Z",
                    },
                    {
                        "side": "offer",
                        "price_level": "74100.1",
                        "new_quantity": "0.2",
                        "event_time": "2026-04-16T17:04:56.965744Z",
                    },
                ],
            }
        ],
    }
    result = _normalized(payload)
    assert result is not None
    assert result["channel"] == "l2_data"
    assert result["product_id"] == "BTC-USD"
    assert result["type"] == "snapshot"
    assert result["update_count"] == 2
    assert result["event_ts_ns"] > 0
    assert result["updates"][0] == {"side": "bid", "price_level": "74100.0", "quantity": "0.1", "event_time": "2026-04-16T17:04:56.965744Z"}


def test_coinbase_l2_data_update_is_not_silently_dropped() -> None:
    payload = {
        "channel": "l2_data",
        "events": [
            {
                "product_id": "BTC-USD",
                "type": "update",
                "updates": [
                    {
                        "side": "offer",
                        "price_level": "74100.1",
                        "new_quantity": "0.25",
                        "event_time": "2026-04-16T17:04:57.000000Z",
                    }
                ],
            }
        ],
    }
    result = _normalized(payload)
    assert result is not None
    assert result["type"] == "update"
    assert result["first_side"] == "offer"
    assert result["first_price_level"] == "74100.1"
    assert result["first_quantity"] == "0.25"


def test_coinbase_source_timestamps_from_trade() -> None:
    payload = {
        "timestamp": "2026-04-16T17:04:57.060264484Z",
        "events": [{"trades": [{"time": "2026-04-16T17:04:56.965744Z"}]}],
    }
    result = _source_timestamps(payload)
    assert result["message_timestamp_ns"] > 0
    assert result["trades_time_ns"] > 0


def test_coinbase_sequence_gap_triggers_reset_policy() -> None:
    tracker = CoinbaseSequenceTracker()
    first = tracker.observe({"channel": "ticker", "sequence_num": 10}, recv_ts_ns=1_000_000_000)
    second = tracker.observe({"channel": "ticker", "sequence_num": 12}, recv_ts_ns=2_000_000_000)
    assert first.requires_reset is False
    assert second.requires_reset is True
    assert second.state == "sequence_gap"
    assert [event["event"] for event in second.lifecycle_events] == [
        "sequence_gap_detected",
        "snapshot_reset_required",
        "channel_resubscribe_triggered",
    ]


def test_coinbase_out_of_order_message_triggers_reset_policy() -> None:
    tracker = CoinbaseSequenceTracker()
    tracker.observe({"channel": "ticker", "sequence_num": 20}, recv_ts_ns=1_000_000_000)
    observation = tracker.observe({"channel": "ticker", "sequence_num": 19}, recv_ts_ns=2_000_000_000)
    assert observation.requires_reset is True
    assert observation.state == "out_of_order"
    assert [event["event"] for event in observation.lifecycle_events] == [
        "out_of_order_detected",
        "snapshot_reset_required",
        "channel_resubscribe_triggered",
    ]
