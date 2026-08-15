from polymarket_recorder.l2 import PolymarketL2StateTracker


def test_l2_tracker_reconstructs_book_from_snapshot_and_price_change() -> None:
    tracker = PolymarketL2StateTracker()
    book_payload = {
        "event_type": "book",
        "asset_id": "111",
        "market": "mkt-1",
        "hash": "0xabc",
        "bids": [{"price": "0.45", "size": "10"}, {"price": "0.44", "size": "12"}],
        "asks": [{"price": "0.55", "size": "8"}, {"price": "0.56", "size": "11"}],
        "tick_size": "0.01",
        "min_order_size": "1",
        "timestamp": 1_776_358_506_123,
    }
    snapshot_records = tracker.process_event(book_payload)
    assert len(snapshot_records) == 1
    assert snapshot_records[0]["record_type"] == "l2_book_state"
    assert snapshot_records[0]["best_bid"] == "0.45"
    assert snapshot_records[0]["best_ask"] == "0.55"
    assert snapshot_records[0]["timestamp_semantics"] == "source_metadata"

    price_change_payload = {
        "event_type": "price_change",
        "market": "mkt-1",
        "price_changes": [
            {"asset_id": "111", "side": "BUY", "price": "0.46", "size": "7", "hash": "0xdef"},
            {"asset_id": "111", "side": "SELL", "price": "0.55", "size": "0", "hash": "0xdef"},
        ],
        "timestamp": 1_776_358_506_456,
    }
    update_records = tracker.process_event(price_change_payload)
    assert len(update_records) == 1
    update = update_records[0]
    assert update["record_type"] == "l2_book_state"
    assert update["snapshot_origin"] == "ws_price_change_reconstructed"
    assert update["best_bid"] == "0.46"
    assert update["best_ask"] == "0.56"
    assert update["level_counts"] == {"bid": 3, "ask": 1}


def test_l2_tracker_emits_trade_and_bbo_records() -> None:
    tracker = PolymarketL2StateTracker()
    trade_payload = {
        "event_type": "last_trade_price",
        "asset_id": "222",
        "market": "mkt-2",
        "price": "0.61",
        "size": "5",
        "side": "BUY",
        "timestamp": 1_776_358_507_000,
    }
    trade_records = tracker.process_event(trade_payload)
    assert trade_records[0]["record_type"] == "trade_execution"
    assert trade_records[0]["price"] == "0.61"
    assert trade_records[0]["timestamp_semantics"] == "source_metadata"

    bbo_payload = {
        "event_type": "best_bid_ask",
        "asset_id": "222",
        "market": "mkt-2",
        "best_bid": "0.60",
        "best_ask": "0.62",
        "timestamp": 1_776_358_507_100,
    }
    bbo_records = tracker.process_event(bbo_payload)
    assert bbo_records[0]["record_type"] == "l2_top_of_book"
    assert bbo_records[0]["spread"] == "0.02"
    assert bbo_records[0]["timestamp_semantics"] == "source_metadata"
