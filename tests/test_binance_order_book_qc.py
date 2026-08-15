from binance_recorder.order_book_qc import BinanceOrderBookIntegrityMonitor


def test_binance_order_book_gap_requires_resnapshot_and_recovers() -> None:
    monitor = BinanceOrderBookIntegrityMonitor(symbol="BTCUSDT", market="spot")
    buffering = monitor.observe_depth_event({"U": 100, "u": 101, "b": [["74100", "1"]], "a": [["74101", "1"]]})
    assert buffering.state == "buffering_snapshot"

    bootstrap = monitor.load_snapshot(
        {"lastUpdateId": 100, "bids": [["74100", "1"]], "asks": [["74101", "1"]]},
        reason="bootstrap",
    )
    assert bootstrap.state == "healthy"
    assert monitor.local_update_id == 101

    broken = monitor.observe_depth_event({"U": 103, "u": 104, "b": [["74100", "2"]], "a": []})
    assert broken.requires_resnapshot is True
    assert broken.state == "resync_required"
    assert broken.quality_class == "continuity_compromised"
    assert broken.lifecycle_events[0]["event"] == "order_book_continuity_broken"

    monitor.observe_depth_event({"U": 105, "u": 106, "b": [["74100", "3"]], "a": []})
    resynced = monitor.load_snapshot(
        {"lastUpdateId": 104, "bids": [["74100", "2"]], "asks": [["74101", "1"]]},
        reason="resync",
    )
    assert resynced.state == "healthy"
    assert monitor.local_update_id == 106
    assert [event["event"] for event in resynced.lifecycle_events] == [
        "order_book_resync_started",
        "order_book_resynced",
    ]


def test_binance_order_book_parity_mismatch_is_reported() -> None:
    monitor = BinanceOrderBookIntegrityMonitor(symbol="BTCUSDT", market="usdm")
    monitor.observe_depth_event({"U": 200, "u": 201, "b": [["74100", "1"]], "a": [["74101", "1"]]})
    monitor.load_snapshot(
        {"lastUpdateId": 200, "bids": [["74100", "1"]], "asks": [["74101", "1"]]},
        reason="bootstrap",
    )
    mismatch = monitor.observe_periodic_snapshot(
        {"lastUpdateId": 201, "bids": [["74100", "9"]], "asks": [["74101", "1"]]}
    )
    assert mismatch.state == "parity_mismatch"
    assert mismatch.quality_class == "continuity_compromised"
    assert mismatch.lifecycle_events[0]["event"] == "order_book_parity_mismatch"
    assert mismatch.requires_resnapshot is True


def test_binance_order_book_deletes_zero_quantity_levels_and_recomputes_best() -> None:
    monitor = BinanceOrderBookIntegrityMonitor(symbol="BTCUSDT", market="spot")
    monitor.observe_depth_event({"U": 100, "u": 101, "b": [["74100", "1.00000000"]], "a": [["74101", "1.00000000"]]})
    monitor.load_snapshot(
        {
            "lastUpdateId": 100,
            "bids": [["74100", "1.00000000"], ["74099", "2.00000000"]],
            "asks": [["74101", "1.00000000"]],
        },
        reason="bootstrap",
    )

    update = monitor.observe_depth_event({"U": 102, "u": 102, "b": [["74100", "0.00000000"]], "a": []})
    assert update.state == "healthy"
    verified = monitor.observe_periodic_snapshot(
        {
            "lastUpdateId": 102,
            "bids": [["74099", "2.00000000"]],
            "asks": [["74101", "1.00000000"]],
        }
    )
    assert verified.state == "parity_verified"


def test_binance_order_book_snapshot_alignment_accepts_last_update_plus_one_bridge() -> None:
    monitor = BinanceOrderBookIntegrityMonitor(symbol="BTCUSDT", market="usdm")
    monitor.observe_depth_event({"U": 100, "u": 101, "pu": 99, "b": [["74100", "1"]], "a": [["74101", "1"]]})

    bootstrap = monitor.load_snapshot(
        {"lastUpdateId": 99, "bids": [["74099", "2"]], "asks": [["74101", "1"]]},
        reason="bootstrap",
    )
    assert bootstrap.state == "healthy"
    assert monitor.local_update_id == 101


def test_binance_usdm_first_bridge_event_allows_snapshot_mismatch_in_pu() -> None:
    monitor = BinanceOrderBookIntegrityMonitor(symbol="BTCUSDT", market="usdm")
    monitor.observe_depth_event({"U": 104, "u": 110, "pu": 103, "b": [["74100", "1"]], "a": [["74101", "1"]]})

    bootstrap = monitor.load_snapshot(
        {"lastUpdateId": 108, "bids": [["74099", "2"]], "asks": [["74101", "1"]]},
        reason="bootstrap",
    )

    assert bootstrap.state == "healthy"
    assert monitor.local_update_id == 110
    assert monitor.awaiting_bridge_event is False


def test_binance_usdm_uses_pu_continuity_instead_of_strict_update_plus_one() -> None:
    monitor = BinanceOrderBookIntegrityMonitor(symbol="BTCUSDT", market="usdm")
    monitor.observe_depth_event({"U": 100, "u": 101, "pu": 99, "b": [["74100", "1"]], "a": [["74101", "1"]]})
    monitor.load_snapshot(
        {"lastUpdateId": 99, "bids": [["74099", "2"]], "asks": [["74101", "1"]]},
        reason="bootstrap",
    )

    update = monitor.observe_depth_event({"U": 112, "u": 120, "pu": 101, "b": [["74100", "3"]], "a": []})

    assert update.state == "healthy"
    assert update.requires_resnapshot is False
    assert monitor.local_update_id == 120


def test_binance_order_book_ignores_duplicate_final_update_id_as_stale() -> None:
    monitor = BinanceOrderBookIntegrityMonitor(symbol="BTCUSDT", market="spot")
    monitor.observe_depth_event({"U": 100, "u": 101, "b": [["74100", "1"]], "a": [["74101", "1"]]})
    monitor.load_snapshot(
        {"lastUpdateId": 100, "bids": [["74100", "1"]], "asks": [["74101", "1"]]},
        reason="bootstrap",
    )

    duplicate = monitor.observe_depth_event({"U": 101, "u": 101, "b": [["74100", "1"]], "a": []})

    assert duplicate.state == "stale_event_ignored"
    assert duplicate.requires_resnapshot is False
