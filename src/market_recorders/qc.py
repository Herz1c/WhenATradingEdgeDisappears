from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from binance_recorder.compression import iter_zstd_jsonl
from market_recorders.archive import manifest_file_state
from market_recorders.file_quality import manifest_quality_class
from market_recorders.source_registry import source_registry_metadata
from market_recorders.transport_qc import TransportTelemetryAccumulator
from polymarket_recorder.rtds_qc import summarize_rtds_stream
from chainlink_recorder.alignment import build_chainlink_alignment_report

LIFECYCLE_ALERT_EVENTS = {
    "service_crash",
    "websocket_error",
    "http_error",
    "manifest_write_error",
    "file_write_error",
    "sequence_gap_detected",
    "out_of_order_detected",
    "snapshot_reset_required",
    "channel_resubscribe_triggered",
    "heartbeat_gap_detected",
    "heartbeat_stale_detected",
    "order_book_continuity_broken",
    "order_book_resync_started",
    "order_book_parity_mismatch",
    "bybit_sequence_continuity_break",
    "bybit_snapshot_reset_received",
    "freshness_spike_detected",
    "silent_gap_detected",
    "backlog_burst_detected",
    "rest_health_state_changed",
}


def _infer_kind(path: Path) -> str:
    stem = path.name.removesuffix(".jsonl.zst")
    return stem.rsplit(".", 1)[-1]


def _source_key(relative: Path) -> str:
    parts = relative.parts
    if not parts:
        return ""
    if parts[0] == "fng":
        return "fng"
    if parts[0] == "polymarket" and len(parts) >= 2 and parts[1] == "btc_updown_5m":
        return "polymarket/btc_updown_5m"
    if parts[0] == "polymarket" and len(parts) >= 4 and parts[1] == "rtds":
        return "/".join(parts[:4])
    return "/".join(parts[:3]) if len(parts) >= 3 else parts[0]


def _expected_manifest_path(root: Path, raw_path: Path) -> Path:
    relative = raw_path.relative_to(root / "raw")
    return root / "manifests" / relative.parent / relative.name.replace(".jsonl.zst", ".manifest.json")


def build_daily_qc_report(root: Path, target_date: date) -> dict[str, Any]:
    root = root.resolve()
    raw_root = root / "raw"
    manifests_root = root / "manifests"
    report: dict[str, Any] = {
        "date": target_date.isoformat(),
        "root": str(root),
        "raw_file_count": 0,
        "manifest_file_count": 0,
        "empty_raw_files": [],
        "missing_manifest_files": [],
        "active_manifest_files": [],
        "manifest_zero_record_files": [],
        "non_finalized_manifest_files": [],
        "recovered_manifest_files": [],
        "corrupt_or_unreadable_files": [],
        "sources": {},
    }
    sources: dict[str, dict[str, Any]] = {}
    for raw_path in raw_root.rglob("*.jsonl.zst"):
        if target_date.isoformat() not in raw_path.parts:
            continue
        report["raw_file_count"] += 1
        relative = raw_path.relative_to(raw_root)
        source_key = _source_key(relative)
        source_bucket = sources.setdefault(
            source_key,
            {
                "raw_file_count": 0,
                "kind_file_counts": Counter(),
                "material_kind_file_counts": Counter(),
                "manifested_record_count_by_kind": Counter(),
                "manifest_file_state_counts": Counter(),
                "source_tier_counts": Counter(),
                "timestamp_semantics_counts": Counter(),
                "recv_ts_sources": Counter(),
                "lifecycle_alert_event_counts": Counter(),
                "quality_class_counts": Counter(),
                "active_manifest_count": 0,
                "non_finalized_manifest_count": 0,
                "recovered_manifest_count": 0,
                "_transport_qc": None,
                "_transport_qc_by_endpoint": {},
            },
        )
        source_bucket["raw_file_count"] += 1
        kind = _infer_kind(raw_path)
        source_bucket["kind_file_counts"][kind] += 1
        if kind != "lifecycle":
            source_bucket["material_kind_file_counts"][kind] += 1
        if raw_path.stat().st_size == 0:
            report["empty_raw_files"].append(str(relative).replace("\\", "/"))
        manifest_path = _expected_manifest_path(root, raw_path)
        if not manifest_path.exists():
            report["missing_manifest_files"].append(str(relative).replace("\\", "/"))
        else:
            report["manifest_file_count"] += 1
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            count = int(manifest_payload.get("record_count", 0))
            source_bucket["manifested_record_count_by_kind"][kind] += count
            source_tier = manifest_payload.get("source_tier") or source_registry_metadata(relative, kind=kind)["source_tier"]
            source_bucket["source_tier_counts"][str(source_tier)] += 1
            file_state = str(
                manifest_payload.get("file_state")
                or manifest_file_state(
                    finalized=bool(manifest_payload.get("finalized", True)),
                    recovered_from_raw=bool(manifest_payload.get("recovered_from_raw", False)),
                )
            )
            source_bucket["manifest_file_state_counts"][file_state] += 1
            quality_class = manifest_quality_class(manifest_payload)
            if quality_class:
                source_bucket["quality_class_counts"][quality_class] += 1
            if count == 0:
                report["manifest_zero_record_files"].append(
                    str(manifest_path.relative_to(manifests_root)).replace("\\", "/")
                )
            if file_state == "active":
                source_bucket["active_manifest_count"] += 1
                report["active_manifest_files"].append(
                    str(manifest_path.relative_to(manifests_root)).replace("\\", "/")
                )
            if not bool(manifest_payload.get("finalized", True)):
                source_bucket["non_finalized_manifest_count"] += 1
                report["non_finalized_manifest_files"].append(
                    str(manifest_path.relative_to(manifests_root)).replace("\\", "/")
                )
            if bool(manifest_payload.get("recovered_from_raw", False)):
                source_bucket["recovered_manifest_count"] += 1
                report["recovered_manifest_files"].append(
                    str(manifest_path.relative_to(manifests_root)).replace("\\", "/")
                )
        try:
            for record in iter_zstd_jsonl(raw_path):
                recv_ts_source = record.get("recv_ts_source")
                if recv_ts_source:
                    source_bucket["recv_ts_sources"][str(recv_ts_source)] += 1
                if kind == "lifecycle":
                    event_type = str(record.get("event") or record.get("event_type") or "")
                    if event_type in LIFECYCLE_ALERT_EVENTS:
                        source_bucket["lifecycle_alert_event_counts"][event_type] += 1
                if str(record.get("source")) == "polymarket_rtds_chainlink":
                    rtds_records = source_bucket.setdefault("_rtds_records", [])
                    rtds_records.append(record)
                if kind != "lifecycle":
                    timestamp_semantics = record.get("timestamp_semantics")
                    if timestamp_semantics:
                        source_bucket["timestamp_semantics_counts"][str(timestamp_semantics)] += 1
                    if source_bucket["_transport_qc"] is None:
                        source_bucket["_transport_qc"] = TransportTelemetryAccumulator(source_key=source_key)
                    source_bucket["_transport_qc"].observe(record)
                    endpoint = record.get("endpoint")
                    if endpoint:
                        endpoint_key = str(endpoint)
                        endpoint_accumulator = source_bucket["_transport_qc_by_endpoint"].setdefault(
                            endpoint_key,
                            TransportTelemetryAccumulator(source_key=f"{source_key}:{endpoint_key}"),
                        )
                        endpoint_accumulator.observe(record)
        except Exception as exc:
            report["corrupt_or_unreadable_files"].append(
                {"file": str(relative).replace("\\", "/"), "error": repr(exc)}
            )
    report["sources"] = {
        name: {
            "raw_file_count": bucket["raw_file_count"],
            "kind_file_counts": dict(bucket["kind_file_counts"]),
            "material_kind_file_counts": dict(bucket["material_kind_file_counts"]),
            "manifested_record_count_by_kind": dict(bucket["manifested_record_count_by_kind"]),
            "manifest_file_state_counts": dict(bucket["manifest_file_state_counts"]),
            "source_tier_counts": dict(bucket["source_tier_counts"]),
            "timestamp_semantics_counts": dict(bucket["timestamp_semantics_counts"]),
            "quality_class_counts": dict(bucket["quality_class_counts"]),
            "recv_ts_sources": dict(bucket["recv_ts_sources"]),
            "lifecycle_alert_event_counts": dict(bucket["lifecycle_alert_event_counts"]),
            "active_manifest_count": bucket["active_manifest_count"],
            "non_finalized_manifest_count": bucket["non_finalized_manifest_count"],
            "recovered_manifest_count": bucket["recovered_manifest_count"],
            **({"transport_qc": bucket["_transport_qc"].summary()} if bucket.get("_transport_qc") else {}),
            **(
                {
                    "transport_qc_by_endpoint": {
                        endpoint: accumulator.summary()
                        for endpoint, accumulator in sorted(bucket["_transport_qc_by_endpoint"].items())
                    }
                }
                if bucket.get("_transport_qc_by_endpoint")
                else {}
            ),
            **(
                {"rtds_freshness_qc": summarize_rtds_stream(bucket["_rtds_records"])}
                if bucket.get("_rtds_records")
                else {}
            ),
        }
        for name, bucket in sorted(sources.items())
    }
    report["summary"] = {
        "sources_with_missing_manifests": sorted(
            {
                source
                for source in report["sources"]
                if any(item.startswith(source) for item in report["missing_manifest_files"])
            }
        ),
        "sources_with_lifecycle_alerts": sorted(
            {
                source
                for source, bucket in report["sources"].items()
                if bucket["lifecycle_alert_event_counts"]
            }
        ),
        "sources_with_partial_tails": sorted(
            {
                source
                for source, bucket in report["sources"].items()
                if bucket["non_finalized_manifest_count"]
            }
        ),
        "sources_with_active_files": sorted(
            {
                source
                for source, bucket in report["sources"].items()
                if bucket["active_manifest_count"]
            }
        ),
        "sources_with_recovered_files": sorted(
            {
                source
                for source, bucket in report["sources"].items()
                if bucket["recovered_manifest_count"]
            }
        ),
        "sources_with_quality_issues": sorted(
            {
                source
                for source, bucket in report["sources"].items()
                if any(
                    quality_class in bucket["quality_class_counts"]
                    for quality_class in ("continuity_compromised", "stale_context", "recovered")
                )
            }
        ),
        "sources_with_transport_alerts": sorted(
            {
                source
                for source, bucket in report["sources"].items()
                if bucket.get("transport_qc", {}).get("warnings")
            }
        ),
        "sources_with_only_lifecycle_output": sorted(
            {
                source
                for source, bucket in report["sources"].items()
                if bucket["raw_file_count"] > 0 and not bucket["material_kind_file_counts"]
            }
        ),
    }
    chainlink_validation = build_chainlink_alignment_report(
        root=root,
        date_from=target_date,
        date_to=target_date,
        finalized_only=True,
        tail_tolerant=False,
        by_hour=True,
    )
    report["chainlink_proxy_validation"] = chainlink_validation
    return report
