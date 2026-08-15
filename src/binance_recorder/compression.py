from __future__ import annotations

import io
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import orjson
import zstandard as zstd


class ZstdJsonlAppender:
    def __init__(self, path: Path, level: int = 3) -> None:
        self.path = path
        self._compressor = zstd.ZstdCompressor(level=level)
        self._file_handle: Any | None = None
        self._writer: Any | None = None
        self._closed = False
        self._has_written = False

    @property
    def has_written(self) -> bool:
        return self._has_written

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed appender")
        if self._writer is not None and self._file_handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = self.path.open("ab")
        self._writer = self._compressor.stream_writer(self._file_handle, closefd=False)

    def write_json(self, record: dict[str, Any]) -> int:
        self._ensure_open()
        payload = orjson.dumps(record)
        self._has_written = True
        assert self._writer is not None
        return self._writer.write(payload + b"\n")

    def flush(self, *, frame: bool = False) -> None:
        if self._closed or self._writer is None or self._file_handle is None:
            return
        self._writer.flush(zstd.FLUSH_FRAME if frame else zstd.FLUSH_BLOCK)
        self._file_handle.flush()

    def close(self) -> None:
        if self._closed:
            return
        if self._writer is None or self._file_handle is None:
            self._closed = True
            return
        self._writer.flush(zstd.FLUSH_FRAME)
        self._writer.close()
        self._file_handle.flush()
        os.fsync(self._file_handle.fileno())
        self._file_handle.close()
        self._closed = True


def iter_zstd_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    yield from iter_zstd_jsonl_with_options(path)


def iter_zstd_jsonl_with_options(
    path: Path,
    *,
    allow_truncated_stream: bool = False,
    allow_partial_final_line: bool = False,
) -> Iterator[dict[str, Any]]:
    if not allow_truncated_stream and not allow_partial_final_line:
        with path.open("rb") as file_handle:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(file_handle) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if line:
                        yield orjson.loads(line)
        return

    with path.open("rb") as file_handle:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(file_handle) as reader:
            buffer = b""
            while True:
                try:
                    chunk = reader.read(65_536)
                except zstd.ZstdError:
                    if not allow_truncated_stream:
                        raise
                    break
                if not chunk:
                    break
                buffer += chunk
                while True:
                    newline_index = buffer.find(b"\n")
                    if newline_index < 0:
                        break
                    line = buffer[:newline_index].strip()
                    buffer = buffer[newline_index + 1 :]
                    if line:
                        yield orjson.loads(line)

            final_line = buffer.strip()
            if not final_line:
                return
            try:
                yield orjson.loads(final_line)
            except orjson.JSONDecodeError:
                if not allow_partial_final_line:
                    raise


def quarantine_corrupt_archive(
    path: Path,
    *,
    logger: logging.Logger | None = None,
    reason: Exception | None = None,
) -> Path:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = path.with_name(f"{path.name}.corrupt-{timestamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.corrupt-{timestamp}-{counter}")
        counter += 1
    path.rename(candidate)
    if logger is not None:
        logger.warning(
            "Quarantined unreadable archive file",
            extra={
                "original_path": str(path),
                "quarantined_path": str(candidate),
                "reason": repr(reason) if reason is not None else None,
            },
        )
    return candidate
