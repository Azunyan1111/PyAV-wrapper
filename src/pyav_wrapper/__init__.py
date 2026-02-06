from __future__ import annotations

import atexit
import os
import sys
import threading
import traceback
from multiprocessing import shared_memory
from typing import Any

_SHM_TRACE_LOCK = threading.Lock()
_SHM_TRACE_ENABLED = False
_SHM_TRACE_RECORDS: dict[int, dict[str, Any]] = {}
_SHM_TRACE_ORIGINALS: dict[str, Any] = {}
_SHM_TRACE_SHOW_STACK = os.getenv("PYAV_WRAPPER_TRACE_SHM_STACK", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _write_shm_trace(message: str) -> None:
    try:
        sys.stderr.write(f"[pyav_wrapper][shm-trace] {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def enable_shared_memory_tracking() -> None:
    """SharedMemoryの生成/解放をstderrへ追跡出力する。"""
    global _SHM_TRACE_ENABLED
    with _SHM_TRACE_LOCK:
        if _SHM_TRACE_ENABLED:
            return

        _SHM_TRACE_ORIGINALS["init"] = shared_memory.SharedMemory.__init__
        _SHM_TRACE_ORIGINALS["close"] = shared_memory.SharedMemory.close
        _SHM_TRACE_ORIGINALS["unlink"] = shared_memory.SharedMemory.unlink

        original_init = _SHM_TRACE_ORIGINALS["init"]
        original_close = _SHM_TRACE_ORIGINALS["close"]
        original_unlink = _SHM_TRACE_ORIGINALS["unlink"]

        def _tracked_init(self, *args, **kwargs):
            create = bool(kwargs.get("create", False))
            if len(args) >= 2:
                create = bool(args[1])
            original_init(self, *args, **kwargs)

            record = {
                "name": getattr(self, "name", "<unknown>"),
                "create": create,
                "closed": False,
                "unlinked": False,
                "stack": "".join(traceback.format_stack(limit=40)),
            }
            with _SHM_TRACE_LOCK:
                _SHM_TRACE_RECORDS[id(self)] = record
            _write_shm_trace(
                f"init name={record['name']} create={record['create']}"
            )
            if _SHM_TRACE_SHOW_STACK:
                _write_shm_trace(f"creation_stack name={record['name']}\n{record['stack']}")

        def _tracked_close(self, *args, **kwargs):
            try:
                return original_close(self, *args, **kwargs)
            finally:
                with _SHM_TRACE_LOCK:
                    record = _SHM_TRACE_RECORDS.get(id(self))
                    if record is not None:
                        record["closed"] = True
                        name = record["name"]
                    else:
                        name = getattr(self, "name", "<unknown>")
                _write_shm_trace(f"close name={name}")

        def _tracked_unlink(self, *args, **kwargs):
            try:
                return original_unlink(self, *args, **kwargs)
            finally:
                with _SHM_TRACE_LOCK:
                    record = _SHM_TRACE_RECORDS.get(id(self))
                    if record is not None:
                        record["unlinked"] = True
                        name = record["name"]
                    else:
                        name = getattr(self, "name", "<unknown>")
                _write_shm_trace(f"unlink name={name}")

        def _dump_leftover_records() -> None:
            with _SHM_TRACE_LOCK:
                leftovers = [
                    record
                    for record in _SHM_TRACE_RECORDS.values()
                    if record.get("create") and not record.get("unlinked")
                ]
            if not leftovers:
                return
            _write_shm_trace(
                "possible leaks detected: "
                f"{len(leftovers)} created shared_memory objects not unlinked"
            )
            for index, record in enumerate(leftovers[:20], start=1):
                _write_shm_trace(
                    f"[{index}] name={record['name']} closed={record['closed']} unlinked={record['unlinked']}\n"
                    f"{record['stack']}"
                )

        shared_memory.SharedMemory.__init__ = _tracked_init
        shared_memory.SharedMemory.close = _tracked_close
        shared_memory.SharedMemory.unlink = _tracked_unlink
        atexit.register(_dump_leftover_records)
        _SHM_TRACE_ENABLED = True
        _write_shm_trace("tracking enabled")


if os.getenv("PYAV_WRAPPER_TRACE_SHM", "").lower() in {"1", "true", "yes", "on"}:
    enable_shared_memory_tracking()

from pyav_wrapper.audio_frame import WrappedAudioFrame
from pyav_wrapper.raw_subprocess_pipe_stream_listener import RawSubprocessPipeStreamListener
from pyav_wrapper.raw_subprocess_pipe_stream_writer import RawSubprocessPipeStreamWriter
from pyav_wrapper.stream_listener import StreamListener
from pyav_wrapper.stream_writer import StreamWriter
from pyav_wrapper.video_frame import WrappedVideoFrame

__all__ = [
    "WrappedAudioFrame",
    "WrappedVideoFrame",
    "StreamListener",
    "StreamWriter",
    "RawSubprocessPipeStreamListener",
    "RawSubprocessPipeStreamWriter",
    "enable_shared_memory_tracking",
]
