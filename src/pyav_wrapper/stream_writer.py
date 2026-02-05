from __future__ import annotations

import fractions
import multiprocessing
import os
import pickle
import queue
from multiprocessing import resource_tracker, shared_memory
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any

import av
import numpy as np

from pyav_wrapper.audio_frame import WrappedAudioFrame
from pyav_wrapper.video_frame import WrappedVideoFrame

if TYPE_CHECKING:
    from threading import Thread

if os.name == "posix":
    import _posixshmem


def _serialize_frame_common(frame: "av.frame.Frame") -> dict[str, Any]:
    time_base = None
    if frame.time_base is not None:
        time_base = (frame.time_base.numerator, frame.time_base.denominator)

    payload = {
        "pts": frame.pts,
        "dts": frame.dts,
        "duration": frame.duration,
        "time_base": time_base,
    }
    opaque = frame.opaque
    if opaque is not None and _is_picklable(opaque):
        payload["opaque"] = opaque
    return payload


def _is_picklable(value: Any) -> bool:
    try:
        pickle.dumps(value)
        return True
    except Exception:
        return False


_SHM_VIDEO_THRESHOLD_BYTES = 256 * 1024
_SHM_AUDIO_THRESHOLD_BYTES = 64 * 1024


def _unregister_created_shared_memory(shm_name: str) -> None:
    if not shm_name:
        return
    tracker_name = shm_name
    if os.name == "posix" and not tracker_name.startswith("/"):
        tracker_name = f"/{tracker_name}"
    try:
        resource_tracker.unregister(tracker_name, "shared_memory")
    except Exception:
        pass


def _attach_shared_memory(shm_name: str) -> shared_memory.SharedMemory | None:
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    _unregister_created_shared_memory(shm.name)
    return shm


def _close_and_unlink_shared_memory(shm: shared_memory.SharedMemory) -> None:
    shm_name = shm.name
    try:
        shm.close()
    except Exception:
        pass
    if os.name == "posix":
        target_name = shm_name
        if not target_name.startswith("/"):
            target_name = f"/{target_name}"
        try:
            _posixshmem.shm_unlink(target_name)
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _cleanup_payload_shared_memory(payload: Any) -> None:
    if isinstance(payload, list):
        for item in payload:
            _cleanup_payload_shared_memory(item)
        return
    if not isinstance(payload, dict):
        return
    if payload.get("storage") != "shm":
        return
    shm_name = payload.get("shm_name")
    if not isinstance(shm_name, str) or not shm_name:
        return
    shm = _attach_shared_memory(shm_name)
    if shm is None:
        return
    _close_and_unlink_shared_memory(shm)


def _pack_bytes_to_shared_memory(
    chunks: list[bytes],
    threshold_bytes: int,
) -> tuple[str, list[tuple[int, int]]] | None:
    total_size = sum(len(chunk) for chunk in chunks)
    if total_size < threshold_bytes:
        return None
    try:
        shm = shared_memory.SharedMemory(create=True, size=total_size)
    except Exception:
        return None

    shm_name = shm.name
    spans: list[tuple[int, int]] = []
    cursor = 0
    try:
        for chunk in chunks:
            chunk_size = len(chunk)
            shm.buf[cursor: cursor + chunk_size] = chunk
            spans.append((cursor, chunk_size))
            cursor += chunk_size
    except Exception:
        _close_and_unlink_shared_memory(shm)
        _unregister_created_shared_memory(shm_name)
        return None
    finally:
        try:
            shm.close()
        except Exception:
            pass
    _unregister_created_shared_memory(shm_name)

    return shm_name, spans


def _serialize_video_frame(
    frame: WrappedVideoFrame,
    include_planes: bool = True,
) -> dict[str, Any]:
    planes: list[bytes] | None = None
    plane_spans: list[tuple[int, int]] | None = None
    shm_name: str | None = None
    storage = "inline"
    if include_planes:
        planes = []
        for plane in frame.frame.planes:
            planes.append(bytes(plane))
        shm_payload = _pack_bytes_to_shared_memory(planes, _SHM_VIDEO_THRESHOLD_BYTES)
        if shm_payload is not None:
            shm_name, plane_spans = shm_payload
            planes = None
            storage = "shm"

    payload = _serialize_frame_common(frame.frame)
    payload.update(
        {
            "format": frame.frame.format.name,
            "width": frame.frame.width,
            "height": frame.frame.height,
            "planes": planes,
            "plane_spans": plane_spans,
            "storage": storage,
            "shm_name": shm_name,
            "pict_type": frame.frame.pict_type,
            "colorspace": frame.frame.colorspace,
            "color_range": frame.frame.color_range,
            "is_bad_frame": frame.is_bad_frame,
        }
    )
    return payload


def _serialize_audio_frame(frame: WrappedAudioFrame) -> dict[str, Any]:
    data = np.ascontiguousarray(frame.frame.to_ndarray())
    raw_bytes = data.tobytes()
    shm_payload = _pack_bytes_to_shared_memory([raw_bytes], _SHM_AUDIO_THRESHOLD_BYTES)
    storage = "inline"
    shm_name: str | None = None
    data_spans: list[tuple[int, int]] | None = None

    payload = _serialize_frame_common(frame.frame)
    if shm_payload is None:
        payload.update(
            {
                "format": frame.frame.format.name,
                "layout": frame.frame.layout.name,
                "sample_rate": frame.frame.sample_rate,
                "rate": frame.frame.rate,
                "samples": frame.frame.samples,
                "data": data,
                "storage": storage,
                "shm_name": shm_name,
                "data_spans": data_spans,
            }
        )
        return payload

    shm_name, data_spans = shm_payload
    storage = "shm"
    payload.update(
        {
            "format": frame.frame.format.name,
            "layout": frame.frame.layout.name,
            "sample_rate": frame.frame.sample_rate,
            "rate": frame.frame.rate,
            "samples": frame.frame.samples,
            "data": None,
            "storage": storage,
            "shm_name": shm_name,
            "data_spans": data_spans,
            "data_shape": data.shape,
            "data_dtype": str(data.dtype),
        }
    )
    return payload


def _apply_common_frame_attrs(frame: "av.frame.Frame", payload: dict[str, Any]) -> None:
    for key in ("pts", "dts", "duration", "side_data", "opaque"):
        if key in payload:
            try:
                setattr(frame, key, payload[key])
            except Exception:
                pass
    time_base = payload.get("time_base")
    if time_base is not None:
        try:
            frame.time_base = fractions.Fraction(time_base[0], time_base[1])
        except Exception:
            pass


def _apply_video_frame_attrs(frame: av.VideoFrame, payload: dict[str, Any]) -> None:
    for key in ("pict_type", "colorspace", "color_range"):
        if key in payload:
            try:
                setattr(frame, key, payload[key])
            except Exception:
                pass


def _apply_audio_frame_attrs(frame: av.AudioFrame, payload: dict[str, Any]) -> None:
    for key in ("sample_rate", "rate", "samples", "format", "layout"):
        if key in payload:
            try:
                setattr(frame, key, payload[key])
            except Exception:
                pass


def _deserialize_video_frame(payload: dict[str, Any]) -> WrappedVideoFrame:
    frame = av.VideoFrame(
        payload["width"],
        payload["height"],
        payload["format"],
    )
    _apply_common_frame_attrs(frame, payload)
    _apply_video_frame_attrs(frame, payload)
    wrapped = WrappedVideoFrame(frame)
    storage = payload.get("storage", "inline")
    if storage == "shm":
        shm_name = payload.get("shm_name")
        plane_spans = payload.get("plane_spans")
        if isinstance(shm_name, str) and isinstance(plane_spans, list):
            shm = None
            try:
                shm = _attach_shared_memory(shm_name)
                if shm is None:
                    return wrapped
                for i, span in enumerate(plane_spans):
                    if not isinstance(span, (list, tuple)) or len(span) != 2:
                        continue
                    offset = int(span[0])
                    size = int(span[1])
                    try:
                        frame.planes[i].update(shm.buf[offset: offset + size])
                    except Exception:
                        continue
            except Exception:
                pass
            finally:
                if shm is not None:
                    _close_and_unlink_shared_memory(shm)
                else:
                    _cleanup_payload_shared_memory(payload)
    else:
        planes = payload.get("planes")
        if planes is not None:
            for i, plane_data in enumerate(planes):
                try:
                    frame.planes[i].update(plane_data)
                except Exception:
                    continue
    wrapped.is_bad_frame = payload.get("is_bad_frame", False)
    return wrapped


def _deserialize_audio_frame(payload: dict[str, Any]) -> WrappedAudioFrame:
    storage = payload.get("storage", "inline")
    if storage == "shm":
        shm_name = payload.get("shm_name")
        data_spans = payload.get("data_spans")
        data_shape = payload.get("data_shape")
        data_dtype = payload.get("data_dtype")
        if (
            isinstance(shm_name, str)
            and isinstance(data_spans, list)
            and len(data_spans) > 0
            and isinstance(data_shape, (list, tuple))
            and isinstance(data_dtype, str)
        ):
            span = data_spans[0]
            if isinstance(span, (list, tuple)) and len(span) == 2:
                offset = int(span[0])
                size = int(span[1])
                shm = None
                try:
                    shm = _attach_shared_memory(shm_name)
                    if shm is not None:
                        raw = bytes(shm.buf[offset: offset + size])
                        data = np.frombuffer(raw, dtype=np.dtype(data_dtype)).reshape(tuple(data_shape))
                        frame = av.AudioFrame.from_ndarray(
                            data,
                            format=payload["format"],
                            layout=payload["layout"],
                        )
                        _apply_common_frame_attrs(frame, payload)
                        _apply_audio_frame_attrs(frame, payload)
                        return WrappedAudioFrame(frame)
                finally:
                    if shm is not None:
                        _close_and_unlink_shared_memory(shm)
                    else:
                        _cleanup_payload_shared_memory(payload)
    frame = av.AudioFrame.from_ndarray(
        payload["data"],
        format=payload["format"],
        layout=payload["layout"],
    )
    _apply_common_frame_attrs(frame, payload)
    _apply_audio_frame_attrs(frame, payload)
    return WrappedAudioFrame(frame)


def _put_with_drop(
    target_queue: multiprocessing.Queue,
    item: Any,
    drop_count: int,
) -> None:
    try:
        target_queue.put_nowait(item)
        return
    except queue.Full:
        for _ in range(drop_count):
            try:
                dropped = target_queue.get_nowait()
                _cleanup_payload_shared_memory(dropped)
            except queue.Empty:
                break
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            _cleanup_payload_shared_memory(item)
            return
    except (ValueError, OSError):
        _cleanup_payload_shared_memory(item)
        return


def _drop_if_full(
    target_queue: multiprocessing.Queue,
    drop_count: int,
) -> bool:
    try:
        if not target_queue.full():
            return True
    except NotImplementedError:
        return True
    except (ValueError, OSError):
        return False

    for _ in range(drop_count):
        try:
            dropped = target_queue.get_nowait()
            _cleanup_payload_shared_memory(dropped)
        except queue.Empty:
            break
        except (ValueError, OSError):
            return False

    try:
        return not target_queue.full()
    except NotImplementedError:
        return True
    except (ValueError, OSError):
        return False


def _is_process_alive(process: multiprocessing.Process | None) -> bool:
    if process is None:
        return False
    try:
        return process.is_alive()
    except (AssertionError, ValueError, OSError):
        return False


def _is_thread_alive(thread: Any) -> bool:
    if thread is None:
        return False
    try:
        return bool(thread.is_alive())
    except Exception:
        return False


class _StreamWriterContainerProxy:
    def __init__(self, on_close) -> None:
        self._on_close = on_close
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_close()

    def __bool__(self) -> bool:
        return not self._closed


def _stream_writer_worker(
    url: str,
    width: int,
    height: int,
    fps: int,
    sample_rate: int,
    audio_layout: str,
    stats_enabled: bool,
    stop_event: multiprocessing.Event,
    video_queue: multiprocessing.Queue,
    audio_queue: multiprocessing.Queue,
    status_queue: multiprocessing.Queue | None,
    control_queue: multiprocessing.Queue | None,
    crop_ratio: float | None = None,
) -> None:
    container: av.Container | None = None
    video_stream = None
    audio_stream = None
    last_video_frame: WrappedVideoFrame | None = None

    stats_video_frame_count: int = 0
    stats_audio_frame_count: int = 0
    stats_last_time: float = time.monotonic()
    stats_status_last_time: float = time.monotonic()
    pending_video_count: int = 0
    pending_audio_count: int = 0
    last_write_time: float | None = None

    def _notify_status(status: str, payload: dict[str, Any] | None = None) -> None:
        if status_queue is None:
            return
        message: dict[str, Any] = {"status": status}
        if payload:
            message.update(payload)
        try:
            status_queue.put_nowait(message)
        except Exception:
            pass

    def _flush_status_if_needed(force: bool = False) -> None:
        nonlocal pending_video_count, pending_audio_count, stats_status_last_time, last_write_time
        now = time.monotonic()
        if not force:
            if pending_video_count == 0 and pending_audio_count == 0:
                return
            if now - stats_status_last_time < 1.0:
                return
        if status_queue is not None:
            message = {
                "status": "write_stats",
                "video_count": pending_video_count,
                "audio_count": pending_audio_count,
                "elapsed": now - stats_status_last_time,
            }
            if last_write_time is not None:
                message["last_write_time"] = last_write_time
            try:
                status_queue.put_nowait(message)
            except Exception:
                pass
        pending_video_count = 0
        pending_audio_count = 0
        stats_status_last_time = now

    def _wait_for_first_video_payload() -> dict[str, Any] | None:
        while not stop_event.is_set():
            try:
                return video_queue.get(timeout=1.0)
            except queue.Empty:
                continue
        return None

    def _open_container(target_width: int, target_height: int) -> tuple[av.Container, Any, Any]:
        if url.startswith("srt://"):
            next_container = av.open(url, mode="w", format="mpegts")
        else:
            next_container = av.open(url, mode="w")

        next_video_stream = next_container.add_stream("libx264", rate=fps)
        next_video_stream.width = target_width
        next_video_stream.height = target_height
        next_video_stream.pix_fmt = "yuv420p"
        next_video_stream.options = {"preset": "ultrafast", "tune": "zerolatency"}

        next_audio_stream = next_container.add_stream("aac", rate=sample_rate)
        next_audio_stream.layout = audio_layout

        return next_container, next_video_stream, next_audio_stream

    def _handle_control_commands() -> bool:
        if control_queue is None:
            return False
        close_requested = False
        while True:
            try:
                command = control_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(command, dict) and command.get("cmd") == "close_container":
                close_requested = True
        return close_requested

    try:
        first_payload = _wait_for_first_video_payload()
        if first_payload is None:
            return
        first_frame = _deserialize_video_frame(first_payload)

        # クロップ設定がある場合は適用
        if crop_ratio is not None:
            first_frame = first_frame.crop_center(crop_ratio)

        try:
            container, video_stream, audio_stream = _open_container(width, height)
            for packet in video_stream.encode(first_frame.frame):
                container.mux(packet)
            last_write_time = time.time()
            stats_video_frame_count += 1
            _notify_status("started", {"started_time": last_write_time})
        except Exception as e:
            _notify_status("error", {"message": str(e)})
            print(f"処理開始エラー: {str(e)}")
            return

        while not stop_event.is_set():
            if _handle_control_commands():
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass
                container = None
                break

            if container is None:
                break

            has_data = False

            try:
                video_payload = video_queue.get(timeout=1 / 30)
            except queue.Empty:
                video_payload = None

            if video_payload is not None:
                wrapped_frame = _deserialize_video_frame(video_payload)
                if wrapped_frame.is_bad_frame:
                    if last_video_frame is not None:
                        last_planes = last_video_frame.get_planes()
                        wrapped_frame.set_planes(last_planes)
                else:
                    last_video_frame = wrapped_frame

                # クロップ設定がある場合は適用
                if crop_ratio is not None:
                    wrapped_frame = wrapped_frame.crop_center(crop_ratio)

                has_data = True
                try:
                    for packet in video_stream.encode(wrapped_frame.frame):
                        container.mux(packet)
                    last_write_time = time.time()
                    stats_video_frame_count += 1
                    pending_video_count += 1
                except Exception as e:
                    print(f"StreamWriter映像muxエラー（スキップ）: {e}", file=sys.stderr)

            while True:
                try:
                    audio_payload = audio_queue.get_nowait()
                except queue.Empty:
                    break

                wrapped_audio = _deserialize_audio_frame(audio_payload)
                has_data = True
                try:
                    for packet in audio_stream.encode(wrapped_audio.frame):
                        container.mux(packet)
                    last_write_time = time.time()
                    stats_audio_frame_count += 1
                    pending_audio_count += 1
                except Exception as e:
                    print(f"StreamWriter音声muxエラー（スキップ）: {e}", file=sys.stderr)

            if not has_data:
                time.sleep(0.001)
            _flush_status_if_needed()

            if stats_enabled:
                now = time.monotonic()
                stats_elapsed = now - stats_last_time
                if stats_elapsed >= 5.0:
                    video_fps = stats_video_frame_count / stats_elapsed
                    audio_fps = stats_audio_frame_count / stats_elapsed
                    print(f"[Writer] video_fps={video_fps:.2f} audio_fps={audio_fps:.2f}")
                    stats_video_frame_count = 0
                    stats_audio_frame_count = 0
                    stats_last_time = now

    except Exception as e:
        print(f"StreamWriter書き込みエラー: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        _flush_status_if_needed(force=True)
        if container is not None:
            try:
                if video_stream is not None:
                    for packet in video_stream.encode():
                        container.mux(packet)
                if audio_stream is not None:
                    for packet in audio_stream.encode():
                        container.mux(packet)
                container.close()
            except Exception as e:
                print(f"StreamWriterクローズ時にエラー: {e}", file=sys.stderr)
                traceback.print_exc()


class StreamWriter:
    """PyAVを使用してSRTストリームへ映像・音声フレームを送信するクラス"""

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        fps: int = 30,
        sample_rate: int = 48000,
        audio_layout: str = "stereo",
        stats_enabled: bool = False,
    ) -> None:
        """
        Args:
            url: SRT出力URL（例: "srt://host:port?mode=caller&latency=120"）
            width: 出力映像幅
            height: 出力映像高さ
            fps: 出力フレームレート
            sample_rate: 音声サンプルレート（デフォルト: 48000）
            audio_layout: 音声チャンネルレイアウト（デフォルト: "stereo"）
            stats_enabled: FPS統計出力を有効にするかどうか
        """
        if width is None or height is None:
            raise ValueError("widthとheightは必須です")
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        self.audio_layout = audio_layout

        # キューの初期化（FPSの1.7倍のキャパシティ）
        self._video_queue_maxlen = int(self.fps * 1.7)
        self._audio_queue_maxlen = int(self.sample_rate * 1.7)
        self.video_queue: queue.Queue[WrappedVideoFrame] = queue.Queue(
            maxsize=self._video_queue_maxlen
        )
        self.audio_queue: queue.Queue[WrappedAudioFrame] = queue.Queue(
            maxsize=self._audio_queue_maxlen
        )

        # スレッド/プロセス管理
        self.is_running = False
        self._thread: "Thread | multiprocessing.Process | None" = None
        self._init_thread: "Thread | None" = None
        self._monitor_thread: threading.Thread | None = None

        # コンテナ/ストリーム
        self._container: _StreamWriterContainerProxy | None = None
        self._video_stream = None
        self._audio_stream = None

        # 最初のフレーム
        self._first_frame: WrappedVideoFrame | None = None

        # 古いフレームの保持
        self._last_video_frame: WrappedVideoFrame | None = None
        self._owner_pid = os.getpid()

        # Reconnection
        self._last_successful_write_time: float = 0.0
        self._restart_threshold: float = 10.0
        self._restart_threshold_increment: float = 1.0
        self._restart_threshold_max: float = 20.0
        self._restart_wait_seconds: float = 5.0
        self._restart_frame_wait_timeout: float = 30.0
        self._restart_lock: threading.Lock = threading.Lock()
        self._restart_in_progress = False

        # FPS統計
        self._stats_enabled = stats_enabled

        # Crop設定
        self._crop_ratio: float | None = None
        self._stats_video_frame_count: int = 0
        self._stats_audio_frame_count: int = 0
        self._stats_last_time: float = time.monotonic()
        self._worker_has_video_reference = False
        self._mp_video_fps: float | None = None
        self._mp_audio_fps: float | None = None

        # multiprocessing
        self._mp_ctx = multiprocessing.get_context()
        self._stop_event: multiprocessing.Event | None = None
        self._video_mp_queue: multiprocessing.Queue | None = None
        self._audio_mp_queue: multiprocessing.Queue | None = None
        self._status_queue: multiprocessing.Queue | None = None
        self._control_queue: multiprocessing.Queue | None = None
        self._write_process: multiprocessing.Process | None = None
        self._video_mp_queue_maxlen = self._video_queue_maxlen
        self._audio_mp_queue_maxlen = min(self._audio_queue_maxlen, 32767)
        self._video_drop_count = max(1, self._video_mp_queue_maxlen // 2)
        self._audio_drop_count = max(1, self._audio_mp_queue_maxlen // 2)

        self._setup_multiprocessing()

        # 自動起動
        self.start()

    def _setup_multiprocessing(self) -> None:
        if self._stop_event is None or self._stop_event.is_set():
            self._stop_event = self._mp_ctx.Event()
        if self._video_mp_queue is None:
            self._video_mp_queue = self._mp_ctx.Queue(
                maxsize=self._video_mp_queue_maxlen
            )
        if self._audio_mp_queue is None:
            self._audio_mp_queue = self._mp_ctx.Queue(
                maxsize=self._audio_mp_queue_maxlen
            )
        if self._status_queue is None:
            self._status_queue = self._mp_ctx.Queue(maxsize=1000)
        if self._control_queue is None:
            self._control_queue = self._mp_ctx.Queue(maxsize=10)

    def _drain_status_queue(self) -> None:
        if self._status_queue is None:
            return
        while True:
            try:
                payload = self._status_queue.get_nowait()
            except queue.Empty:
                break
            status = payload.get("status")
            if status == "write_success":
                last_write_time = payload.get("last_write_time")
                if last_write_time is not None:
                    self._last_successful_write_time = last_write_time
                else:
                    self._last_successful_write_time = time.time()
                kind = payload.get("kind")
                if kind == "video":
                    self._stats_video_frame_count += 1
                    self._worker_has_video_reference = True
                elif kind == "audio":
                    self._stats_audio_frame_count += 1
            elif status == "write_stats":
                video_count = payload.get("video_count", 0)
                audio_count = payload.get("audio_count", 0)
                elapsed = payload.get("elapsed")
                last_write_time = payload.get("last_write_time")
                if last_write_time is not None:
                    self._last_successful_write_time = last_write_time
                elif video_count or audio_count:
                    self._last_successful_write_time = time.time()
                if video_count:
                    self._stats_video_frame_count += video_count
                    self._worker_has_video_reference = True
                if audio_count:
                    self._stats_audio_frame_count += audio_count
                if elapsed:
                    if video_count:
                        self._mp_video_fps = video_count / elapsed
                    if audio_count:
                        self._mp_audio_fps = audio_count / elapsed
            elif status == "started":
                started_time = payload.get("started_time")
                if started_time is not None:
                    self._last_successful_write_time = started_time
                else:
                    self._last_successful_write_time = time.time()
            elif status == "error":
                message = payload.get("message")
                if message:
                    print(f"処理開始エラー: {message}")

    def _request_container_close(self) -> None:
        if self._control_queue is None:
            return
        try:
            self._control_queue.put_nowait({"cmd": "close_container"})
        except Exception:
            pass

    def set_crop_ratio(self, ratio: float | None) -> None:
        """クロップ比率を設定

        Args:
            ratio: クロップ比率（0.0〜1.0）。Noneの場合はクロップしない。
        """
        if ratio is not None and not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be between 0.0 and 1.0, got {ratio}")
        self._crop_ratio = ratio

    def enqueue_video_frame(self, frame: WrappedVideoFrame) -> None:
        """映像フレームをキューに追加（ノンブロッキング）

        Args:
            frame: 送信するWrappedVideoFrame
        """
        try:
            if self.video_queue.full():
                try:
                    drop_count = max(1, self.video_queue.qsize() // 2)
                except NotImplementedError:
                    drop_count = 1
                for _ in range(drop_count):
                    try:
                        self.video_queue.get_nowait()
                    except queue.Empty:
                        break

            self.video_queue.put_nowait(frame)
        except Exception as e:
            print(f"映像フレームをキューに追加中にエラー: {repr(e)}")

        if self._video_mp_queue is None:
            return
        try:
            if not _drop_if_full(self._video_mp_queue, self._video_drop_count):
                return
            include_planes = True
            if frame.is_bad_frame and self._worker_has_video_reference:
                include_planes = False
            payload = _serialize_video_frame(frame, include_planes=include_planes)
            _put_with_drop(
                self._video_mp_queue,
                payload,
                self._video_drop_count,
            )
        except Exception as e:
            print(f"映像フレームをキューに追加中にエラー: {repr(e)}")

    def enqueue_video_frames(self, frames: list[WrappedVideoFrame]) -> None:
        """複数の映像フレームをキューに追加

        Args:
            frames: 送信するWrappedVideoFrameのリスト
        """
        for frame in frames:
            self.enqueue_video_frame(frame)

    def enqueue_video_payload(self, payload: dict[str, Any]) -> None:
        """シリアライズ済みの映像ペイロードを直接MPキューへ追加"""
        if self._video_mp_queue is None:
            _cleanup_payload_shared_memory(payload)
            return
        try:
            if not _drop_if_full(self._video_mp_queue, self._video_drop_count):
                _cleanup_payload_shared_memory(payload)
                return
            _put_with_drop(
                self._video_mp_queue,
                payload,
                self._video_drop_count,
            )
        except Exception as e:
            _cleanup_payload_shared_memory(payload)
            print(f"映像ペイロードをMPキューに追加中にエラー: {repr(e)}")

    def enqueue_audio_frame(self, frame: WrappedAudioFrame) -> None:
        """音声フレームをキューに追加（ノンブロッキング）

        Args:
            frame: 送信するWrappedAudioFrame
        """
        try:
            if self.audio_queue.full():
                try:
                    drop_count = max(1, self.audio_queue.qsize() // 2)
                except NotImplementedError:
                    drop_count = 1
                for _ in range(drop_count):
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break

            self.audio_queue.put_nowait(frame)
        except Exception as e:
            print(f"音声フレームをキューに追加中にエラー: {repr(e)}")

        if self._audio_mp_queue is None:
            return
        try:
            if not _drop_if_full(self._audio_mp_queue, self._audio_drop_count):
                return
            payload = _serialize_audio_frame(frame)
            _put_with_drop(
                self._audio_mp_queue,
                payload,
                self._audio_drop_count,
            )
        except Exception as e:
            print(f"音声フレームをキューに追加中にエラー: {repr(e)}")

    def enqueue_audio_frames(self, frames: list[WrappedAudioFrame]) -> None:
        """複数の音声フレームをキューに追加

        Args:
            frames: 送信するWrappedAudioFrameのリスト
        """
        for frame in frames:
            self.enqueue_audio_frame(frame)

    def enqueue_audio_payload(self, payload: dict[str, Any]) -> None:
        """シリアライズ済みの音声ペイロードを直接MPキューへ追加"""
        if self._audio_mp_queue is None:
            _cleanup_payload_shared_memory(payload)
            return
        try:
            if not _drop_if_full(self._audio_mp_queue, self._audio_drop_count):
                _cleanup_payload_shared_memory(payload)
                return
            _put_with_drop(
                self._audio_mp_queue,
                payload,
                self._audio_drop_count,
            )
        except Exception as e:
            _cleanup_payload_shared_memory(payload)
            print(f"音声ペイロードをMPキューに追加中にエラー: {repr(e)}")

    def start(self) -> None:
        """start_processing() をスレッドで起動するラッパー"""
        self._init_thread = threading.Thread(target=self.start_processing, daemon=True)
        self._init_thread.start()

    def start_processing(self) -> str:
        """コンテナを初期化し、書き込みプロセスを開始"""
        try:
            if self.is_running:
                return "ストリーム処理は既に開始しています"
            self._setup_multiprocessing()
            self.is_running = True
            self._start_write_process()
            status = self._wait_for_start_status()
            if status.startswith("処理開始エラー"):
                self.is_running = False
                return status
            self._clear_local_queues()
            self._monitor_thread = threading.Thread(
                target=self._monitor_write_updates, daemon=True
            )
            self._monitor_thread.start()
            return status
        except Exception as e:
            self.is_running = False
            return f"処理開始エラー: {str(e)}"

    def _wait_for_start_status(self) -> str:
        if self._status_queue is None:
            return "ストリーム処理を開始しました"
        while True:
            if not self.is_running:
                return "処理開始エラー: 停止しました"
            try:
                status = self._status_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if status.get("status") == "started":
                self._last_successful_write_time = time.time()
                return "ストリーム処理を開始しました"
            if status.get("status") == "error":
                message = status.get("message", "")
                return f"処理開始エラー: {message}"

    def _start_write_process(self, width: int | None = None, height: int | None = None) -> None:
        if not self.is_running:
            return
        self._worker_has_video_reference = False
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        self._stop_event = self._mp_ctx.Event()
        if (
            self._video_mp_queue is None
            or self._audio_mp_queue is None
            or self._status_queue is None
            or self._control_queue is None
        ):
            self._setup_multiprocessing()
        if self._write_process and self._write_process.is_alive():
            return
        if self._status_queue is not None:
            try:
                while True:
                    self._status_queue.get_nowait()
            except queue.Empty:
                pass
        if self._control_queue is not None:
            try:
                while True:
                    self._control_queue.get_nowait()
            except queue.Empty:
                pass

        self._write_process = self._mp_ctx.Process(
            target=_stream_writer_worker,
            args=(
                self.url,
                width,
                height,
                self.fps,
                self.sample_rate,
                self.audio_layout,
                self._stats_enabled,
                self._stop_event,
                self._video_mp_queue,
                self._audio_mp_queue,
                self._status_queue,
                self._control_queue,
                self._crop_ratio,
            ),
            daemon=True,
        )
        self._write_process.start()
        self._thread = self._write_process
        self.container = _StreamWriterContainerProxy(self._request_container_close)

    def _stop_write_process(self) -> None:
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        if _is_process_alive(self._write_process):
            self._write_process.join(timeout=5.0)
        if _is_process_alive(self._write_process):
            self._write_process.terminate()
            self._write_process.join(timeout=5.0)
        self._write_process = None
        if self._thread is not None and isinstance(self._thread, multiprocessing.Process):
            self._thread = None

    def _close_mp_queues(self) -> None:
        video_mp_queue = getattr(self, "_video_mp_queue", None)
        if video_mp_queue is not None:
            self._clear_mp_queue(video_mp_queue)
            try:
                video_mp_queue.cancel_join_thread()
                video_mp_queue.close()
            except Exception:
                pass
        audio_mp_queue = getattr(self, "_audio_mp_queue", None)
        if audio_mp_queue is not None:
            self._clear_mp_queue(audio_mp_queue)
            try:
                audio_mp_queue.cancel_join_thread()
                audio_mp_queue.close()
            except Exception:
                pass
        status_queue = getattr(self, "_status_queue", None)
        if status_queue is not None:
            try:
                status_queue.cancel_join_thread()
                status_queue.close()
            except Exception:
                pass
        control_queue = getattr(self, "_control_queue", None)
        if control_queue is not None:
            try:
                control_queue.cancel_join_thread()
                control_queue.close()
            except Exception:
                pass
        self._video_mp_queue = None
        self._audio_mp_queue = None
        self._status_queue = None
        self._control_queue = None

    @property
    def container(self) -> _StreamWriterContainerProxy | None:
        return self._container

    @container.setter
    def container(self, value: _StreamWriterContainerProxy | None) -> None:
        self._container = value
        write_process = getattr(self, "_write_process", None)
        if value is None and write_process is not None:
            self._stop_write_process()

    def _write_frames(self) -> None:
        """エンコード・送信ループ (StreamListenerの_read_framesに対応)"""
        try:
            if self._first_frame is None:
                return

            # クロップ設定がある場合は適用
            first_frame = self._first_frame
            if self._crop_ratio is not None:
                first_frame = first_frame.crop_center(self._crop_ratio)

            # 最初のフレームを処理（元のPTSをそのまま使用）
            for packet in self._video_stream.encode(first_frame.frame):
                self.container.mux(packet)
            self._last_successful_write_time = time.time()
            self._stats_video_frame_count += 1

            while self.is_running:
                if self.container is None:
                    break

                has_data = False

                # 映像フレームの処理
                wrapped_frame = self._process_video_frame()
                if wrapped_frame is not None:
                    # クロップ設定がある場合は適用
                    if self._crop_ratio is not None:
                        wrapped_frame = wrapped_frame.crop_center(self._crop_ratio)

                    has_data = True
                    # 元のPTSとtime_baseをそのまま使用
                    try:
                        for packet in self._video_stream.encode(wrapped_frame.frame):
                            self.container.mux(packet)
                        self._last_successful_write_time = time.time()
                        self._stats_video_frame_count += 1
                    except Exception as e:
                        print(
                            f"StreamWriter映像muxエラー（スキップ）: {e}",
                            file=sys.stderr,
                        )

                # 音声フレームの処理（キューにある全フレームを処理）
                while True:
                    wrapped_audio = self._get_audio_frame()
                    if wrapped_audio is None:
                        break
                    has_data = True
                    # 元のPTSとtime_baseをそのまま使用
                    try:
                        for packet in self._audio_stream.encode(wrapped_audio.frame):
                            self.container.mux(packet)
                        self._last_successful_write_time = time.time()
                        self._stats_audio_frame_count += 1
                    except Exception as e:
                        print(
                            f"StreamWriter音声muxエラー（スキップ）: {e}",
                            file=sys.stderr,
                        )

                if not has_data:
                    time.sleep(0.001)

        except Exception as e:
            print(f"StreamWriter書き込みエラー: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            # フラッシュ & クローズ
            if self.container:
                try:
                    for packet in self._video_stream.encode():
                        self.container.mux(packet)
                    for packet in self._audio_stream.encode():
                        self.container.mux(packet)
                    self.container.close()
                except Exception as e:
                    print(f"StreamWriterクローズ時にエラー: {e}", file=sys.stderr)
                    traceback.print_exc()

    def _monitor_write_updates(self) -> None:
        """書き込み更新を監視し、閾値超過時に再接続を試みる"""
        while self.is_running:
            self._drain_status_queue()

            if not self.is_running:
                break

            write_process = getattr(self, "_write_process", None)
            if write_process is not None:
                if self.container is None:
                    self._stop_write_process()
                    time.sleep(0.1)
                    continue
                if write_process and not _is_process_alive(write_process):
                    self._force_restart()

            if self._last_successful_write_time > 0.0:
                elapsed = time.time() - self._last_successful_write_time
                if elapsed > self._restart_threshold:
                    print(
                        f"書き込み更新が{elapsed:.1f}秒間停止。再接続を試みます "
                        f"(閾値: {self._restart_threshold:.1f}秒)"
                    )
                    self._force_restart()

            # FPS統計出力
            if self._stats_enabled:
                now = time.monotonic()
                stats_elapsed = now - self._stats_last_time
                if stats_elapsed >= 5.0:
                    video_fps = self._stats_video_frame_count / stats_elapsed
                    audio_fps = self._stats_audio_frame_count / stats_elapsed
                    if self._mp_video_fps is not None or self._mp_audio_fps is not None:
                        mp_video_fps = (
                            f"{self._mp_video_fps:.2f}"
                            if self._mp_video_fps is not None
                            else "N/A"
                        )
                        mp_audio_fps = (
                            f"{self._mp_audio_fps:.2f}"
                            if self._mp_audio_fps is not None
                            else "N/A"
                        )
                        print(
                            "[Writer] video_fps="
                            f"{video_fps:.2f} audio_fps={audio_fps:.2f} "
                            f"mp_video_fps={mp_video_fps} mp_audio_fps={mp_audio_fps}"
                        )
                    else:
                        print(
                            f"[Writer] video_fps={video_fps:.2f} audio_fps={audio_fps:.2f}"
                        )
                    self._stats_video_frame_count = 0
                    self._stats_audio_frame_count = 0
                    self._stats_last_time = now

            time.sleep(1.0)

    def _force_restart(self) -> None:
        with self._restart_lock:
            if not self.is_running or self._restart_in_progress:
                return
            self._restart_in_progress = True
            self._restart_threshold += self._restart_threshold_increment
            if self._restart_threshold >= self._restart_threshold_max:
                print(
                    f"再接続閾値が上限({self._restart_threshold_max}秒)に達しました。"
                    "プロセスを終了します。"
                )
                os._exit(1)
        try:
            self._restart_connection()
        finally:
            with self._restart_lock:
                self._restart_in_progress = False

    def _interruptible_sleep(self, seconds: float) -> None:
        """is_runningチェック付きスリープ。0.5秒間隔で分割。"""
        remaining = seconds
        while remaining > 0 and self.is_running:
            sleep_time = min(remaining, 0.5)
            time.sleep(sleep_time)
            remaining -= sleep_time

    def _clear_mp_queue(self, target: multiprocessing.Queue | None) -> None:
        if target is None:
            return
        reader = getattr(target, "_reader", None)
        while True:
            try:
                if reader is not None and not reader.poll():
                    break
                dropped = target.get_nowait()
                _cleanup_payload_shared_memory(dropped)
            except queue.Empty:
                break
            except (ValueError, OSError):
                break

    def _clear_local_queues(self) -> None:
        while True:
            try:
                self.video_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _wait_for_first_frame(self, timeout: float) -> WrappedVideoFrame | None:
        """タイムアウト付きで最初のフレームをキューから取得。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running:
                return None
            try:
                return self.video_queue.get(timeout=1.0)
            except queue.Empty:
                continue
        return None

    def _restart_connection(self) -> bool:
        """ストリーム接続を再確立する。サブクラスでオーバーライド可能。

        Returns:
            bool: 再接続に成功した場合True、失敗した場合False
        """
        if not self.is_running:
            return False

        # 古いcontainerをNoneに設定後close
        old_container = self.container
        self.container = None
        if old_container:
            try:
                old_container.close()
            except Exception:
                pass

        # 古い書込プロセス/スレッドを待機
        if self._write_process is not None:
            self._stop_write_process()
        elif _is_thread_alive(self._thread):
            self._thread.join(timeout=5.0)
            self._thread = None

        if not self.is_running:
            return False

        # 再接続待機
        self._interruptible_sleep(self._restart_wait_seconds)

        if not self.is_running:
            return False

        self._clear_mp_queue(self._video_mp_queue)
        self._clear_mp_queue(self._audio_mp_queue)
        self._clear_local_queues()

        # キューから次のフレームを取得して_first_frameに設定（タイムアウト付き）
        first_frame = self._wait_for_first_frame(self._restart_frame_wait_timeout)
        if first_frame is None:
            return False

        self._first_frame = first_frame

        # フレームから実際の解像度を取得
        actual_width = first_frame.frame.width
        actual_height = first_frame.frame.height

        self._last_successful_write_time = time.time()
        self._start_write_process(width=actual_width, height=actual_height)
        print("StreamWriter再接続に成功しました")
        return True

    def stop(self) -> None:
        """ストリーム処理を停止"""
        if getattr(self, "_owner_pid", os.getpid()) != os.getpid():
            return
        self.is_running = False
        init_thread = getattr(self, "_init_thread", None)
        if _is_thread_alive(init_thread):
            init_thread.join(timeout=5.0)
        self._init_thread = None

        monitor_thread = getattr(self, "_monitor_thread", None)
        if _is_thread_alive(monitor_thread):
            monitor_thread.join(timeout=10.0)
        self._monitor_thread = None

        write_process = getattr(self, "_write_process", None)
        if write_process is not None:
            self._stop_write_process()
        else:
            thread = getattr(self, "_thread", None)
            if _is_thread_alive(thread):
                thread.join(timeout=5.0)
            self._thread = None

        self._close_mp_queues()

    def __del__(self) -> None:
        """リソース解放"""
        try:
            if getattr(self, "_owner_pid", os.getpid()) != os.getpid():
                return
            self.stop()
        except Exception:
            pass

    def _process_video_frame(self) -> WrappedVideoFrame | None:
        """映像フレームを取得

        is_bad_frameがTrueの場合は、最後に送信したフレームの生データで差し替える。

        Returns:
            WrappedVideoFrame、またはNone
        """
        try:
            wrapped_frame: WrappedVideoFrame = self.video_queue.get(timeout=1 / 30)
        except queue.Empty:
            return None

        if wrapped_frame.is_bad_frame:
            # 最後に送信したフレームの生データで差し替え
            if self._last_video_frame is not None:
                last_planes = self._last_video_frame.get_planes()
                wrapped_frame.set_planes(last_planes)
        else:
            # 生データがある場合は保存
            self._last_video_frame = wrapped_frame

        return wrapped_frame

    def _get_audio_frame(self) -> WrappedAudioFrame | None:
        """音声フレームをキューから取得

        Returns:
            WrappedAudioFrame、またはNone
        """
        try:
            wrapped_frame: WrappedAudioFrame = self.audio_queue.get_nowait()
        except queue.Empty:
            return None

        return wrapped_frame
