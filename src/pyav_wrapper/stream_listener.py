from __future__ import annotations

import collections
import fractions
import multiprocessing
import pickle
import queue
import threading
import time
from multiprocessing import shared_memory
from typing import Any

import av
import numpy as np

from pyav_wrapper.audio_frame import WrappedAudioFrame
from pyav_wrapper.video_frame import WrappedVideoFrame


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
    if _is_picklable(opaque):
        payload["opaque"] = opaque
    return payload


def _is_picklable(value: Any) -> bool:
    try:
        pickle.dumps(value)
        return True
    except Exception:
        return False


_SHM_VIDEO_THRESHOLD_BYTES = 1 << 60
_SHM_AUDIO_THRESHOLD_BYTES = 1 << 60


def _cleanup_payload_shared_memory(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("storage") != "shm":
        return
    shm_name = payload.get("shm_name")
    if not isinstance(shm_name, str) or not shm_name:
        return
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return
    except Exception:
        return
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


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

    spans: list[tuple[int, int]] = []
    cursor = 0
    try:
        for chunk in chunks:
            chunk_size = len(chunk)
            shm.buf[cursor: cursor + chunk_size] = chunk
            spans.append((cursor, chunk_size))
            cursor += chunk_size
        shm_name = shm.name
    finally:
        try:
            shm.close()
        except Exception:
            pass

    return shm_name, spans


def _serialize_video_frame(frame: av.VideoFrame) -> dict[str, Any]:
    planes: list[bytes] | None = []
    for plane in frame.planes:
        planes.append(bytes(plane))

    storage = "inline"
    shm_name: str | None = None
    plane_spans: list[tuple[int, int]] | None = None
    shm_payload = _pack_bytes_to_shared_memory(planes, _SHM_VIDEO_THRESHOLD_BYTES)
    if shm_payload is not None:
        shm_name, plane_spans = shm_payload
        planes = None
        storage = "shm"

    payload = _serialize_frame_common(frame)
    payload.update(
        {
            "format": frame.format.name,
            "width": frame.width,
            "height": frame.height,
            "planes": planes,
            "plane_spans": plane_spans,
            "storage": storage,
            "shm_name": shm_name,
            "pict_type": frame.pict_type,
            "colorspace": frame.colorspace,
            "color_range": frame.color_range,
        }
    )
    return payload


def _serialize_audio_frame(frame: av.AudioFrame) -> dict[str, Any]:
    data = np.ascontiguousarray(frame.to_ndarray())
    raw_bytes = data.tobytes()
    shm_payload = _pack_bytes_to_shared_memory([raw_bytes], _SHM_AUDIO_THRESHOLD_BYTES)

    payload = _serialize_frame_common(frame)
    if shm_payload is None:
        payload.update(
            {
                "format": frame.format.name,
                "layout": frame.layout.name,
                "sample_rate": frame.sample_rate,
                "rate": frame.rate,
                "samples": frame.samples,
                "data": data,
                "storage": "inline",
                "shm_name": None,
                "data_spans": None,
            }
        )
        return payload

    shm_name, data_spans = shm_payload
    payload.update(
        {
            "format": frame.format.name,
            "layout": frame.layout.name,
            "sample_rate": frame.sample_rate,
            "rate": frame.rate,
            "samples": frame.samples,
            "data": None,
            "storage": "shm",
            "shm_name": shm_name,
            "data_spans": data_spans,
            "data_shape": data.shape,
            "data_dtype": str(data.dtype),
        }
    )
    return payload


def _put_with_drop(
    target_queue: multiprocessing.Queue,
    item: dict[str, Any],
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


def _stream_listener_decode_worker(
    url: str,
    width: int,
    height: int,
    stop_event: multiprocessing.Event,
    video_queue: multiprocessing.Queue,
    audio_queue: multiprocessing.Queue,
    video_drop_count: int,
    audio_drop_count: int,
    crop_ratio: float | None = None,
) -> None:
    container: av.Container | None = None
    try:
        container = av.open(url)

        video_stream = (
            container.streams.video[0]
            if container.streams.video
            else None
        )
        audio_stream = (
            container.streams.audio[0]
            if container.streams.audio
            else None
        )

        streams_to_decode = [
            s for s in [video_stream, audio_stream] if s is not None
        ]

        for frame in container.decode(*streams_to_decode):
            if stop_event.is_set():
                break

            if isinstance(frame, av.VideoFrame):
                if width is not None and height is not None:
                    if frame.width != width or frame.height != height:
                        frame = frame.reformat(width=width, height=height)
                if crop_ratio is not None:
                    frame = WrappedVideoFrame(frame).crop_center(crop_ratio).frame
                payload = _serialize_video_frame(frame)
                _put_with_drop(video_queue, payload, video_drop_count)

            elif isinstance(frame, av.AudioFrame):
                payload = _serialize_audio_frame(frame)
                _put_with_drop(audio_queue, payload, audio_drop_count)

    except Exception as e:
        print(f"フレーム読み込みエラー: {str(e)}")
    finally:
        if container:
            try:
                container.close()
            except Exception:
                pass


class _StreamListenerContainerProxy:
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


class StreamListener:
    """PyAVでストリームを受信し、Video/Audioフレームをバッファリングするクラス"""

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        fps: int = 30,
        sample_rate: int = 48000,
        audio_layout: str = "stereo",
        stats_enabled: bool = False,
        crop_ratio: float | None = None,
    ):
        """
        Args:
            url: 任意のストリームURL（srt://, rtmp://, udp://等）
            width: 出力幅（リサイズ用）
            height: 出力高さ（リサイズ用）
            fps: 想定フレームレート（保持用）
            sample_rate: 想定音声サンプルレート（保持用）
            audio_layout: 想定音声チャンネルレイアウト（保持用）
            stats_enabled: FPS統計出力を有効にするかどうか
            crop_ratio: 受信時に適用するクロップ比率（0.0〜1.0）
        """
        if width is None or height is None:
            raise ValueError("widthとheightは必須です")
        if crop_ratio is not None and not (0.0 < crop_ratio <= 1.0):
            raise ValueError(f"crop_ratio must be between 0.0 and 1.0, got {crop_ratio}")
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        self.audio_layout = audio_layout
        self.is_running = False
        self.container: _StreamListenerContainerProxy | None = None
        self._read_thread: multiprocessing.Process | None = None

        # Video
        self.batch_size = fps
        video_queue_maxlen = int(self.batch_size * 1.7)
        # 高解像度時はキュー上限を抑え、メモリ増大によるプロセス終了を防ぐ
        if self.width * self.height >= 1280 * 720:
            video_queue_maxlen = min(video_queue_maxlen, self.batch_size)
        self.video_queue: collections.deque[WrappedVideoFrame] = collections.deque(
            maxlen=video_queue_maxlen
        )
        self.video_queue_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.current_frame: WrappedVideoFrame | None = None

        # Audio
        self.batch_size_audio = self.sample_rate
        self._audio_batch_samples = self.sample_rate
        self._audio_queue_max_samples = int(self.sample_rate * 1.7)
        self._audio_queue_samples = 0
        self.audio_queue: collections.deque[WrappedAudioFrame] = collections.deque()
        self.audio_queue_lock = threading.Lock()

        # multiprocessing
        self._mp_ctx = multiprocessing.get_context()
        self._stop_event: multiprocessing.Event | None = None
        self._video_mp_queue: multiprocessing.Queue | None = None
        self._audio_mp_queue: multiprocessing.Queue | None = None
        self._video_mp_queue_maxlen = video_queue_maxlen
        self._audio_mp_queue_maxlen = video_queue_maxlen
        self._video_drop_count = max(1, self._video_mp_queue_maxlen // 10)
        self._audio_drop_count = max(1, self._audio_mp_queue_maxlen // 10)

        # Reconnection
        self._last_successful_read_time: float = time.time()
        self._restart_threshold: float = 10.0
        self._restart_threshold_increment: float = 1.0
        self._restart_threshold_max: float = 20.0
        self._restart_wait_seconds: float = 5.0
        self._restart_lock: threading.Lock = threading.Lock()
        self._restart_requested = False
        self._restart_in_progress = False
        self._monitor_thread: threading.Thread | None = None

        # FPS統計
        self._stats_enabled = stats_enabled
        self._stats_video_frame_count: int = 0
        self._stats_audio_frame_count: int = 0
        self._stats_last_time: float = time.monotonic()
        self._crop_ratio: float | None = crop_ratio

        self.start_processing()

    def start_processing(self) -> str:
        """ストリーム処理を開始"""
        try:
            self._setup_multiprocessing()
            self.container = _StreamListenerContainerProxy(self._request_restart)
            self.is_running = True
            self._last_successful_read_time = time.time()
            self._start_read_process()
            self._monitor_thread = threading.Thread(
                target=self._drain_and_monitor, daemon=True
            )
            self._monitor_thread.start()
            return "ストリーム処理を開始しました"
        except Exception as e:
            return f"処理開始エラー: {str(e)}"

    def _setup_multiprocessing(self) -> None:
        self._stop_event = self._mp_ctx.Event()
        self._video_mp_queue = self._mp_ctx.Queue(maxsize=self._video_mp_queue_maxlen)
        self._audio_mp_queue = self._mp_ctx.Queue(maxsize=self._audio_mp_queue_maxlen)

    def _start_read_process(self) -> None:
        if self._stop_event is None:
            self._stop_event = self._mp_ctx.Event()
        self._read_thread = self._mp_ctx.Process(
            target=_stream_listener_decode_worker,
            args=(
                self.url,
                self.width,
                self.height,
                self._stop_event,
                self._video_mp_queue,
                self._audio_mp_queue,
                self._video_drop_count,
                self._audio_drop_count,
                self._crop_ratio,
            ),
            daemon=True,
        )
        self._read_thread.start()

    def set_crop_ratio(self, ratio: float | None) -> None:
        """受信時に適用するクロップ比率を設定

        Args:
            ratio: クロップ比率（0.0〜1.0）。Noneの場合はクロップしない。
        """
        if ratio is not None and not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be between 0.0 and 1.0, got {ratio}")
        if ratio == self._crop_ratio:
            return
        self._crop_ratio = ratio
        if self.is_running:
            self._request_restart()

    def _stop_read_process(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=5.0)
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.terminate()
            self._read_thread.join(timeout=5.0)
        self._read_thread = None

    def _close_mp_queues(self) -> None:
        if self._video_mp_queue is not None:
            try:
                self._video_mp_queue.close()
                self._video_mp_queue.join_thread()
            except Exception:
                pass
        if self._audio_mp_queue is not None:
            try:
                self._audio_mp_queue.close()
                self._audio_mp_queue.join_thread()
            except Exception:
                pass
        self._video_mp_queue = None
        self._audio_mp_queue = None

    def _deserialize_video_frame(self, payload: dict[str, Any]) -> WrappedVideoFrame:
        frame = av.VideoFrame(
            payload["width"],
            payload["height"],
            payload["format"],
        )
        self._apply_common_frame_attrs(frame, payload)
        self._apply_video_frame_attrs(frame, payload)
        wrapped = WrappedVideoFrame(frame)
        storage = payload.get("storage", "inline")
        if storage == "shm":
            shm_name = payload.get("shm_name")
            plane_spans = payload.get("plane_spans")
            if isinstance(shm_name, str) and isinstance(plane_spans, list):
                shm = None
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
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
                        try:
                            shm.close()
                        except Exception:
                            pass
                        try:
                            shm.unlink()
                        except Exception:
                            pass
                    else:
                        _cleanup_payload_shared_memory(payload)
        else:
            planes = payload.get("planes")
            if planes is not None:
                for i, plane_data in enumerate(planes):
                    try:
                        if isinstance(plane_data, np.ndarray):
                            frame.planes[i].update(plane_data.tobytes())
                        else:
                            frame.planes[i].update(plane_data)
                    except Exception:
                        continue
        return wrapped

    def _deserialize_audio_frame(self, payload: dict[str, Any]) -> WrappedAudioFrame:
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
                        shm = shared_memory.SharedMemory(name=shm_name)
                        raw = bytes(shm.buf[offset: offset + size])
                        data = np.frombuffer(raw, dtype=np.dtype(data_dtype)).reshape(tuple(data_shape))
                        frame = av.AudioFrame.from_ndarray(
                            data,
                            format=payload["format"],
                            layout=payload["layout"],
                        )
                        self._apply_common_frame_attrs(frame, payload)
                        self._apply_audio_frame_attrs(frame, payload)
                        wrapped = WrappedAudioFrame(frame)
                        return wrapped
                    finally:
                        if shm is not None:
                            try:
                                shm.close()
                            except Exception:
                                pass
                            try:
                                shm.unlink()
                            except Exception:
                                pass
                        else:
                            _cleanup_payload_shared_memory(payload)
        frame = av.AudioFrame.from_ndarray(
            payload["data"],
            format=payload["format"],
            layout=payload["layout"],
        )
        self._apply_common_frame_attrs(frame, payload)
        self._apply_audio_frame_attrs(frame, payload)
        return WrappedAudioFrame(frame)

    def _apply_common_frame_attrs(self, frame: "av.frame.Frame", payload: dict[str, Any]) -> None:
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

    def _apply_video_frame_attrs(self, frame: av.VideoFrame, payload: dict[str, Any]) -> None:
        for key in ("pict_type", "colorspace", "color_range"):
            if key in payload:
                try:
                    setattr(frame, key, payload[key])
                except Exception:
                    pass

    def _apply_audio_frame_attrs(self, frame: av.AudioFrame, payload: dict[str, Any]) -> None:
        for key in ("sample_rate", "rate", "samples", "format", "layout"):
            if key in payload:
                try:
                    setattr(frame, key, payload[key])
                except Exception:
                    pass

    def _drain_and_monitor(self) -> None:
        """プロセスから届くフレームを取り込み、監視も行う"""
        while self.is_running:
            if self._restart_requested:
                self._restart_requested = False
                self._force_restart()
                continue

            had_data = False
            had_data |= self._drain_video_queue()
            had_data |= self._drain_audio_queue()

            if not had_data:
                time.sleep(0.1)

            if not self.is_running:
                break

            if self._read_thread and not self._read_thread.is_alive():
                self._force_restart()
                continue

            elapsed = time.time() - self._last_successful_read_time
            if elapsed > self._restart_threshold:
                print(
                    f"フレーム更新が{elapsed:.1f}秒間停止。再接続を試みます "
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
                    print(f"[Listener] video_fps={video_fps:.2f} audio_fps={audio_fps:.2f}")
                    self._stats_video_frame_count = 0
                    self._stats_audio_frame_count = 0
                    self._stats_last_time = now

    def _monitor_frame_updates(self) -> None:
        """互換性のためのラッパー。旧メソッド名で監視ループを起動する。"""
        self._drain_and_monitor()

    def _drain_video_queue(self) -> bool:
        if self._video_mp_queue is None:
            return False
        drained = False
        while True:
            try:
                payload = self._video_mp_queue.get_nowait()
            except queue.Empty:
                break
            except (ValueError, OSError):
                break

            wrapped = self._deserialize_video_frame(payload)
            with self.frame_lock:
                self.current_frame = wrapped
            self.append_video_queue(wrapped)
            self._last_successful_read_time = time.time()
            self._stats_video_frame_count += 1
            drained = True
        return drained

    def _drain_audio_queue(self) -> bool:
        if self._audio_mp_queue is None:
            return False
        drained = False
        while True:
            try:
                payload = self._audio_mp_queue.get_nowait()
            except queue.Empty:
                break
            except (ValueError, OSError):
                break

            wrapped = self._deserialize_audio_frame(payload)
            self.append_audio_queue(wrapped)
            self._last_successful_read_time = time.time()
            self._stats_audio_frame_count += 1
            drained = True
        return drained

    def append_video_queue(self, frame: WrappedVideoFrame) -> None:
        """Videoキューにフレームを追加"""
        if frame is None:
            return
        with self.video_queue_lock:
            try:
                if len(self.video_queue) >= self.video_queue.maxlen:
                    for _ in range(max(1, len(self.video_queue) // 10)):
                        self.video_queue.pop()
                self.video_queue.append(frame)
            except Exception as e:
                print(f"Videoキュー追加エラー: {repr(e)}")

    def pop_all_video_queue(self) -> list[WrappedVideoFrame]:
        """Videoキューからバッチサイズ分のフレームを取り出す"""
        with self.video_queue_lock:
            if len(self.video_queue) < self.batch_size:
                return []
            return_frames: list[WrappedVideoFrame] = []
            for _ in range(self.batch_size):
                return_frames.append(self.video_queue.popleft())
            return return_frames

    def _get_audio_frame_samples(self, frame: WrappedAudioFrame) -> int:
        """AudioFrameのサンプル数を正規化して取得"""
        samples = frame.frame.samples
        if samples is None or samples <= 0:
            return 0
        return samples

    def append_audio_queue(self, frame: WrappedAudioFrame) -> None:
        """Audioキューにフレームを追加"""
        if frame is None:
            return
        with self.audio_queue_lock:
            try:
                samples = self._get_audio_frame_samples(frame)
                while (
                    self._audio_queue_samples + samples > self._audio_queue_max_samples
                    and self.audio_queue
                ):
                    dropped_frame = self.audio_queue.popleft()
                    dropped_samples = self._get_audio_frame_samples(dropped_frame)
                    self._audio_queue_samples = max(
                        0,
                        self._audio_queue_samples - dropped_samples,
                    )
                self.audio_queue.append(frame)
                self._audio_queue_samples += samples
            except Exception as e:
                print(f"Audioキュー追加エラー: {repr(e)}")

    def pop_all_audio_queue(self) -> list[WrappedAudioFrame]:
        """Audioキューからバッチサイズ分のフレームを取り出す"""
        with self.audio_queue_lock:
            if self._audio_queue_samples < self._audio_batch_samples:
                return []
            return_frames: list[WrappedAudioFrame] = []
            collected_samples = 0
            while self.audio_queue and collected_samples < self._audio_batch_samples:
                frame = self.audio_queue.popleft()
                return_frames.append(frame)
                samples = self._get_audio_frame_samples(frame)
                collected_samples += samples
                self._audio_queue_samples = max(
                    0,
                    self._audio_queue_samples - samples,
                )
            return return_frames

    def stop(self) -> None:
        """ストリーム処理を停止"""
        self.is_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        self._monitor_thread = None

        self._stop_read_process()
        self._close_mp_queues()

        with self._restart_lock:
            if self.container is not None:
                try:
                    self.container.close()
                except Exception:
                    pass
                self.container = None

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

    def _request_restart(self) -> None:
        if not self.is_running:
            return
        self._restart_requested = True
        self._last_successful_read_time = 0.0

    def _interruptible_sleep(self, seconds: float) -> None:
        """is_runningチェック付きスリープ。0.5秒間隔で分割。"""
        remaining = seconds
        while remaining > 0 and self.is_running:
            sleep_time = min(remaining, 0.5)
            time.sleep(sleep_time)
            remaining -= sleep_time

    def _clear_local_queues(self) -> None:
        with self.video_queue_lock:
            self.video_queue.clear()
        with self.audio_queue_lock:
            self.audio_queue.clear()
            self._audio_queue_samples = 0
        with self.frame_lock:
            self.current_frame = None

    def _restart_connection(self) -> None:
        """ストリーム接続を再確立する。サブクラスでオーバーライド可能。"""
        if not self.is_running:
            return

        self._stop_read_process()
        self._close_mp_queues()
        self._interruptible_sleep(self._restart_wait_seconds)

        if not self.is_running:
            return

        self._clear_local_queues()
        self._setup_multiprocessing()
        self._start_read_process()
        self.container = _StreamListenerContainerProxy(self._request_restart)
        self._last_successful_read_time = time.time()
        print("再接続に成功しました")
