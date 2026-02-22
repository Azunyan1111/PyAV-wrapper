from __future__ import annotations

import collections
import fractions
import multiprocessing
import os
import pickle
import queue
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Callable

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
    if opaque is not None and _is_picklable(opaque):
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
    payload = _serialize_frame_common(frame)
    planes: list[bytes] | None = []
    for plane in frame.planes:
        planes.append(bytes(plane))

    storage = "inline"
    shm_name: str | None = None
    plane_spans: list[tuple[int, int]] | None = None
    if planes is not None and _SHM_AUDIO_THRESHOLD_BYTES > 0:
        total_size = sum(len(plane) for plane in planes)
        if total_size >= _SHM_AUDIO_THRESHOLD_BYTES:
            shm_payload = _pack_bytes_to_shared_memory(planes, _SHM_AUDIO_THRESHOLD_BYTES)
            if shm_payload is not None:
                shm_name, plane_spans = shm_payload
                planes = None
                storage = "shm"

    payload.update(
        {
            "payload_type": "audio_frame",
            "format": frame.format.name,
            "layout": frame.layout.name,
            "sample_rate": frame.sample_rate,
            "rate": frame.rate,
            "samples": frame.samples,
            "planes": planes,
            "plane_spans": plane_spans,
            "storage": storage,
            "shm_name": shm_name,
            "data": None,
            "data_spans": None,
        }
    )
    return payload


def _serialize_audio_packet(packet: av.Packet, stream: Any) -> dict[str, Any]:
    codec_name = ""
    sample_rate: int | None = None
    layout_name: str | None = None
    try:
        codec_ctx = stream.codec_context
        codec_name = str(codec_ctx.name).lower()
        sample_rate = codec_ctx.sample_rate
        layout = codec_ctx.layout
        if layout is not None:
            layout_name = layout.name
    except Exception:
        pass

    time_base = packet.time_base
    if time_base is None:
        try:
            time_base = stream.time_base
        except Exception:
            time_base = None
    serialized_time_base: tuple[int, int] | None = None
    if time_base is not None:
        serialized_time_base = (time_base.numerator, time_base.denominator)

    return {
        "payload_type": "audio_packet",
        "codec_name": codec_name,
        "sample_rate": sample_rate,
        "layout": layout_name,
        "packet_bytes": bytes(packet),
        "pts": packet.pts,
        "dts": packet.dts,
        "duration": packet.duration,
        "time_base": serialized_time_base,
    }


def _put_with_drop(
    target_queue: multiprocessing.Queue,
    item: Any,
    drop_count: int,
) -> None:
    try:
        target_queue.put_nowait(item)
        return
    except queue.Full:
        print(f"[Listener] mp queue full, dropping {drop_count} frames")
        for _ in range(drop_count):
            try:
                dropped = target_queue.get_nowait()
                _cleanup_payload_shared_memory(dropped)
            except queue.Empty:
                break
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            print("[Listener] mp queue still full after drop, discarding new frame")
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
    audio_packet_passthrough_enabled: bool,
    audio_packet_codec_name: str,
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

        streams_to_read = [
            s for s in [video_stream, audio_stream] if s is not None
        ]

        for packet in container.demux(*streams_to_read):
            if stop_event.is_set():
                break

            if packet.dts is None:
                continue

            if packet.stream.type == "video":
                for frame in packet.decode():
                    if width is not None and height is not None:
                        if frame.width != width or frame.height != height:
                            frame = frame.reformat(width=width, height=height)
                    payload = _serialize_video_frame(frame)
                    _put_with_drop(video_queue, payload, video_drop_count)

            elif packet.stream.type == "audio":
                if audio_packet_passthrough_enabled:
                    codec_name = ""
                    try:
                        codec_name = str(packet.stream.codec_context.name).lower()
                    except Exception:
                        pass
                    if codec_name != audio_packet_codec_name:
                        raise RuntimeError(
                            "Opusパケット転送モードではopus音声のみを扱えます: "
                            f"detected={codec_name or 'unknown'}"
                        )
                    payload = _serialize_audio_packet(packet, packet.stream)
                    _put_with_drop(audio_queue, payload, audio_drop_count)
                else:
                    for frame in packet.decode():
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
        batch_size: int = 30,
        sample_rate: int = 48000,
        audio_layout: str = "stereo",
        stats_enabled: bool = False,
    ):
        """
        Args:
            url: 任意のストリームURL（srt://, rtmp://, udp://等）
            width: 出力幅（リサイズ用）
            height: 出力高さ（リサイズ用）
            batch_size: 想定フレームレート（保持用）
            sample_rate: 想定音声サンプルレート（保持用）
            audio_layout: 想定音声チャンネルレイアウト（保持用）
            stats_enabled: FPS統計出力を有効にするかどうか
        """
        if width is None or height is None:
            raise ValueError("widthとheightは必須です")
        self.url = url
        self.width = width
        self.height = height
        self.sample_rate = sample_rate
        self.audio_layout = audio_layout
        self.is_running = False
        self.container: _StreamListenerContainerProxy | None = None
        self._read_thread: multiprocessing.Process | None = None

        # Video
        self.batch_size = batch_size
        video_queue_maxlen = int(self.batch_size * 1.7)
        # 高解像度時はキュー上限を抑え、メモリ増大によるプロセス終了を防ぐ
        # if self.width * self.height >= 1280 * 720:
        #     video_queue_maxlen = min(video_queue_maxlen, self.batch_size)
        self.video_queue: collections.deque[WrappedVideoFrame] = collections.deque(
            maxlen=video_queue_maxlen
        )
        self.video_queue_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.current_frame: WrappedVideoFrame | None = None
        self._video_payload_forwarder: Callable[[dict[str, Any]], None] | None = None
        self._video_forward_only = False

        # Audio
        self.batch_size_audio = self.sample_rate
        self._audio_batch_samples = self.sample_rate
        self._audio_queue_max_samples = int(self.sample_rate * 1.7)
        self._audio_queue_samples = 0
        self.audio_queue: collections.deque[WrappedAudioFrame] = collections.deque()
        self.audio_queue_lock = threading.Lock()
        self._audio_payload_forwarder: Callable[[dict[str, Any]], None] | None = None
        self._audio_forward_only = False
        self._audio_packet_payload_forwarder: Callable[[dict[str, Any]], None] | None = None
        self._audio_packet_forward_only = False
        self._audio_packet_passthrough_enabled = False
        self._audio_packet_codec_name = "opus"

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
        if (
            self._stop_event is None
            or self._stop_event.is_set()
            or self._video_mp_queue is None
            or self._audio_mp_queue is None
        ):
            self._setup_multiprocessing()
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
                self._audio_packet_passthrough_enabled,
                self._audio_packet_codec_name,
            ),
            daemon=True,
        )
        self._read_thread.start()

    def set_batch_size(self, batch_size: int) -> None:
        """映像フレームの取り出しバッチサイズを設定する

        Args:
            batch_size: 一度に取り出すフレーム数（1以上の整数）
        """
        if batch_size < 1:
            raise ValueError(f"batch_sizeは1以上である必要があります: {batch_size}")
        self.batch_size = batch_size

    def set_audio_payload_forwarder(
        self,
        forwarder: Callable[[dict[str, Any]], None] | None,
        forward_only: bool = False,
    ) -> None:
        """音声payloadを受信時に直接転送するforwarderを設定する

        Args:
            forwarder: payload(dict)を受け取るコールバック。Noneで無効化。
            forward_only: Trueの場合、転送成功時はローカルaudio_queueへ積まない。
        """
        self._audio_payload_forwarder = forwarder
        self._audio_forward_only = bool(forwarder is not None and forward_only)
        if forwarder is not None and getattr(self, "_audio_packet_passthrough_enabled", False):
            self.set_audio_packet_payload_forwarder(
                None,
                codec_name=getattr(self, "_audio_packet_codec_name", "opus"),
                forward_only=True,
            )

    def set_audio_packet_payload_forwarder(
        self,
        forwarder: Callable[[dict[str, Any]], None] | None,
        codec_name: str = "opus",
        forward_only: bool = True,
    ) -> None:
        """音声packet payloadを受信時に直接転送するforwarderを設定する

        Args:
            forwarder: packet payload(dict)を受け取るコールバック。Noneで無効化。
            codec_name: passthrough対象コーデック名。現状はopusのみ対応。
            forward_only: True以外はサポートしない。
        """
        normalized_codec_name = str(codec_name).lower()
        if normalized_codec_name != "opus":
            raise ValueError(f"codec_nameはopusのみ対応しています: {codec_name}")
        if forwarder is not None and not forward_only:
            raise ValueError("audio packet passthroughではforward_only=Trueのみ対応しています")

        mode_changed = (
            getattr(self, "_audio_packet_passthrough_enabled", False) != (forwarder is not None)
            or getattr(self, "_audio_packet_codec_name", "opus") != normalized_codec_name
        )
        self._audio_packet_payload_forwarder = forwarder
        self._audio_packet_forward_only = bool(forwarder is not None and forward_only)
        self._audio_packet_passthrough_enabled = bool(forwarder is not None)
        self._audio_packet_codec_name = normalized_codec_name
        if mode_changed and getattr(self, "is_running", False):
            self._request_restart()

    def forward_audio_to_writer(self, writer: Any, forward_only: bool = True) -> None:
        """writerのenqueue_audio_payloadへ音声payloadを直接転送する

        Args:
            writer: enqueue_audio_payload(payload)を実装するwriterインスタンス
            forward_only: Trueの場合、転送成功時はローカルaudio_queueへ積まない。
        """
        enqueue_audio_payload = getattr(writer, "enqueue_audio_payload", None)
        if not callable(enqueue_audio_payload):
            raise ValueError("writerはenqueue_audio_payload(payload)を実装している必要があります")
        self.set_audio_payload_forwarder(enqueue_audio_payload, forward_only=forward_only)

    def forward_opus_packets_to_writer(self, writer: Any, forward_only: bool = True) -> None:
        """writerのenqueue_audio_packet_payloadへOpus packet payloadを直接転送する

        Args:
            writer: enqueue_audio_packet_payload(payload)を実装するwriterインスタンス
            forward_only: True以外はサポートしない。
        """
        enqueue_audio_packet_payload = getattr(writer, "enqueue_audio_packet_payload", None)
        if not callable(enqueue_audio_packet_payload):
            raise ValueError(
                "writerはenqueue_audio_packet_payload(payload)を実装している必要があります"
            )
        self.set_audio_packet_payload_forwarder(
            enqueue_audio_packet_payload,
            codec_name="opus",
            forward_only=forward_only,
        )

    def set_video_payload_forwarder(
        self,
        forwarder: Callable[[dict[str, Any]], None] | None,
        forward_only: bool = False,
    ) -> None:
        """映像payloadを受信時に直接転送するforwarderを設定する

        Args:
            forwarder: payload(dict)を受け取るコールバック。Noneで無効化。
            forward_only: Trueの場合、転送成功時はローカルvideo_queueへ積まない。
        """
        self._video_payload_forwarder = forwarder
        self._video_forward_only = bool(forwarder is not None and forward_only)

    def forward_video_to_writer(self, writer: Any, forward_only: bool = True) -> None:
        """writerのenqueue_video_payloadへ映像payloadを直接転送する

        Args:
            writer: enqueue_video_payload(payload)を実装するwriterインスタンス
            forward_only: Trueの場合、転送成功時はローカルvideo_queueへ積まない。
        """
        enqueue_video_payload = getattr(writer, "enqueue_video_payload", None)
        if not callable(enqueue_video_payload):
            raise ValueError("writerはenqueue_video_payload(payload)を実装している必要があります")
        self.set_video_payload_forwarder(enqueue_video_payload, forward_only=forward_only)

    def _stop_read_process(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._read_thread is not None:
            self._read_thread.join(timeout=5.0)
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.terminate()
            self._read_thread.join(timeout=5.0)
        self._read_thread = None

    def _close_mp_queues(self) -> None:
        if self._video_mp_queue is not None:
            try:
                self._video_mp_queue.cancel_join_thread()
                self._video_mp_queue.close()
            except Exception:
                pass
        if self._audio_mp_queue is not None:
            try:
                self._audio_mp_queue.cancel_join_thread()
                self._audio_mp_queue.close()
            except Exception:
                pass
        self._video_mp_queue = None
        self._audio_mp_queue = None
        self._stop_event = None

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
        if payload.get("payload_type") == "audio_packet":
            raise ValueError("audio_packet payloadはAudioFrameにデシリアライズできません")
        planes = payload.get("planes")
        if isinstance(planes, list):
            frame = av.AudioFrame(
                format=payload["format"],
                layout=payload["layout"],
                samples=payload["samples"],
            )
            for i, plane_data in enumerate(planes):
                try:
                    frame.planes[i].update(plane_data)
                except Exception:
                    continue
            self._apply_common_frame_attrs(frame, payload)
            self._apply_audio_frame_attrs(frame, payload)
            return WrappedAudioFrame(frame)

        storage = payload.get("storage", "inline")
        if storage == "shm":
            shm_name = payload.get("shm_name")
            plane_spans = payload.get("plane_spans")
            if (
                isinstance(shm_name, str)
                and isinstance(plane_spans, list)
                and plane_spans
            ):
                shm = None
                try:
                    shm = shared_memory.SharedMemory(name=shm_name)
                    frame = av.AudioFrame(
                        format=payload["format"],
                        layout=payload["layout"],
                        samples=payload["samples"],
                    )
                    for i, span in enumerate(plane_spans):
                        if not isinstance(span, (list, tuple)) or len(span) != 2:
                            continue
                        offset = int(span[0])
                        size = int(span[1])
                        try:
                            frame.planes[i].update(shm.buf[offset: offset + size])
                        except Exception:
                            continue
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
                time.sleep(1/60)

            if not self.is_running:
                break

            elapsed = time.time() - self._last_successful_read_time
            if self._read_thread and not self._read_thread.is_alive():
                if elapsed > self._restart_threshold:
                    print(
                        f"読み込みプロセスが停止し、{elapsed:.1f}秒間フレーム更新なし。"
                        "再接続を試みます"
                    )
                    self._force_restart()
                    continue

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
        max_payloads = max(1, self._video_mp_queue_maxlen)
        processed_payloads = 0

        def _iter_payloads(payload_obj: Any):
            if isinstance(payload_obj, dict):
                yield payload_obj
                return
            if isinstance(payload_obj, list):
                for payload in payload_obj:
                    if isinstance(payload, dict):
                        yield payload

        while processed_payloads < max_payloads:
            try:
                payload_obj = self._video_mp_queue.get_nowait()
            except queue.Empty:
                break
            except (ValueError, OSError):
                break
            processed_payloads += 1

            for payload in _iter_payloads(payload_obj):
                forwarded = False
                forwarder = self._video_payload_forwarder
                if forwarder is not None:
                    try:
                        forwarder(payload)
                        forwarded = True
                    except Exception as e:
                        print(f"Video payload転送エラー: {repr(e)}")

                if not (forwarded and self._video_forward_only):
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
        max_payloads = max(1, self._audio_mp_queue_maxlen)
        processed_payloads = 0

        def _iter_payloads(payload_obj: Any):
            if isinstance(payload_obj, dict):
                yield payload_obj
                return
            if isinstance(payload_obj, list):
                for payload in payload_obj:
                    if isinstance(payload, dict):
                        yield payload

        while processed_payloads < max_payloads:
            try:
                payload_obj = self._audio_mp_queue.get_nowait()
            except queue.Empty:
                break
            except (ValueError, OSError):
                break
            processed_payloads += 1

            for payload in _iter_payloads(payload_obj):
                payload_type = payload.get("payload_type", "audio_frame")
                forwarded = False
                if payload_type == "audio_packet":
                    forwarder = self._audio_packet_payload_forwarder
                else:
                    forwarder = self._audio_payload_forwarder
                if forwarder is not None:
                    try:
                        forwarder(payload)
                        forwarded = True
                    except Exception as e:
                        print(f"Audio payload転送エラー: {repr(e)}")

                if payload_type == "audio_packet":
                    if not (forwarded and self._audio_packet_forward_only):
                        if forwarder is None:
                            print("audio_packet payload受信: 転送先未設定のため破棄します")
                        else:
                            print("audio_packet payloadはforward_only=True以外をサポートしません")
                else:
                    if not (forwarded and self._audio_forward_only):
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
                    drop_count = max(1, len(self.video_queue) // 10)
                    print(f"[Listener] video queue full ({len(self.video_queue)}/{self.video_queue.maxlen}), dropping {drop_count} frames")
                    for _ in range(drop_count):
                        self.video_queue.popleft()
                self.video_queue.append(frame)
            except Exception as e:
                print(f"Videoキュー追加エラー: {repr(e)}")

    def pop_all_video_queue(self) -> list[WrappedVideoFrame]:
        """Videoキューからバッチサイズ分のフレームを取り出す"""
        with self.video_queue_lock:
            if len(self.video_queue) > self.batch_size:
                self.video_queue.popleft()
            if len(self.video_queue) < self.batch_size:
                return []
            return_frames: list[WrappedVideoFrame] = []
            for _ in range(self.batch_size):
                video_frame = self.video_queue.popleft()
                if video_frame.frame.pts is not None and video_frame.frame.pts % 30 == 0:
                    pass#print(f"pipeline diff pop■:{(time.time() - video_frame.create_time) * 1000:.3f} ms. pts:{video_frame.frame.pts}")
                return_frames.append(video_frame)
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
                if (
                    self._audio_queue_samples + samples > self._audio_queue_max_samples
                    and self.audio_queue
                ):
                    print(f"[Listener] audio queue full ({self._audio_queue_samples}/{self._audio_queue_max_samples} samples), dropping old frames")
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

    def _reset_for_restart(self) -> None:
        self.container = None
        self._stop_read_process()
        self._close_mp_queues()
        self._clear_local_queues()
        self._restart_requested = False

    def _restart_connection(self) -> None:
        """ストリーム接続を再確立する。サブクラスでオーバーライド可能。"""
        if not self.is_running:
            return

        self._reset_for_restart()
        self._interruptible_sleep(self._restart_wait_seconds)

        if not self.is_running:
            return

        self._setup_multiprocessing()
        self._start_read_process()
        self.container = _StreamListenerContainerProxy(self._request_restart)
        self._restart_requested = False
        self._last_successful_read_time = time.time()
        self._stats_video_frame_count = 0
        self._stats_audio_frame_count = 0
        self._stats_last_time = time.monotonic()
        print("再接続に成功しました")
