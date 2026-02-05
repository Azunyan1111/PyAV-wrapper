from __future__ import annotations

import multiprocessing
import queue
import subprocess
import sys
import time
import traceback

import av

from pyav_wrapper.audio_frame import WrappedAudioFrame
from pyav_wrapper.stream_writer import (
    StreamWriter,
    _StreamWriterContainerProxy,
    _deserialize_audio_frame,
    _deserialize_video_frame,
    _drop_if_full,
    _put_with_drop,
    _serialize_audio_frame,
    _serialize_video_frame,
)
from pyav_wrapper.video_frame import WrappedVideoFrame

DEFAULT_STDERR_LOG_PATH = "/var/log/pyav_wrapper_writer.log"


def _open_worker_stderr(stderr_log_path: str | None):
    if stderr_log_path is None:
        return subprocess.DEVNULL, None
    try:
        stderr_file = open(stderr_log_path, "a")
        return stderr_file, stderr_file
    except Exception:
        return subprocess.DEVNULL, None


def _raw_subprocess_pipe_stream_writer_worker(
    command: list[str],
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
    stderr_log_path: str | None,
    crop_ratio: float | None = None,
) -> None:
    container: av.Container | None = None
    video_stream = None
    audio_stream = None
    process: subprocess.Popen | None = None
    stderr_file = None
    last_video_frame = None

    stats_video_frame_count: int = 0
    stats_audio_frame_count: int = 0
    stats_last_time: float = time.monotonic()
    stats_status_last_time: float = time.monotonic()
    pending_video_count: int = 0
    pending_audio_count: int = 0
    last_write_time: float | None = None

    def _notify_status(status: str, payload: dict[str, object] | None = None) -> None:
        if status_queue is None:
            return
        message: dict[str, object] = {"status": status}
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

    def _wait_for_first_video_payload():
        while not stop_event.is_set():
            try:
                return video_queue.get(timeout=1.0)
            except queue.Empty:
                continue
        return None

    def _open_container():
        if process is None or process.stdin is None:
            raise RuntimeError("サブプロセスのstdinが取得できません")
        next_container = av.open(process.stdin, format="matroska", mode="w")

        next_video_stream = next_container.add_stream("rawvideo", rate=fps)
        next_video_stream.width = width
        next_video_stream.height = height
        next_video_stream.pix_fmt = "yuv420p"

        next_audio_stream = next_container.add_stream("pcm_s16le", rate=sample_rate)
        next_audio_stream.layout = audio_layout

        return next_container, next_video_stream, next_audio_stream

    def _handle_control_commands() -> bool:
        if control_queue is None:
            return False
        close_requested = False
        while True:
            try:
                command_payload = control_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(command_payload, dict) and command_payload.get("cmd") == "close_container":
                close_requested = True
        return close_requested

    try:
        first_payload = _wait_for_first_video_payload()
        if first_payload is None:
            return
        first_frame = _deserialize_video_frame(first_payload)

        if crop_ratio is not None:
            first_frame = first_frame.crop_center(crop_ratio)

        try:
            stderr_dest, stderr_file = _open_worker_stderr(stderr_log_path)
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=stderr_dest,
            )
            container, video_stream, audio_stream = _open_container()
            for packet in video_stream.encode(first_frame.frame):
                container.mux(packet)
            last_write_time = time.time()
            stats_video_frame_count += 1
            if not first_frame.is_bad_frame:
                last_video_frame = first_frame
            _notify_status("started", {"started_time": last_write_time})
        except Exception as e:
            _notify_status("error", {"message": str(e)})
            print(f"処理開始エラー: {str(e)}", file=sys.stderr)
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
                    print(f"RawSubprocessPipeStreamWriter映像muxエラー（スキップ）: {e}", file=sys.stderr)

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
                    print(f"RawSubprocessPipeStreamWriter音声muxエラー（スキップ）: {e}", file=sys.stderr)

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
        print(f"RawSubprocessPipeStreamWriter書き込みエラー: {e}", file=sys.stderr)
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
                print(f"RawSubprocessPipeStreamWriterクローズ時にエラー: {e}", file=sys.stderr)
                traceback.print_exc()

        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    try:
                        process.kill()
                        process.wait()
                    except Exception:
                        pass

        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:
                pass


class RawSubprocessPipeStreamWriter(StreamWriter):
    """サブプロセスのstdinパイプへMKV形式のrawvideo+PCMを書き込むStreamWriter"""

    def __init__(
            self,
            command: list[str],
            width: int,
            height: int,
            fps: int = 30,
            sample_rate: int = 48000,
            audio_layout: str = "stereo",
            stats_enabled: bool = False,
            stderr_log_path: str | None = None,
    ):
        """
        Args:
            command: 実行するコマンド（例: ["./deps/whip-client", "https://..."]）
            width: 出力映像幅
            height: 出力映像高さ
            fps: 出力フレームレート
            sample_rate: 音声サンプルレート（デフォルト: 48000）
            audio_layout: 音声チャンネルレイアウト（デフォルト: "stereo"）
            stats_enabled: FPS統計出力を有効にするかどうか
            stderr_log_path: サブプロセスのstderr出力先ファイルパス
                            Noneの場合は出力しない
        """
        self._command = command
        self._process: subprocess.Popen | None = None
        self._stderr_log_path = stderr_log_path
        self._stderr_file = None
        self._last_video_dimensions: tuple[int, int] | None = None
        super().__init__(
            url="pipe:",
            width=width,
            height=height,
            fps=fps,
            sample_rate=sample_rate,
            audio_layout=audio_layout,
            stats_enabled=stats_enabled,
        )

    def set_stderr_log_path(self, path: str | None) -> None:
        """サブプロセスのstderr出力先ファイルパスを設定する

        Args:
            path: 出力先ファイルパス。Noneの場合は出力しない。
                  次回のプロセス起動時から有効。
        """
        self._stderr_log_path = path

    def enqueue_video_frame(self, frame: WrappedVideoFrame) -> None:
        """映像フレームをMPキューに追加（軽量化: ローカルキューは使用しない）

        Args:
            frame: 送信するWrappedVideoFrame
        """
        if self._video_mp_queue is None:
            return

        try:
            self._last_video_dimensions = (frame.frame.width, frame.frame.height)
        except Exception:
            pass

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
            print(f"映像フレームをMPキューに追加中にエラー: {repr(e)}")

    def enqueue_audio_frame(self, frame: WrappedAudioFrame) -> None:
        """音声フレームをMPキューに追加（軽量化: ローカルキューは使用しない）

        Args:
            frame: 送信するWrappedAudioFrame
        """
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
            print(f"音声フレームをMPキューに追加中にエラー: {repr(e)}")

    def _open_stderr_file(self):
        """stderrログファイルを開く"""
        if self._stderr_log_path is not None:
            self._stderr_file = open(self._stderr_log_path, "a")
            return self._stderr_file
        return subprocess.DEVNULL

    def _close_stderr_file(self):
        """stderrログファイルを閉じる"""
        stderr_file = getattr(self, "_stderr_file", None)
        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

    def start_processing(self) -> str:
        """サブプロセスを起動し、stdinパイプへストリーム書き込みを開始"""
        return super().start_processing()

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
            target=_raw_subprocess_pipe_stream_writer_worker,
            args=(
                self._command,
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
                self._stderr_log_path,
                self._crop_ratio,
            ),
            daemon=True,
        )
        self._write_process.start()
        self._thread = self._write_process
        self.container = _StreamWriterContainerProxy(self._request_container_close)

    def stop(self) -> None:
        """ストリーム処理とサブプロセスを停止"""
        super().stop()
        self._process = None
        self._close_stderr_file()

    def _restart_connection(self) -> bool:
        """サブプロセスパイプ接続を再確立する。

        Returns:
            bool: 再接続に成功した場合True、失敗した場合False
        """
        if not self.is_running:
            return False

        old_container = self.container
        self.container = None
        if old_container:
            try:
                old_container.close()
            except Exception:
                pass

        if self._write_process is not None:
            self._stop_write_process()
        elif self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            self._thread = None

        if not self.is_running:
            return False

        self._interruptible_sleep(self._restart_wait_seconds)

        if not self.is_running:
            return False

        self._clear_mp_queue(self._video_mp_queue)
        self._clear_mp_queue(self._audio_mp_queue)
        self._clear_local_queues()

        width = self.width
        height = self.height
        if self._last_video_dimensions is not None:
            width, height = self._last_video_dimensions

        self._last_successful_write_time = time.time()
        self._start_write_process(width=width, height=height)
        print("RawSubprocessPipeStreamWriter再接続を開始しました")
        return True
