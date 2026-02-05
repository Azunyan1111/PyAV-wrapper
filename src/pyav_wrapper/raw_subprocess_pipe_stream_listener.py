import subprocess
import time

import av

from pyav_wrapper.stream_listener import (
    StreamListener,
    _put_with_drop,
    _serialize_audio_frame,
    _serialize_video_frame,
)
from pyav_wrapper.video_frame import WrappedVideoFrame

DEFAULT_STDERR_LOG_PATH = "/var/log/pyav_wrapper_listener.log"


def _raw_subprocess_pipe_decode_worker(
    command: list[str],
    width: int,
    height: int,
    stop_event,
    video_queue,
    audio_queue,
    video_drop_count: int,
    audio_drop_count: int,
    start_timeout_seconds: float,
    start_retry_interval_seconds: float,
    stderr_log_path: str | None,
    crop_ratio: float | None = None,
) -> None:
    process: subprocess.Popen | None = None
    container: av.container.Container | None = None
    stderr_file = None
    try:
        if stderr_log_path is not None:
            stderr_file = open(stderr_log_path, "a")
            stderr_dest = stderr_file
        else:
            stderr_dest = subprocess.DEVNULL

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_dest,
        )

        deadline = time.monotonic() + start_timeout_seconds
        last_error: str | None = None
        while time.monotonic() < deadline and not stop_event.is_set():
            if process.poll() is not None:
                last_error = f"サブプロセスが終了しました (returncode={process.returncode})"
                break
            try:
                container = av.open(process.stdout, format="matroska", mode="r")
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(start_retry_interval_seconds)

        if container is None:
            if last_error is not None:
                print(f"コンテナのopenに失敗しました: {last_error}")
            return

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
        if process is not None:
            try:
                if process.stdout:
                    process.stdout.close()
            except Exception:
                pass
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            except Exception:
                pass
        if stderr_file is not None:
            try:
                stderr_file.close()
            except Exception:
                pass


class RawSubprocessPipeStreamListener(StreamListener):
    """サブプロセスのstdoutパイプからMKV形式のrawvideo+PCMを受信するStreamListener"""

    def __init__(
        self,
        command: list[str],
        width: int,
        height: int,
        fps: int = 30,
        sample_rate: int = 48000,
        audio_layout: str = "stereo",
        stderr_log_path: str | None = None,
        crop_ratio: float | None = None,
    ):
        """
        Args:
            command: 実行するコマンド（例: ["./deps/whep-client", "https://..."]）
            width: 出力幅（リサイズ用）
            height: 出力高さ（リサイズ用）
            fps: 想定フレームレート（保持用）
            sample_rate: 想定音声サンプルレート（保持用）
            audio_layout: 想定音声チャンネルレイアウト（保持用）
            stderr_log_path: サブプロセスのstderr出力先ファイルパス
                            Noneの場合は出力しない
            crop_ratio: 受信時に適用するクロップ比率（0.0〜1.0）
        """
        self._command = command
        self._process: subprocess.Popen | None = None
        self._start_timeout_seconds = 15.0
        self._start_retry_interval_seconds = 1.0
        self._start_error: str | None = None
        self._stderr_log_path = stderr_log_path
        super().__init__(
            url="pipe:",
            width=width,
            height=height,
            fps=fps,
            sample_rate=sample_rate,
            audio_layout=audio_layout,
            crop_ratio=crop_ratio,
        )

    def set_stderr_log_path(self, path: str | None) -> None:
        """サブプロセスのstderr出力先ファイルパスを設定する

        Args:
            path: 出力先ファイルパス。Noneの場合は出力しない。
                  次回のプロセス起動時から有効。
        """
        self._stderr_log_path = path

    def _start_read_process(self) -> None:
        if self._stop_event is None:
            self._stop_event = self._mp_ctx.Event()
        self._read_thread = self._mp_ctx.Process(
            target=_raw_subprocess_pipe_decode_worker,
            args=(
                self._command,
                self.width,
                self.height,
                self._stop_event,
                self._video_mp_queue,
                self._audio_mp_queue,
                self._video_drop_count,
                self._audio_drop_count,
                self._start_timeout_seconds,
                self._start_retry_interval_seconds,
                self._stderr_log_path,
                self._crop_ratio,
            ),
            daemon=True,
        )
        self._read_thread.start()

    def stop(self) -> None:
        """ストリーム処理とサブプロセスを停止"""
        super().stop()
        self._process = None

    def _restart_connection(self) -> None:
        """サブプロセスパイプ接続を再確立する"""
        self._start_error = None
        self._process = None
        super()._restart_connection()
