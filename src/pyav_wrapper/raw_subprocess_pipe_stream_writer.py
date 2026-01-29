import queue
import subprocess
import threading
import time

import av

from pyav_wrapper.stream_writer import StreamWriter


class RawSubprocessPipeStreamWriter(StreamWriter):
    """サブプロセスのstdinパイプへMKV形式のrawvideo+PCMを書き込むStreamWriter"""

    def __init__(
        self,
        command: list[str],
        width: int,
        height: int,
        fps: int,
        sample_rate: int = 48000,
        audio_layout: str = "stereo",
    ):
        """
        Args:
            command: 実行するコマンド（例: ["./deps/whip-client", "https://..."]）
            width: 出力映像幅
            height: 出力映像高さ
            fps: 出力フレームレート
            sample_rate: 音声サンプルレート（デフォルト: 48000）
            audio_layout: 音声チャンネルレイアウト（デフォルト: "stereo"）
        """
        self._command = command
        self._process: subprocess.Popen | None = None
        super().__init__(
            url="pipe:",
            width=width,
            height=height,
            fps=fps,
            sample_rate=sample_rate,
            audio_layout=audio_layout,
        )

    def start_processing(self) -> str:
        """サブプロセスを起動し、stdinパイプへストリーム書き込みを開始"""
        try:
            # 最初のフレームを待って解像度を取得
            first_frame = None
            while first_frame is None:
                try:
                    first_frame = self.video_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

            self._first_frame = first_frame

            # フレームから実際の解像度を取得
            actual_width = first_frame.frame.width
            actual_height = first_frame.frame.height

            # サブプロセスを起動
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # stdinパイプをMKVコンテナとして開く
            self.container = av.open(
                self._process.stdin, format="matroska", mode="w"
            )

            # rawvideo ストリーム追加
            self._video_stream = self.container.add_stream("rawvideo", rate=self.fps)
            self._video_stream.width = actual_width
            self._video_stream.height = actual_height
            self._video_stream.pix_fmt = "yuv420p"

            # pcm_s16le 音声ストリーム追加
            self._audio_stream = self.container.add_stream(
                "pcm_s16le", rate=self.sample_rate
            )
            self._audio_stream.layout = self.audio_layout

            self.is_running = True
            self._thread = threading.Thread(target=self._write_frames, daemon=True)
            self._thread.start()
            return "サブプロセスパイプへのストリーム書き込みを開始しました"
        except Exception as e:
            if self._process is not None:
                self._process.kill()
                self._process.wait()
                self._process = None
            return f"処理開始エラー: {str(e)}"

    def stop(self) -> None:
        """ストリーム処理とサブプロセスを停止"""
        super().stop()
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
