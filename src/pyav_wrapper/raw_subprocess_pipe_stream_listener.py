import subprocess
import threading

import av

from pyav_wrapper.stream_listener import StreamListener


class RawSubprocessPipeStreamListener(StreamListener):
    """サブプロセスのstdoutパイプからMKV形式のrawvideo+PCMを受信するStreamListener"""

    def __init__(self, command: list[str], width: int | None = None, height: int | None = None):
        """
        Args:
            command: 実行するコマンド（例: ["./deps/whep-client", "https://..."]）
            width: 出力幅（リサイズ用、Noneで元サイズ維持）
            height: 出力高さ（リサイズ用、Noneで元サイズ維持）
        """
        self._command = command
        self._process: subprocess.Popen | None = None
        super().__init__(url="pipe:", width=width, height=height)

    def start_processing(self) -> str:
        """サブプロセスを起動し、stdoutパイプからストリーム処理を開始"""
        try:
            self._process = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.container = av.open(self._process.stdout, format="matroska", mode="r")
            self.is_running = True
            self._read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self._read_thread.start()
            return "サブプロセスパイプからのストリーム処理を開始しました"
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
