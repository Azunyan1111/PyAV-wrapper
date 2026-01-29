import subprocess
import threading
import time

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
            self._last_successful_read_time = time.time()
            self._read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self._read_thread.start()
            self._monitor_thread = threading.Thread(target=self._monitor_frame_updates, daemon=True)
            self._monitor_thread.start()
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

    def _restart_connection(self) -> None:
        """サブプロセスパイプ接続を再確立する"""
        if not self.is_running:
            return

        # 古いcontainerをclose
        if self.container:
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None

        # 古い読み込みスレッドを待機
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=5.0)

        # 古いサブプロセスを停止
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            except Exception:
                pass
            self._process = None

        if not self.is_running:
            return

        # 再接続待機
        time.sleep(self._restart_wait_seconds)

        if not self.is_running:
            return

        # 新しいサブプロセスを起動
        try:
            self._process = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.container = av.open(self._process.stdout, format="matroska", mode="r")
            self._last_successful_read_time = time.time()
            self._read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self._read_thread.start()
            print("サブプロセスパイプの再接続に成功しました")
        except Exception as e:
            print(f"サブプロセスパイプの再接続エラー: {str(e)}")
            if self._process is not None:
                try:
                    self._process.kill()
                    self._process.wait()
                except Exception:
                    pass
                self._process = None
