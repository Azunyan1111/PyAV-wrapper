import queue
import subprocess
import sys
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
            fps: int = 30,
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
            self._video_stream.width = self.width
            self._video_stream.height = self.height
            self._video_stream.pix_fmt = "yuv420p"

            # pcm_s16le 音声ストリーム追加
            self._audio_stream = self.container.add_stream(
                "pcm_s16le", rate=self.sample_rate
            )
            self._audio_stream.layout = self.audio_layout

            self.is_running = True
            self._last_successful_write_time = time.time()
            self._thread = threading.Thread(target=self._write_frames, daemon=True)
            self._thread.start()
            self._monitor_thread = threading.Thread(target=self._monitor_write_updates, daemon=True)
            self._monitor_thread.start()
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

    def _restart_connection(self) -> bool:
        """サブプロセスパイプ接続を再確立する。

        Returns:
            bool: 再接続に成功した場合True、失敗した場合False
        """
        if not self.is_running:
            return False

        # 古いcontainerをNoneに設定後close（_write_framesがbreakする）
        old_container = self.container
        self.container = None
        if old_container:
            try:
                old_container.close()
            except Exception:
                pass

        # 古い書込スレッドを待機
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

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
            return False

        # 再接続待機
        self._interruptible_sleep(self._restart_wait_seconds)

        if not self.is_running:
            return False

        # 新しいサブプロセスを起動
        try:
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
            self._video_stream.width = self.width
            self._video_stream.height = self.height
            self._video_stream.pix_fmt = "yuv420p"

            # pcm_s16le 音声ストリーム追加
            self._audio_stream = self.container.add_stream(
                "pcm_s16le", rate=self.sample_rate
            )
            self._audio_stream.layout = self.audio_layout

            self._last_successful_write_time = time.time()
            self._thread = threading.Thread(target=self._write_frames, daemon=True)
            self._thread.start()
            print("サブプロセスパイプのStreamWriter再接続に成功しました")
            return True
        except Exception as e:
            print(f"サブプロセスパイプのStreamWriter再接続エラー: {str(e)}", file=sys.stderr)
            if self._process is not None:
                try:
                    self._process.kill()
                    self._process.wait()
                except Exception:
                    pass
                self._process = None
            return False
