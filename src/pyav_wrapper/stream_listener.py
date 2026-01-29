import collections
import os
import threading
import time

import av

from pyav_wrapper.audio_frame import WrappedAudioFrame
from pyav_wrapper.video_frame import WrappedVideoFrame


class StreamListener:
    """PyAVでストリームを受信し、Video/Audioフレームをバッファリングするクラス"""

    def __init__(self, url: str, width: int | None = None, height: int | None = None):
        """
        Args:
            url: 任意のストリームURL（srt://, rtmp://, udp://等）
            width: 出力幅（リサイズ用、Noneで元サイズ維持）
            height: 出力高さ（リサイズ用、Noneで元サイズ維持）
        """
        self.url = url
        self.width = width
        self.height = height
        self.is_running = False
        self.container: av.Container | None = None
        self._read_thread: threading.Thread | None = None

        # Video
        self.batch_size = 30
        self.video_queue: collections.deque[WrappedVideoFrame] = collections.deque(
            maxlen=int(self.batch_size * 1.7)
        )
        self.video_queue_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.current_frame: WrappedVideoFrame | None = None

        # Audio
        self.audio_queue: collections.deque[WrappedAudioFrame] = collections.deque(
            maxlen=int(self.batch_size * 1.7)
        )
        self.audio_queue_lock = threading.Lock()

        # Reconnection
        self._last_successful_read_time: float = time.time()
        self._restart_threshold: float = 10.0
        self._restart_threshold_increment: float = 1.0
        self._restart_threshold_max: float = 20.0
        self._restart_wait_seconds: float = 5.0
        self._restart_lock: threading.Lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None

        self.start_processing()

    def start_processing(self) -> str:
        """ストリーム処理を開始"""
        try:
            self.container = av.open(self.url)
            self.is_running = True
            self._last_successful_read_time = time.time()
            self._read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self._read_thread.start()
            self._monitor_thread = threading.Thread(target=self._monitor_frame_updates, daemon=True)
            self._monitor_thread.start()
            return "ストリーム処理を開始しました"
        except Exception as e:
            return f"処理開始エラー: {str(e)}"

    def _read_frames(self) -> None:
        """フレーム読み込みスレッド（Video + Audio同時デコード）"""
        try:
            container = self.container
            if container is None:
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
                if not self.is_running:
                    break

                self._last_successful_read_time = time.time()

                if isinstance(frame, av.VideoFrame):
                    if self.width is not None and self.height is not None:
                        if frame.width != self.width or frame.height != self.height:
                            frame = frame.reformat(width=self.width, height=self.height)
                    wrapped = WrappedVideoFrame(frame)
                    with self.frame_lock:
                        self.current_frame = wrapped
                    self.append_video_queue(wrapped)

                elif isinstance(frame, av.AudioFrame):
                    wrapped = WrappedAudioFrame(frame)
                    self.append_audio_queue(wrapped)

        except Exception as e:
            print(f"フレーム読み込みエラー: {str(e)}")
        finally:
            self._last_successful_read_time = time.time()

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
            frames = list(self.video_queue)
            return_frames = frames[: self.batch_size]
            self.video_queue = collections.deque(
                frames[self.batch_size :], maxlen=int(self.batch_size * 1.7)
            )
            return return_frames

    def append_audio_queue(self, frame: WrappedAudioFrame) -> None:
        """Audioキューにフレームを追加"""
        if frame is None:
            return
        with self.audio_queue_lock:
            try:
                if len(self.audio_queue) >= self.audio_queue.maxlen:
                    for _ in range(max(1, len(self.audio_queue) // 10)):
                        self.audio_queue.pop()
                self.audio_queue.append(frame)
            except Exception as e:
                print(f"Audioキュー追加エラー: {repr(e)}")

    def pop_all_audio_queue(self) -> list[WrappedAudioFrame]:
        """Audioキューからバッチサイズ分のフレームを取り出す"""
        with self.audio_queue_lock:
            if len(self.audio_queue) < self.batch_size:
                return []
            frames = list(self.audio_queue)
            return_frames = frames[: self.batch_size]
            self.audio_queue = collections.deque(
                frames[self.batch_size :], maxlen=int(self.batch_size * 1.7)
            )
            return return_frames

    def stop(self) -> None:
        """ストリーム処理を停止"""
        self.is_running = False
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=5.0)
        self._read_thread = None
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        self._monitor_thread = None
        with self._restart_lock:
            if self.container:
                try:
                    self.container.close()
                except Exception:
                    pass
                self.container = None

    def _monitor_frame_updates(self) -> None:
        """フレーム更新を監視し、閾値超過時に再接続を試みる"""
        while self.is_running:
            time.sleep(1.0)
            if not self.is_running:
                break

            elapsed = time.time() - self._last_successful_read_time
            if elapsed > self._restart_threshold:
                print(f"フレーム更新が{elapsed:.1f}秒間停止。再接続を試みます (閾値: {self._restart_threshold:.1f}秒)")
                with self._restart_lock:
                    if not self.is_running:
                        break
                    self._restart_threshold += self._restart_threshold_increment
                    if self._restart_threshold >= self._restart_threshold_max:
                        print(f"再接続閾値が上限({self._restart_threshold_max}秒)に達しました。プロセスを終了します。")
                        os._exit(1)
                    self._restart_connection()

    def _restart_connection(self) -> None:
        """ストリーム接続を再確立する。サブクラスでオーバーライド可能。"""
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

        if not self.is_running:
            return

        # 再接続待機
        time.sleep(self._restart_wait_seconds)

        if not self.is_running:
            return

        # 新しいcontainerを作成
        try:
            self.container = av.open(self.url)
            self._last_successful_read_time = time.time()
            self._read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self._read_thread.start()
            print("再接続に成功しました")
        except Exception as e:
            print(f"再接続エラー: {str(e)}")
