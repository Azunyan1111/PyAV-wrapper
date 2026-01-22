import collections
import threading

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

        self.start_processing()

    def start_processing(self) -> str:
        """ストリーム処理を開始"""
        try:
            self.container = av.open(self.url)
            self.is_running = True
            self._read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self._read_thread.start()
            return "ストリーム処理を開始しました"
        except Exception as e:
            return f"処理開始エラー: {str(e)}"

    def _read_frames(self) -> None:
        """フレーム読み込みスレッド（Video + Audio同時デコード）"""
        try:
            video_stream = (
                self.container.streams.video[0]
                if self.container.streams.video
                else None
            )
            audio_stream = (
                self.container.streams.audio[0]
                if self.container.streams.audio
                else None
            )

            streams_to_decode = [
                s for s in [video_stream, audio_stream] if s is not None
            ]

            for frame in self.container.decode(*streams_to_decode):
                if not self.is_running:
                    break

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
            if self.container:
                self.container.close()

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
