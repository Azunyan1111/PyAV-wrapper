import queue
import threading
import time
from typing import TYPE_CHECKING

import av

from pyav_wrapper.audio_frame import WrappedAudioFrame
from pyav_wrapper.video_frame import WrappedVideoFrame

if TYPE_CHECKING:
    from threading import Thread


class StreamWriter:
    """PyAVを使用してSRTストリームへ映像・音声フレームを送信するクラス"""

    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        fps: int,
        sample_rate: int = 48000,
        audio_layout: str = "stereo",
    ) -> None:
        """
        Args:
            url: SRT出力URL（例: "srt://host:port?mode=caller&latency=120"）
            width: 出力映像幅
            height: 出力映像高さ
            fps: 出力フレームレート
            sample_rate: 音声サンプルレート（デフォルト: 48000）
            audio_layout: 音声チャンネルレイアウト（デフォルト: "stereo"）
        """
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        self.audio_layout = audio_layout

        # キューの初期化（FPSの1.7倍のキャパシティ）
        # スレッドベースなのでqueue.Queue（pickle不要）を使用
        # WrappedVideoFrame/WrappedAudioFrameを直接入れる
        queue_size = int(fps * 1.7)
        self.video_queue: queue.Queue[WrappedVideoFrame] = queue.Queue(
            maxsize=queue_size
        )
        self.audio_queue: queue.Queue[WrappedAudioFrame] = queue.Queue(
            maxsize=queue_size
        )

        # スレッド管理
        self.running = False
        self._thread: "Thread | None" = None

        # 古いフレームの保持（フレームが取得できない場合の再利用用）
        self._last_video_frame: "WrappedVideoFrame | None" = None

        # 自動起動
        self.start()

    def enqueue_video_frame(self, frame: WrappedVideoFrame) -> None:
        """映像フレームをキューに追加（ノンブロッキング）

        Args:
            frame: 送信するWrappedVideoFrame
        """
        try:
            if self.video_queue.full():
                # キューが満杯の場合は古いフレームを捨てる（50%程度）
                try:
                    drop_count = max(1, self.video_queue.qsize() // 2)
                except NotImplementedError:
                    drop_count = 1  # macOSではqsize()が使えない
                for _ in range(drop_count):
                    try:
                        self.video_queue.get_nowait()
                    except queue.Empty:
                        break

            # WrappedVideoFrameを直接キューに入れる
            self.video_queue.put_nowait(frame)
        except Exception as e:
            print(f"映像フレームをキューに追加中にエラー: {repr(e)}")

    def enqueue_video_frames(self, frames: list[WrappedVideoFrame]) -> None:
        """複数の映像フレームをキューに追加

        Args:
            frames: 送信するWrappedVideoFrameのリスト
        """
        for frame in frames:
            self.enqueue_video_frame(frame)

    def enqueue_audio_frame(self, frame: WrappedAudioFrame) -> None:
        """音声フレームをキューに追加（ノンブロッキング）

        Args:
            frame: 送信するWrappedAudioFrame
        """
        try:
            if self.audio_queue.full():
                # キューが満杯の場合は古いフレームを捨てる（50%程度）
                try:
                    drop_count = max(1, self.audio_queue.qsize() // 2)
                except NotImplementedError:
                    drop_count = 1  # macOSではqsize()が使えない
                for _ in range(drop_count):
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break

            # WrappedAudioFrameを直接キューに入れる
            self.audio_queue.put_nowait(frame)
        except Exception as e:
            print(f"音声フレームをキューに追加中にエラー: {repr(e)}")

    def enqueue_audio_frames(self, frames: list[WrappedAudioFrame]) -> None:
        """複数の音声フレームをキューに追加

        Args:
            frames: 送信するWrappedAudioFrameのリスト
        """
        for frame in frames:
            self.enqueue_audio_frame(frame)

    def start(self) -> None:
        """エンコード・送信スレッドを開始"""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(
                target=self._process_queue_loop,
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """エンコード・送信スレッドを停止"""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

    def __del__(self) -> None:
        """リソース解放"""
        self.stop()

    def _process_queue_loop(self) -> None:
        """キューからフレームを取り出してエンコード・送信するプロセスループ"""
        # 別プロセスなのでimportが必要
        import sys
        import traceback

        try:
            # 最初のフレームを待って解像度を取得
            first_frame = None
            while self.running and first_frame is None:
                try:
                    first_frame = self.video_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

            if first_frame is None:
                return

            # フレームから実際の解像度を取得
            actual_width = first_frame.frame.width
            actual_height = first_frame.frame.height

            # PyAVコンテナとストリームを初期化
            # ファイル出力の場合はformatを明示せず拡張子から自動判別
            # SRT出力の場合はmpegtsを使用
            if self.url.startswith("srt://"):
                container = av.open(self.url, mode="w", format="mpegts")
            else:
                container = av.open(self.url, mode="w")

            video_stream = container.add_stream("libx264", rate=self.fps)
            video_stream.width = actual_width
            video_stream.height = actual_height
            video_stream.pix_fmt = "yuv420p"
            video_stream.options = {"preset": "ultrafast", "tune": "zerolatency"}

            audio_stream = container.add_stream("aac", rate=self.sample_rate)
            audio_stream.layout = self.audio_layout

            # 最初のフレームを処理（元のPTSをそのまま使用）
            for packet in video_stream.encode(first_frame.frame):
                container.mux(packet)

            while self.running:
                has_data = False

                # 映像フレームの処理
                wrapped_frame = self._process_video_frame()
                if wrapped_frame is not None:
                    has_data = True
                    # 元のPTSとtime_baseをそのまま使用
                    for packet in video_stream.encode(wrapped_frame.frame):
                        container.mux(packet)

                # 音声フレームの処理（キューにある全フレームを処理）
                while True:
                    wrapped_audio = self._get_audio_frame()
                    if wrapped_audio is None:
                        break
                    has_data = True
                    # 元のPTSとtime_baseをそのまま使用
                    for packet in audio_stream.encode(wrapped_audio.frame):
                        container.mux(packet)

                if not has_data:
                    time.sleep(0.001)

            # フラッシュ
            try:
                for packet in video_stream.encode():
                    container.mux(packet)
                for packet in audio_stream.encode():
                    container.mux(packet)
                container.close()
            except Exception as e:
                print(f"StreamWriterクローズ時にエラー: {e}", file=sys.stderr)
                traceback.print_exc()
        except Exception as e:
            print(f"StreamWriter初期化エラー: {e}", file=sys.stderr)
            traceback.print_exc()

    def _process_video_frame(self) -> "WrappedVideoFrame | None":
        """映像フレームを取得

        is_bad_frameがTrueの場合は、最後に送信したフレームの生データで差し替える。

        Returns:
            WrappedVideoFrame、またはNone
        """
        try:
            wrapped_frame: WrappedVideoFrame = self.video_queue.get(timeout=1 / 30)
        except queue.Empty:
            return None

        if wrapped_frame.is_bad_frame:
            # 最後に送信したフレームの生データで差し替え
            if self._last_video_frame is not None:
                last_planes = self._last_video_frame.get_planes()
                wrapped_frame.set_planes(last_planes)
        else:
            # 生データがある場合は保存
            self._last_video_frame = wrapped_frame

        return wrapped_frame

    def _get_audio_frame(self) -> "WrappedAudioFrame | None":
        """音声フレームをキューから取得

        Returns:
            WrappedAudioFrame、またはNone
        """
        try:
            wrapped_frame: WrappedAudioFrame = self.audio_queue.get_nowait()
        except queue.Empty:
            return None

        return wrapped_frame
