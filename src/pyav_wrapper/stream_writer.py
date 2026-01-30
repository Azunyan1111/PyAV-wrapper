import queue
import sys
import threading
import time
import traceback
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
        self.is_running = False
        self._thread: "Thread | None" = None
        self._init_thread: "Thread | None" = None

        # コンテナ（インスタンス変数として管理）
        self.container: av.Container | None = None
        self._video_stream = None
        self._audio_stream = None

        # 最初のフレーム（start_processingで取得、_write_framesで使用）
        self._first_frame: WrappedVideoFrame | None = None

        # 古いフレームの保持（フレームが取得できない場合の再利用用）
        self._last_video_frame: "WrappedVideoFrame | None" = None

        # リアルタイムペーシング用
        self._wall_clock_origin: float = 0.0
        self._pts_origin: float = 0.0

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
        """start_processing() をスレッドで起動するラッパー"""
        self._init_thread = threading.Thread(target=self.start_processing, daemon=True)
        self._init_thread.start()

    def start_processing(self) -> str:
        """コンテナを初期化し、書き込みスレッドを開始"""
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

            # PyAVコンテナとストリームを初期化
            # ファイル出力の場合はformatを明示せず拡張子から自動判別
            # SRT出力の場合はmpegtsを使用
            if self.url.startswith("srt://"):
                self.container = av.open(self.url, mode="w", format="mpegts")
            else:
                self.container = av.open(self.url, mode="w")

            self._video_stream = self.container.add_stream("libx264", rate=self.fps)
            self._video_stream.width = actual_width
            self._video_stream.height = actual_height
            self._video_stream.pix_fmt = "yuv420p"
            self._video_stream.options = {"preset": "ultrafast", "tune": "zerolatency"}

            self._audio_stream = self.container.add_stream("aac", rate=self.sample_rate)
            self._audio_stream.layout = self.audio_layout

            self.is_running = True
            self._thread = threading.Thread(target=self._write_frames, daemon=True)
            self._thread.start()
            return "ストリーム処理を開始しました"
        except Exception as e:
            return f"処理開始エラー: {str(e)}"

    def _init_realtime_clock(self, frame: av.VideoFrame) -> None:
        """リアルタイムペーシングの基準時刻を設定する

        Args:
            frame: 最初の映像フレーム（PTS/time_baseから基準再生時刻を算出）
        """
        self._wall_clock_origin = time.monotonic()
        if frame.pts is not None and frame.time_base is not None:
            self._pts_origin = float(frame.pts * frame.time_base)
        else:
            self._pts_origin = 0.0

    def _pace_frame(self, frame: av.VideoFrame) -> None:
        """PTSに基づき壁時計とアラインメントするためスリープする

        Args:
            frame: ペーシング対象の映像フレーム
        """
        if frame.pts is None or frame.time_base is None:
            return

        frame_time_sec = float(frame.pts * frame.time_base)
        target_elapsed = frame_time_sec - self._pts_origin
        actual_elapsed = time.monotonic() - self._wall_clock_origin
        sleep_duration = target_elapsed - actual_elapsed

        if sleep_duration > 0:
            time.sleep(min(sleep_duration, 1.0))

    def _write_frames(self) -> None:
        """エンコード・送信ループ (StreamListenerの_read_framesに対応)"""
        try:
            # リアルタイムペーシングの基準時刻を設定
            self._init_realtime_clock(self._first_frame.frame)

            # 最初のフレームを処理（元のPTSをそのまま使用）
            for packet in self._video_stream.encode(self._first_frame.frame):
                self.container.mux(packet)

            while self.is_running:
                has_data = False

                # 映像フレームの処理
                wrapped_frame = self._process_video_frame()
                if wrapped_frame is not None:
                    has_data = True
                    self._pace_frame(wrapped_frame.frame)
                    # 元のPTSとtime_baseをそのまま使用
                    for packet in self._video_stream.encode(wrapped_frame.frame):
                        self.container.mux(packet)

                # 音声フレームの処理（キューにある全フレームを処理）
                while True:
                    wrapped_audio = self._get_audio_frame()
                    if wrapped_audio is None:
                        break
                    has_data = True
                    # 元のPTSとtime_baseをそのまま使用
                    for packet in self._audio_stream.encode(wrapped_audio.frame):
                        self.container.mux(packet)

                if not has_data:
                    time.sleep(0.001)

        except Exception as e:
            print(f"StreamWriter書き込みエラー: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            # フラッシュ & クローズ
            if self.container:
                try:
                    for packet in self._video_stream.encode():
                        self.container.mux(packet)
                    for packet in self._audio_stream.encode():
                        self.container.mux(packet)
                    self.container.close()
                except Exception as e:
                    print(f"StreamWriterクローズ時にエラー: {e}", file=sys.stderr)
                    traceback.print_exc()

    def stop(self) -> None:
        """ストリーム処理を停止"""
        self.is_running = False
        if self._init_thread and self._init_thread.is_alive():
            self._init_thread.join(timeout=5.0)
        self._init_thread = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

    def __del__(self) -> None:
        """リソース解放"""
        self.stop()

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
