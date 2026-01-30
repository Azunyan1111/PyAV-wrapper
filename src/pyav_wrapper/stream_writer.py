import os
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

        # Reconnection
        self._last_successful_write_time: float = 0.0
        self._restart_threshold: float = 10.0
        self._restart_threshold_increment: float = 1.0
        self._restart_threshold_max: float = 20.0
        self._restart_wait_seconds: float = 5.0
        self._restart_frame_wait_timeout: float = 30.0
        self._restart_lock: threading.Lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None

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
            self._last_successful_write_time = time.time()
            self._thread = threading.Thread(target=self._write_frames, daemon=True)
            self._thread.start()
            self._monitor_thread = threading.Thread(target=self._monitor_write_updates, daemon=True)
            self._monitor_thread.start()
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
            self._last_successful_write_time = time.time()

            while self.is_running:
                if self.container is None:
                    break

                has_data = False

                # 映像フレームの処理
                wrapped_frame = self._process_video_frame()
                if wrapped_frame is not None:
                    has_data = True
                    self._pace_frame(wrapped_frame.frame)
                    # 元のPTSとtime_baseをそのまま使用
                    try:
                        for packet in self._video_stream.encode(wrapped_frame.frame):
                            self.container.mux(packet)
                        self._last_successful_write_time = time.time()
                    except Exception as e:
                        print(f"StreamWriter映像muxエラー（スキップ）: {e}", file=sys.stderr)

                # 音声フレームの処理（キューにある全フレームを処理）
                while True:
                    wrapped_audio = self._get_audio_frame()
                    if wrapped_audio is None:
                        break
                    has_data = True
                    # 元のPTSとtime_baseをそのまま使用
                    try:
                        for packet in self._audio_stream.encode(wrapped_audio.frame):
                            self.container.mux(packet)
                        self._last_successful_write_time = time.time()
                    except Exception as e:
                        print(f"StreamWriter音声muxエラー（スキップ）: {e}", file=sys.stderr)

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
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10.0)
        self._monitor_thread = None

    def _monitor_write_updates(self) -> None:
        """書き込み更新を監視し、閾値超過時に再接続を試みる"""
        while self.is_running:
            time.sleep(1.0)
            if not self.is_running:
                break

            elapsed = time.time() - self._last_successful_write_time
            if elapsed > self._restart_threshold:
                print(f"書き込み更新が{elapsed:.1f}秒間停止。再接続を試みます (閾値: {self._restart_threshold:.1f}秒)")
                with self._restart_lock:
                    if not self.is_running:
                        break
                    self._restart_threshold += self._restart_threshold_increment
                    if self._restart_threshold >= self._restart_threshold_max:
                        print(f"再接続閾値が上限({self._restart_threshold_max}秒)に達しました。プロセスを終了します。")
                        os._exit(1)
                    self._restart_connection()
                    # _restart_connectionがFalseを返した場合もループを継続し再試行する

    def _interruptible_sleep(self, seconds: float) -> None:
        """is_runningチェック付きスリープ。0.5秒間隔で分割。"""
        remaining = seconds
        while remaining > 0 and self.is_running:
            sleep_time = min(remaining, 0.5)
            time.sleep(sleep_time)
            remaining -= sleep_time

    def _wait_for_first_frame(self, timeout: float) -> "WrappedVideoFrame | None":
        """タイムアウト付きで最初のフレームをキューから取得。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running:
                return None
            try:
                return self.video_queue.get(timeout=1.0)
            except queue.Empty:
                continue
        return None

    def _restart_connection(self) -> bool:
        """ストリーム接続を再確立する。サブクラスでオーバーライド可能。

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

        if not self.is_running:
            return False

        # 再接続待機
        self._interruptible_sleep(self._restart_wait_seconds)

        if not self.is_running:
            return False

        # キューから次のフレームを取得して_first_frameに設定（タイムアウト付き）
        first_frame = self._wait_for_first_frame(self._restart_frame_wait_timeout)
        if first_frame is None:
            return False

        self._first_frame = first_frame

        # フレームから実際の解像度を取得
        actual_width = first_frame.frame.width
        actual_height = first_frame.frame.height

        # 新コンテナ・ストリームを作成
        try:
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

            self._last_successful_write_time = time.time()
            self._thread = threading.Thread(target=self._write_frames, daemon=True)
            self._thread.start()
            print("StreamWriter再接続に成功しました")
            return True
        except Exception as e:
            print(f"StreamWriter再接続エラー: {str(e)}")
            return False

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
