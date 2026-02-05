import subprocess
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from pyav_wrapper import (
    StreamListener,
    StreamWriter,
    WrappedAudioFrame,
    WrappedVideoFrame,
)


def create_test_video_frame(width: int = 1280, height: int = 720) -> WrappedVideoFrame:
    """テスト用のWrappedVideoFrameを作成"""
    frame = av.VideoFrame(width, height, "yuv420p")
    for plane in frame.planes:
        data = np.full(plane.buffer_size, 128, dtype=np.uint8)
        plane.update(data)
    return WrappedVideoFrame(frame)


def create_test_audio_frame(
    samples: int = 1024, sample_rate: int = 48000
) -> WrappedAudioFrame:
    """テスト用のWrappedAudioFrameを作成"""
    audio_data = np.zeros((2, samples), dtype=np.float32)
    frame = av.AudioFrame.from_ndarray(audio_data, format="fltp", layout="stereo")
    frame.sample_rate = sample_rate
    return WrappedAudioFrame(frame)


class TestStreamWriterInit:
    """StreamWriter初期化のテスト"""

    def test_init_with_file_url(self):
        """ファイルURLで初期化できる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1024,
                height=576,
                fps=30,
            )
            assert writer is not None
            assert writer.width == 1024
            assert writer.height == 576
            assert writer.fps == 30
            writer.stop()


class TestStreamWriterVideoQueue:
    """StreamWriter映像キューのテスト"""

    def test_enqueue_video_frame(self):
        """映像フレームをエンキューできる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1024,
                height=576,
                fps=30,
            )

            frame = create_test_video_frame(1280, 720)
            writer.enqueue_video_frame(frame)

            # start_processingが最初のフレームを受信してスレッドを起動するまで待機
            time.sleep(2.0)

            # スレッドが動作していることを確認
            assert writer._thread is not None
            writer.stop()

    def test_enqueue_video_frames(self):
        """複数の映像フレームをエンキューできる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1024,
                height=576,
                fps=30,
            )

            frames = [create_test_video_frame(1280, 720) for _ in range(5)]
            writer.enqueue_video_frames(frames)

            writer.stop()


class TestStreamWriterAudioQueue:
    """StreamWriter音声キューのテスト"""

    def test_enqueue_audio_frame(self):
        """音声フレームをエンキューできる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1024,
                height=576,
                fps=30,
            )

            frame = create_test_audio_frame()
            writer.enqueue_audio_frame(frame)

            writer.stop()

    def test_enqueue_audio_frames(self):
        """複数の音声フレームをエンキューできる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1024,
                height=576,
                fps=30,
            )

            frames = [create_test_audio_frame() for _ in range(5)]
            writer.enqueue_audio_frames(frames)

            writer.stop()


class TestStreamWriterControl:
    """StreamWriter制御のテスト"""

    def test_stop(self):
        """stopで処理を停止できる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1024,
                height=576,
                fps=30,
            )

            time.sleep(0.5)
            writer.stop()

            assert writer.is_running is False


@pytest.mark.srt
class TestStreamWriterSRTIntegration:
    """SRT出力統合テスト"""

    def test_srt_output_video_only(self, srt_output_url):
        """映像のみをSRTへ送信"""
        writer = StreamWriter(
            url=srt_output_url,
            width=1024,
            height=576,
            fps=30,
        )

        print(f"\nSRT送信開始: {srt_output_url}")

        start_time = time.time()
        duration = 10.0
        frame_count = 0

        while time.time() - start_time < duration:
            frame = create_test_video_frame(1280, 720)
            frame.frame.pts = frame_count
            writer.enqueue_video_frame(frame)
            frame_count += 1
            time.sleep(1 / 30)

        writer.stop()

        print(f"送信フレーム数: {frame_count}")
        assert frame_count > 0

    def test_srt_output_video_and_audio(self, srt_output_url):
        """映像と音声をSRTへ送信"""
        writer = StreamWriter(
            url=srt_output_url,
            width=1024,
            height=576,
            fps=30,
        )

        print(f"\nSRT送信開始: {srt_output_url}")

        start_time = time.time()
        duration = 10.0
        video_frame_count = 0
        audio_frame_count = 0

        while time.time() - start_time < duration:
            video_frame = create_test_video_frame(1280, 720)
            video_frame.frame.pts = video_frame_count
            writer.enqueue_video_frame(video_frame)
            video_frame_count += 1

            audio_frame = create_test_audio_frame()
            audio_frame.frame.pts = audio_frame_count * 1024
            writer.enqueue_audio_frame(audio_frame)
            audio_frame_count += 1

            time.sleep(1 / 30)

        writer.stop()

        print(f"映像フレーム数: {video_frame_count}")
        print(f"音声フレーム数: {audio_frame_count}")
        assert video_frame_count > 0
        assert audio_frame_count > 0


@pytest.mark.srt
class TestStreamWriterWithListenerIntegration:
    """StreamListener受信→グレースケール変換→StreamWriter送信の統合テスト"""

    def test_listener_to_writer_grayscale_stream(self, srt_source_url, srt_output_url):
        """StreamListenerで受信→グレースケール変換→StreamWriterでSRTへ送信"""
        duration = 10.0

        # StreamListenerでSRTストリームを受信
        listener = StreamListener(srt_source_url, width=640, height=480)

        time.sleep(1.0)

        # 最初の映像フレームを取得して解像度を確認
        first_frame = None
        while first_frame is None and listener.is_running:
            if len(listener.video_queue) > 0:
                first_frame = listener.video_queue[0]
            time.sleep(0.1)

        assert first_frame is not None, "映像フレームを受信できませんでした"

        width = first_frame.frame.width
        height = first_frame.frame.height

        # StreamWriterを初期化（SRT出力）
        writer = StreamWriter(
            url=srt_output_url,
            width=width,
            height=height,
            fps=30,
        )

        start_time = time.time()
        video_frame_count = 0
        audio_frame_count = 0

        print(f"\nSRT送信開始: {srt_output_url}")
        print(f"解像度: {width}x{height}")

        last_print_time = 0

        while time.time() - start_time < duration:
            # 映像フレームの処理
            with listener.video_queue_lock:
                if len(listener.video_queue) > 0:
                    wrapped_frame = listener.video_queue.popleft()
                else:
                    wrapped_frame = None

            if wrapped_frame is not None:
                # グレースケール変換（U, Vプレーンを128に）
                planes = wrapped_frame.get_planes()

                if len(planes) > 1:
                    planes[1][:] = 128
                if len(planes) > 2:
                    planes[2][:] = 128

                # 元のフレームに書き戻す
                wrapped_frame.set_planes(planes)

                # StreamWriterのキューに追加
                writer.enqueue_video_frame(wrapped_frame)
                video_frame_count += 1

            # 音声フレームの処理
            with listener.audio_queue_lock:
                audio_frames_to_process = list(listener.audio_queue)
                listener.audio_queue.clear()

            for wrapped_audio in audio_frames_to_process:
                writer.enqueue_audio_frame(wrapped_audio)
                audio_frame_count += 1

            # 進捗表示（5秒ごと）
            elapsed = time.time() - start_time
            if int(elapsed) != last_print_time and int(elapsed) % 5 == 0:
                print(
                    f"  経過: {int(elapsed)}秒, 映像: {video_frame_count}フレーム, 音声: {audio_frame_count}フレーム"
                )
                last_print_time = int(elapsed)

            time.sleep(0.001)

        listener.stop()
        writer.stop()

        print(f"\nSRT送信完了")
        print(f"映像フレーム数: {video_frame_count}")
        print(f"音声フレーム数: {audio_frame_count}")

        assert video_frame_count > 0, "映像フレームが送信されませんでした"
        assert audio_frame_count > 0, "音声フレームが送信されませんでした"

    def test_listener_to_writer_crop_stream(self, srt_source_url, srt_output_url):
        """StreamListenerで受信→中央80%クロップ→StreamWriterでSRTへ送信"""
        duration = 10.0

        # StreamListenerでSRTストリームを受信
        listener = StreamListener(srt_source_url, width=640, height=480)

        time.sleep(1.0)

        # 最初の映像フレームを取得して解像度を確認
        first_frame = None
        while first_frame is None and listener.is_running:
            if len(listener.video_queue) > 0:
                first_frame = listener.video_queue[0]
            time.sleep(0.1)

        assert first_frame is not None, "映像フレームを受信できませんでした"

        original_width = first_frame.frame.width
        original_height = first_frame.frame.height

        # 最初の音声フレームを取得してサンプルレートを確認
        first_audio = None
        while first_audio is None and listener.is_running:
            if len(listener.audio_queue) > 0:
                first_audio = listener.audio_queue[0]
            time.sleep(0.1)

        assert first_audio is not None, "音声フレームを受信できませんでした"

        input_sample_rate = first_audio.frame.sample_rate
        input_audio_layout = first_audio.frame.layout.name

        # クロップ後のサイズを計算（80%）
        crop_ratio = 0.8
        crop_width = int(original_width * crop_ratio)
        crop_height = int(original_height * crop_ratio)
        # 2の倍数に調整
        crop_width = crop_width - (crop_width % 2)
        crop_height = crop_height - (crop_height % 2)

        print(f"\n元の解像度: {original_width}x{original_height}")
        print(f"クロップ後の解像度: {crop_width}x{crop_height}")
        print(f"入力音声サンプルレート: {input_sample_rate}Hz")
        print(f"入力音声レイアウト: {input_audio_layout}")

        # StreamWriterを初期化（クロップ後のサイズで、入力と同じサンプルレートを使用）
        writer = StreamWriter(
            url=srt_output_url,
            width=crop_width,
            height=crop_height,
            fps=30,
            sample_rate=input_sample_rate,
            audio_layout=input_audio_layout,
        )

        start_time = time.time()
        video_frame_count = 0
        audio_frame_count = 0

        print(f"SRT送信開始: {srt_output_url}")

        last_print_time = 0

        while time.time() - start_time < duration:
            # 映像フレームの処理
            with listener.video_queue_lock:
                if len(listener.video_queue) > 0:
                    wrapped_frame = listener.video_queue.popleft()
                else:
                    wrapped_frame = None

            if wrapped_frame is not None:
                # 中央80%クロップ
                cropped_frame = wrapped_frame.crop_center(ratio=crop_ratio)

                # StreamWriterのキューに追加
                writer.enqueue_video_frame(cropped_frame)
                video_frame_count += 1

            # 音声フレームの処理
            with listener.audio_queue_lock:
                audio_frames_to_process = list(listener.audio_queue)
                listener.audio_queue.clear()

            for wrapped_audio in audio_frames_to_process:
                writer.enqueue_audio_frame(wrapped_audio)
                audio_frame_count += 1

            # 進捗表示（5秒ごと）
            elapsed = time.time() - start_time
            if int(elapsed) != last_print_time and int(elapsed) % 5 == 0:
                print(
                    f"  経過: {int(elapsed)}秒, 映像: {video_frame_count}フレーム, 音声: {audio_frame_count}フレーム"
                )
                last_print_time = int(elapsed)

            time.sleep(0.001)

        listener.stop()
        writer.stop()

        print(f"\nSRT送信完了")
        print(f"映像フレーム数: {video_frame_count}")
        print(f"音声フレーム数: {audio_frame_count}")

        assert video_frame_count > 0, "映像フレームが送信されませんでした"
        assert audio_frame_count > 0, "音声フレームが送信されませんでした"


@pytest.mark.srt
class TestStreamWriterFileOutput:
    """StreamWriterファイル出力テスト（音声確認用）"""

    def test_listener_to_writer_file_output(self, srt_source_url):
        """StreamListenerで受信→StreamWriterでファイルに書き出し（音声確認用）"""
        output_file = Path(__file__).parent / "output_stream_writer.ts"
        duration = 10.0

        listener = StreamListener(srt_source_url, width=640, height=480)
        time.sleep(1.0)

        # 最初の映像フレームを取得
        first_frame = None
        while first_frame is None and listener.is_running:
            if len(listener.video_queue) > 0:
                first_frame = listener.video_queue[0]
            time.sleep(0.1)

        assert first_frame is not None, "映像フレームを受信できませんでした"

        # 最初の音声フレームを取得
        first_audio = None
        while first_audio is None and listener.is_running:
            if len(listener.audio_queue) > 0:
                first_audio = listener.audio_queue[0]
            time.sleep(0.1)

        assert first_audio is not None, "音声フレームを受信できませんでした"

        width = first_frame.frame.width
        height = first_frame.frame.height
        sample_rate = first_audio.frame.sample_rate
        audio_layout = first_audio.frame.layout.name

        print(f"\n解像度: {width}x{height}")
        print(f"サンプルレート: {sample_rate}Hz")
        print(f"音声レイアウト: {audio_layout}")

        writer = StreamWriter(
            url=str(output_file),
            width=width,
            height=height,
            fps=30,
            sample_rate=sample_rate,
            audio_layout=audio_layout,
        )

        start_time = time.time()
        video_count = 0
        audio_count = 0

        while time.time() - start_time < duration:
            # バッチ処理: 映像30フレーム分を取り出す
            video_frames = listener.pop_all_video_queue()
            audio_frames = listener.pop_all_audio_queue()

            if len(video_frames) == 0 and len(audio_frames) == 0:
                # バッチが溜まるまで待機
                time.sleep(0.1)
                continue

            # パイプライン処理（ここでは単純にenqueue）
            for frame in video_frames:
                writer.enqueue_video_frame(frame)
                video_count += 1

            for af in audio_frames:
                writer.enqueue_audio_frame(af)
                audio_count += 1

        listener.stop()
        writer.stop()
        time.sleep(1.0)  # フラッシュ完了を待つ

        print(f"映像フレーム数: {video_count}")
        print(f"音声フレーム数: {audio_count}")
        print(f"出力ファイル: {output_file}")

        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # ffprobeで音声情報を確認
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=duration,sample_rate,channels",
             "-of", "csv=p=0", str(output_file)],
            capture_output=True, text=True
        )
        print(f"ffprobe結果: {result.stdout.strip()}")

        # 動画の長さも確認
        result2 = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=duration",
             "-of", "csv=p=0", str(output_file)],
            capture_output=True, text=True
        )
        print(f"映像duration: {result2.stdout.strip()}")


class TestStreamWriterLastFrameReuse:
    """古いフレーム再利用機能のテスト"""

    def test_last_video_frame_initialized_as_none(self):
        """_last_video_frameが初期状態でNoneである"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )
            assert writer._last_video_frame is None
            writer.stop()

    def test_last_video_frame_saved_after_processing(self):
        """生データがあるフレームを処理後、_last_video_frameに保存される"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )
            # スレッドを停止して直接テスト
            writer.stop()

            # 生データありのフレームをキューに入れて直接処理
            frame = create_test_video_frame(1280, 720)
            frame.frame.pts = 100
            writer.video_queue.put(frame)
            writer._process_video_frame()

            assert writer._last_video_frame is not None

    def test_bad_frame_replaced_with_last_frame_data(self):
        """is_bad_frameがTrueのフレームは、最後のフレームの生データで差し替えられる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )
            # スレッドを停止して直接テスト
            writer.stop()

            # 1. 正常なフレームを処理（_last_video_frameに保存される）
            frame1 = create_test_video_frame(1280, 720)
            frame1.frame.pts = 100
            # Y planeを特定の値（200）で埋める
            y_plane = frame1.frame.planes[0]
            y_data = np.full(y_plane.buffer_size, 200, dtype=np.uint8)
            y_plane.update(y_data)
            writer.video_queue.put(frame1)
            writer._process_video_frame()

            # 2. is_bad_frame=Trueのフレームを作成
            bad_frame = av.VideoFrame(1280, 720, "yuv420p")
            bad_frame.pts = 200  # PTSは異なる値
            wrapped_bad = WrappedVideoFrame(bad_frame)
            wrapped_bad.is_bad_frame = True

            # _process_video_frameを直接テスト
            writer.video_queue.put(wrapped_bad)
            result = writer._process_video_frame()

            # 結果の検証
            assert result is not None
            # PTSは元のフレームのまま維持される
            assert result.frame.pts == 200
            # 生データが差し替えられたことを確認（Y planeが200で埋められている）
            result_planes = result.get_planes()
            assert len(result_planes) == 3
            assert result_planes[0][0, 0] == 200


class TestStreamWriterMuxErrorRecovery:
    """muxエラー発生時にWriterスレッドが生存し続けることのテスト"""

    def test_writer_thread_survives_pts_discontinuity(self):
        """PTS不連続（Listener再接続シミュレーション）後もWriterスレッドが生存する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_pts_discontinuity.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )

            # フェーズ1: 正常なフレーム（PTS=0,3000,6000,...）を送信
            for i in range(10):
                frame = av.VideoFrame(640, 480, "yuv420p")
                for plane in frame.planes:
                    data = np.full(plane.buffer_size, 128, dtype=np.uint8)
                    plane.update(data)
                frame.pts = i * 3000
                frame.time_base = Fraction(1, 90000)
                writer.enqueue_video_frame(WrappedVideoFrame(frame))

            # スレッド起動と最初のフレーム処理を待機
            time.sleep(2.0)
            assert writer._thread is not None
            assert writer._thread.is_alive(), "フェーズ1後にWriterスレッドが停止している"

            # フェーズ2: PTS不連続（再接続シミュレーション）
            # PTSが大きな値から0に戻る
            for i in range(10):
                frame = av.VideoFrame(640, 480, "yuv420p")
                for plane in frame.planes:
                    data = np.full(plane.buffer_size, 128, dtype=np.uint8)
                    plane.update(data)
                frame.pts = i * 3000  # 0に戻る
                frame.time_base = Fraction(1, 90000)
                writer.enqueue_video_frame(WrappedVideoFrame(frame))

            time.sleep(2.0)
            assert writer._thread.is_alive(), "PTS不連続後にWriterスレッドが停止している"

            # フェーズ3: 不連続後も正常なフレームを処理できる
            for i in range(10):
                frame = av.VideoFrame(640, 480, "yuv420p")
                for plane in frame.planes:
                    data = np.full(plane.buffer_size, 128, dtype=np.uint8)
                    plane.update(data)
                frame.pts = (i + 10) * 3000
                frame.time_base = Fraction(1, 90000)
                writer.enqueue_video_frame(WrappedVideoFrame(frame))

            time.sleep(2.0)
            assert writer._thread.is_alive(), "フェーズ3後にWriterスレッドが停止している"
            assert writer.is_running is True

            writer.stop()
            assert output_file.exists()
            assert output_file.stat().st_size > 0


class TestRestartConnectionFrameWaitTimeout:
    """_restart_connectionのフレーム待ちタイムアウトに関するテスト"""

    def test_restart_connection_returns_false_when_no_frames_available(self):
        """フレーム未供給時に_restart_connection()がタイムアウトしてFalseを返す"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_frame_wait_timeout.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 999.0
            writer._restart_threshold_max = 9999.0
            writer._restart_wait_seconds = 0.1
            writer._restart_frame_wait_timeout = 2.0

            # フレームを送信してスレッド起動を待つ
            frame = create_test_video_frame(640, 480)
            frame.frame.pts = 0
            frame.frame.time_base = Fraction(1, 90000)
            writer.enqueue_video_frame(frame)
            time.sleep(2.0)
            assert writer._thread is not None
            assert writer._thread.is_alive()

            # フレームを供給せず_restart_connectionを呼ぶ
            start = time.monotonic()
            result = writer._restart_connection()
            elapsed = time.monotonic() - start

            # 戻り値がFalse、所要時間 < 5秒
            assert result is False, "_restart_connectionがFalseを返すべきです"
            assert elapsed < 5.0, f"タイムアウトまでに{elapsed:.1f}秒かかりました（5秒以内であるべき）"

            writer.stop()

    def test_restart_connection_returns_true_when_frame_available(self):
        """フレーム供給ありで_restart_connection()がTrueを返す"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_frame_wait_success.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 999.0
            writer._restart_threshold_max = 9999.0
            writer._restart_wait_seconds = 0.1
            writer._restart_frame_wait_timeout = 10.0

            # フレームを送信してスレッド起動を待つ
            frame = create_test_video_frame(640, 480)
            frame.frame.pts = 0
            frame.frame.time_base = Fraction(1, 90000)
            writer.enqueue_video_frame(frame)
            time.sleep(2.0)
            assert writer._thread is not None

            # 遅延供給スレッドで1秒後にフレーム投入
            def delayed_feed():
                time.sleep(1.0)
                f = create_test_video_frame(640, 480)
                f.frame.pts = 0
                f.frame.time_base = Fraction(1, 90000)
                writer.enqueue_video_frame(f)

            feeder = threading.Thread(target=delayed_feed, daemon=True)
            feeder.start()

            result = writer._restart_connection()
            assert result is True, "_restart_connectionがTrueを返すべきです"

            writer.stop()

    def test_monitor_retries_after_frame_wait_timeout(self):
        """フレーム待ちタイムアウト後にmonitorループが再試行し閾値が累積増加する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_monitor_retry.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 3.0
            writer._restart_wait_seconds = 0.1
            writer._restart_frame_wait_timeout = 2.0
            writer._restart_threshold_max = 20.0
            increment = writer._restart_threshold_increment
            initial_threshold = writer._restart_threshold

            # フレームを送信してスレッド起動を待つ
            frame = create_test_video_frame(640, 480)
            frame.frame.pts = 0
            frame.frame.time_base = Fraction(1, 90000)
            writer.enqueue_video_frame(frame)
            time.sleep(2.0)
            assert writer._thread is not None

            # フレームを供給せず放置 -> monitorが再接続を複数回試行
            # 1回目: 検知~4秒 + wait0.1秒 + frame_wait2秒 = ~6秒
            # 2回目: 検知~1秒(閾値超過済み) + wait0.1秒 + frame_wait2秒 = ~3秒
            # 合計 ~9秒で2回以上の再接続試行
            time.sleep(15.0)

            # 2回以上増加していることを検証
            expected_min = initial_threshold + 2 * increment
            assert writer._restart_threshold >= expected_min, \
                f"閾値が2回以上増加していません: 現在={writer._restart_threshold}, " \
                f"期待>={expected_min}"

            writer.stop()


class TestStopResponsivenessDuringRestart:
    """stop()のレスポンシブ性テスト"""

    def test_stop_completes_quickly_during_restart_wait(self):
        """_restart_wait_seconds中にstop()が迅速に完了する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_stop_responsive.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 999.0
            writer._restart_threshold_max = 9999.0
            writer._restart_wait_seconds = 10.0
            writer._restart_frame_wait_timeout = 30.0

            # フレームを送信してスレッド起動を待つ
            frame = create_test_video_frame(640, 480)
            frame.frame.pts = 0
            frame.frame.time_base = Fraction(1, 90000)
            writer.enqueue_video_frame(frame)
            time.sleep(2.0)
            assert writer._thread is not None

            # バックグラウンドで_restart_connectionを起動
            restart_thread = threading.Thread(
                target=writer._restart_connection, daemon=True
            )
            restart_thread.start()
            time.sleep(0.5)  # restartがsleepに入るのを待つ

            # stop()が迅速に完了することを検証
            start = time.monotonic()
            writer.stop()
            elapsed = time.monotonic() - start

            assert elapsed < 3.0, \
                f"stop()が{elapsed:.1f}秒かかりました（3秒以内であるべき）"

            restart_thread.join(timeout=3.0)
            assert not restart_thread.is_alive(), "再接続スレッドが終了していません"

    def test_stop_join_timeout_sufficient_for_monitor_thread(self):
        """stop()完了後にmonitorスレッドが確実に停止している"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_stop_monitor.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )

            # フレームを送信してスレッド起動を待つ
            frame = create_test_video_frame(640, 480)
            frame.frame.pts = 0
            frame.frame.time_base = Fraction(1, 90000)
            writer.enqueue_video_frame(frame)
            time.sleep(2.0)
            assert writer._monitor_thread is not None
            assert writer._monitor_thread.is_alive()

            monitor_thread = writer._monitor_thread

            writer.stop()

            assert not monitor_thread.is_alive(), \
                "stop()完了後にmonitorスレッドが停止していません"


class TestStreamWriterReconnection:
    """StreamWriter再接続機能のテスト"""

    def test_reconnection_variables_initialized(self):
        """再接続関連変数が正しく初期化される"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )

            assert writer._last_successful_write_time == 0.0
            assert writer._restart_threshold == 10.0
            assert writer._restart_threshold_increment == 1.0
            assert writer._restart_threshold_max == 20.0
            assert writer._restart_wait_seconds == 5.0
            assert writer._restart_lock is not None
            assert writer._monitor_thread is None

            writer.stop()

    def test_has_monitor_and_restart_methods(self):
        """_monitor_write_updatesと_restart_connectionメソッドが存在する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )

            assert hasattr(writer, "_monitor_write_updates")
            assert callable(writer._monitor_write_updates)
            assert hasattr(writer, "_restart_connection")
            assert callable(writer._restart_connection)

            writer.stop()

    def test_last_successful_write_time_updated_on_mux(self):
        """mux成功時に_last_successful_write_timeが更新される"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )

            # 最初のフレームを送信してスレッド起動を待つ
            frame = create_test_video_frame(1280, 720)
            frame.frame.pts = 0
            frame.frame.time_base = Fraction(1, 90000)
            writer.enqueue_video_frame(frame)

            time.sleep(2.0)

            # mux成功後、_last_successful_write_timeが0.0より大きくなる
            assert writer._last_successful_write_time > 0.0

            writer.stop()

    def test_write_frames_loop_breaks_when_container_none(self):
        """container=Noneで_write_framesループが終了する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )

            # フレームを送信してスレッド起動
            frame = create_test_video_frame(1280, 720)
            frame.frame.pts = 0
            frame.frame.time_base = Fraction(1, 90000)
            writer.enqueue_video_frame(frame)

            time.sleep(2.0)
            assert writer._thread is not None
            assert writer._thread.is_alive()

            # containerをNoneに設定
            writer.container = None

            # スレッドが終了するのを待つ
            writer._thread.join(timeout=5.0)
            assert not writer._thread.is_alive()

            writer.stop()

    def test_restart_connection_recreates_container_and_thread(self):
        """_restart_connectionがコンテナとスレッドを再作成する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_restart_direct.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            # 監視スレッドが先に_restart_connectionを呼ばないよう閾値を高く設定
            writer._restart_threshold = 999.0
            writer._restart_threshold_max = 9999.0
            writer._restart_wait_seconds = 0.1

            # フレームを送信してスレッド起動を待つ
            frame = create_test_video_frame(640, 480)
            frame.frame.pts = 0
            frame.frame.time_base = Fraction(1, 90000)
            writer.enqueue_video_frame(frame)

            time.sleep(2.0)
            assert writer._thread is not None
            assert writer._thread.is_alive()
            assert writer.container is not None
            old_thread = writer._thread
            old_container = writer.container

            # _restart_connectionは内部でcontainer=None設定→旧スレッドjoin→
            # queue.getでフレーム待ちするため、遅延供給スレッドでフレームを投入する
            # （先にenqueueすると_write_framesに消費されてしまう）
            def delayed_feed():
                time.sleep(1.0)
                f = create_test_video_frame(640, 480)
                f.frame.pts = 0
                f.frame.time_base = Fraction(1, 90000)
                writer.enqueue_video_frame(f)

            feeder = threading.Thread(target=delayed_feed, daemon=True)
            feeder.start()

            # _restart_connectionを直接呼び出す（フレーム取得までブロック）
            writer._restart_connection()

            # 再作成されたことを検証
            assert writer.container is not None, "再接続後のcontainerがNoneです"
            assert writer.container is not old_container, "containerが再作成されていません"
            assert writer._thread is not None, "再接続後のスレッドがNoneです"
            assert writer._thread.is_alive(), "再接続後のスレッドが動作していません"
            assert writer._thread is not old_thread, "スレッドが再作成されていません"

            # 再接続後にフレームを送信してmux成功を確認
            write_time_before = writer._last_successful_write_time
            for i in range(5):
                f = create_test_video_frame(640, 480)
                f.frame.pts = (i + 1) * 3000
                f.frame.time_base = Fraction(1, 90000)
                writer.enqueue_video_frame(f)

            time.sleep(2.0)
            assert writer._last_successful_write_time > write_time_before, \
                "再接続後にmuxが成功していません"

            writer.stop()

    def test_monitor_triggers_restart_on_write_stall(self):
        """フレーム供給停止後、監視スレッドが再接続をトリガーし書き込みが再開する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_monitor_restart.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            # 閾値を短くしてテストを高速化
            # __init__後すぐに設定（start_processingはフレーム待ちでブロック中）
            writer._restart_threshold = 3.0
            writer._restart_wait_seconds = 0.1
            writer._restart_threshold_max = 20.0

            # フェーズ1: 正常にフレームを送信してmux成功を確認
            for i in range(10):
                frame = create_test_video_frame(640, 480)
                frame.frame.pts = i * 3000
                frame.frame.time_base = Fraction(1, 90000)
                writer.enqueue_video_frame(frame)

            time.sleep(2.0)
            assert writer._thread is not None
            assert writer._thread.is_alive()
            assert writer._last_successful_write_time > 0.0
            old_thread = writer._thread
            initial_threshold = writer._restart_threshold  # 3.0

            # フェーズ2: フレーム供給を停止（Listener死亡シミュレーション）
            # 閾値3秒 + 監視間隔1秒 = 最大4秒で検知
            # _restart_connectionがqueue.getでフレームを待つため、遅延供給する
            def delayed_feed():
                # _restart_connectionがqueue.get()に到達する頃にフレームを投入
                # 検知(~4秒) + join(~即座) + wait(0.1秒) = ~4.5秒後
                time.sleep(5.0)
                for i in range(10):
                    f = create_test_video_frame(640, 480)
                    f.frame.pts = i * 3000
                    f.frame.time_base = Fraction(1, 90000)
                    writer.enqueue_video_frame(f)
                    time.sleep(0.05)

            feeder = threading.Thread(target=delayed_feed, daemon=True)
            feeder.start()

            # フェーズ3: 再接続完了を待つ
            # 検知(~4秒) + wait(0.1秒) + フレーム供給(5秒) + 起動 = ~8秒
            time.sleep(9.0)

            # 再接続が発生したことを検証
            assert writer._restart_threshold > initial_threshold, \
                f"閾値が増加していません: {writer._restart_threshold} (初期値: {initial_threshold})"
            assert writer.container is not None, "再接続後のcontainerがNoneです"
            assert writer._thread is not None, "再接続後のスレッドがNoneです"
            assert writer._thread.is_alive(), "再接続後のスレッドが動作していません"
            assert writer._thread is not old_thread, "スレッドが再作成されていません"

            writer.stop()


class TestStreamWriterReconnectionScenarios:
    """StreamWriter再接続: 実障害シナリオテスト

    PTS巻き戻し・コンテナ破損など、実際のListener再接続で発生する
    障害パターンを再現し、StreamWriterの再接続機構を検証する。
    """

    def _create_pts_video_frame(
        self, width: int, height: int, pts: int, time_base: Fraction = Fraction(1, 90000)
    ) -> WrappedVideoFrame:
        """PTS付きテスト用映像フレームを作成"""
        frame = av.VideoFrame(width, height, "yuv420p")
        for plane in frame.planes:
            data = np.full(plane.buffer_size, 128, dtype=np.uint8)
            plane.update(data)
        frame.pts = pts
        frame.time_base = time_base
        return WrappedVideoFrame(frame)

    def _create_pts_audio_frame(
        self, pts: int, sample_rate: int = 48000, time_base: Fraction = Fraction(1, 90000)
    ) -> WrappedAudioFrame:
        """PTS付きテスト用音声フレームを作成"""
        audio_data = np.zeros((2, 1024), dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(audio_data, format="fltp", layout="stereo")
        frame.sample_rate = sample_rate
        frame.pts = pts
        frame.time_base = time_base
        return WrappedAudioFrame(frame)

    def test_pts_reset_triggers_reconnection_and_resumes_writing(self):
        """シナリオ1: PTS巻き戻し->muxエラー連続->監視検知->再接続->正常mux再開"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_pts_reset_reconnect.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 3.0
            writer._restart_wait_seconds = 0.1

            # フェーズ1: 正常なフレーム(PTS=0,3000,...,27000)を送信
            for i in range(10):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, i * 3000)
                )

            time.sleep(2.0)
            assert writer._thread is not None
            assert writer._thread.is_alive()
            assert writer._last_successful_write_time > 0.0
            old_thread = writer._thread

            # フェーズ2: PTS=0にリセットしたフレームを大量送信(muxエラー連続)
            for i in range(20):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, i * 3000)
                )
                time.sleep(0.05)

            # フェーズ3: 監視スレッドが検知->再接続を待つ
            # 閾値3秒 + 監視間隔1秒 + wait0.1秒 + フレーム待ち
            # 再接続後のフレームも供給する
            def delayed_feed():
                time.sleep(5.0)
                for i in range(10):
                    writer.enqueue_video_frame(
                        self._create_pts_video_frame(640, 480, i * 3000)
                    )
                    time.sleep(0.05)

            feeder = threading.Thread(target=delayed_feed, daemon=True)
            feeder.start()

            time.sleep(9.0)

            # 再接続が発生したことを検証
            assert writer._thread is not None
            assert writer._thread.is_alive()
            assert writer._thread is not old_thread, "スレッドが再作成されていません"
            assert writer._restart_threshold > 3.0, "閾値が増加していません"

            writer.stop()

    def test_pts_reset_video_and_audio_triggers_reconnection(self):
        """シナリオ2: 映像+音声両方のPTS巻き戻し->再接続->両方正常化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_pts_reset_av.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 3.0
            writer._restart_wait_seconds = 0.1

            # フェーズ1: 映像+音声を正常送信
            for i in range(10):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, i * 3000)
                )
                writer.enqueue_audio_frame(
                    self._create_pts_audio_frame(i * 1920)
                )

            time.sleep(2.0)
            assert writer._thread is not None
            assert writer._thread.is_alive()
            old_thread = writer._thread

            # フェーズ2: 映像・音声ともPTS=0にリセット(muxエラー)
            for i in range(20):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, i * 3000)
                )
                writer.enqueue_audio_frame(
                    self._create_pts_audio_frame(i * 1920)
                )
                time.sleep(0.05)

            # フェーズ3: 再接続後のフレーム供給
            def delayed_feed():
                time.sleep(5.0)
                for i in range(10):
                    writer.enqueue_video_frame(
                        self._create_pts_video_frame(640, 480, i * 3000)
                    )
                    writer.enqueue_audio_frame(
                        self._create_pts_audio_frame(i * 1920)
                    )
                    time.sleep(0.05)

            feeder = threading.Thread(target=delayed_feed, daemon=True)
            feeder.start()

            time.sleep(9.0)

            # 再接続が発生したことを検証
            assert writer._thread is not None
            assert writer._thread.is_alive()
            assert writer._thread is not old_thread, "スレッドが再作成されていません"

            # 再接続後にmuxが成功していること
            write_time_after_reconnect = writer._last_successful_write_time
            assert write_time_after_reconnect > 0.0

            writer.stop()

    def test_mux_error_stops_updating_last_successful_write_time(self):
        """シナリオ3: PTS巻き戻しmuxエラー中に_last_successful_write_timeが更新停止"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_mux_error_time.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            # 監視スレッドによる再接続を抑制
            writer._restart_threshold = 999.0
            writer._restart_threshold_max = 9999.0

            # フェーズ1: 正常フレーム送信
            for i in range(10):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, i * 3000)
                )

            time.sleep(2.0)
            assert writer._last_successful_write_time > 0.0
            time_after_normal = writer._last_successful_write_time

            # フェーズ2: PTS巻き戻しフレーム送信(muxエラー)
            for i in range(10):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, i * 3000)
                )

            time.sleep(2.0)
            time_after_reset = writer._last_successful_write_time

            # muxエラー中は_last_successful_write_timeが更新されないことを確認
            # (全く同じか、ほぼ変わらないはず)
            assert abs(time_after_reset - time_after_normal) < 1.0, \
                f"muxエラー中に_last_successful_write_timeが更新されています: " \
                f"正常後={time_after_normal}, リセット後={time_after_reset}"

            writer.stop()

    def test_reconnection_while_frames_continuously_supplied(self):
        """シナリオ4: 再接続中もフレーム供給が継続する場合"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_reconnect_continuous.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 999.0
            writer._restart_threshold_max = 9999.0
            writer._restart_wait_seconds = 0.1

            # フレーム送信してスレッド起動
            writer.enqueue_video_frame(
                self._create_pts_video_frame(640, 480, 0)
            )
            time.sleep(2.0)
            assert writer._thread is not None
            assert writer._thread.is_alive()

            # バックグラウンドでフレーム供給を継続
            supply_running = threading.Event()
            supply_running.set()

            def continuous_supply():
                i = 0
                while supply_running.is_set():
                    writer.enqueue_video_frame(
                        self._create_pts_video_frame(640, 480, i * 3000)
                    )
                    i += 1
                    time.sleep(0.05)

            supplier = threading.Thread(target=continuous_supply, daemon=True)
            supplier.start()

            # 再接続を直接呼び出し
            writer._restart_connection()

            # 再接続完了後、フレームが処理されることを確認
            write_time_before = writer._last_successful_write_time
            time.sleep(2.0)
            assert writer._last_successful_write_time > write_time_before, \
                "再接続後にmuxが成功していません"
            assert writer._thread is not None
            assert writer._thread.is_alive()

            supply_running.clear()
            writer.stop()

    def test_threshold_increments_after_reconnection(self):
        """シナリオ5: 再接続のたびに閾値が段階的に増加する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_threshold_increment.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 3.0
            writer._restart_wait_seconds = 0.1

            initial_threshold = writer._restart_threshold  # 3.0

            # フレーム送信してスレッド起動
            for i in range(10):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, i * 3000)
                )

            time.sleep(2.0)
            assert writer._thread is not None
            old_thread = writer._thread

            # PTS巻き戻しで再接続をトリガー(フレーム供給停止)
            for i in range(10):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, i * 3000)
                )

            # 再接続後のフレーム供給
            def delayed_feed():
                time.sleep(5.0)
                for i in range(10):
                    writer.enqueue_video_frame(
                        self._create_pts_video_frame(640, 480, i * 3000)
                    )
                    time.sleep(0.05)

            feeder = threading.Thread(target=delayed_feed, daemon=True)
            feeder.start()

            time.sleep(9.0)

            # 閾値が増加していることを確認
            expected_threshold = initial_threshold + writer._restart_threshold_increment
            assert writer._restart_threshold >= expected_threshold, \
                f"閾値が増加していません: 現在={writer._restart_threshold}, " \
                f"期待>={expected_threshold}"
            assert writer._thread is not old_thread, "スレッドが再作成されていません"

            writer.stop()

    def test_double_pts_reset_two_reconnections_succeed(self):
        """シナリオ6: 2回連続PTS巻き戻し->2回再接続成功"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_double_reconnect.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 999.0
            writer._restart_threshold_max = 9999.0
            writer._restart_wait_seconds = 0.1

            initial_threshold = writer._restart_threshold

            # フレーム送信してスレッド起動
            writer.enqueue_video_frame(
                self._create_pts_video_frame(640, 480, 0)
            )
            time.sleep(2.0)
            assert writer._thread is not None
            first_thread = writer._thread

            # 1回目の再接続(遅延供給あり)
            def feed_for_restart():
                time.sleep(1.0)
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, 0)
                )

            feeder1 = threading.Thread(target=feed_for_restart, daemon=True)
            feeder1.start()
            writer._restart_connection()

            second_thread = writer._thread
            assert second_thread is not first_thread, "1回目の再接続でスレッドが変更されていません"

            # 正常なフレームを処理
            for i in range(5):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, (i + 1) * 3000)
                )
            time.sleep(1.0)

            # 2回目の再接続
            feeder2 = threading.Thread(target=feed_for_restart, daemon=True)
            feeder2.start()
            writer._restart_connection()

            third_thread = writer._thread
            assert third_thread is not second_thread, "2回目の再接続でスレッドが変更されていません"
            assert third_thread is not first_thread, "3つ目のスレッドが最初のスレッドと同一です"

            # 再接続後のmux成功確認
            write_time_before = writer._last_successful_write_time
            for i in range(5):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, (i + 1) * 3000)
                )
            time.sleep(2.0)
            assert writer._last_successful_write_time > write_time_before

            writer.stop()

    def test_stop_during_reconnection_no_deadlock(self):
        """シナリオ7: 再接続中にstop()呼び出し->デッドロックしない"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_stop_deadlock.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 999.0
            writer._restart_threshold_max = 9999.0
            writer._restart_wait_seconds = 0.1

            # フレーム送信してスレッド起動
            writer.enqueue_video_frame(
                self._create_pts_video_frame(640, 480, 0)
            )
            time.sleep(2.0)
            assert writer._thread is not None

            # バックグラウンドで_restart_connectionを呼ぶ(フレーム未供給->getでブロック)
            restart_thread = threading.Thread(
                target=writer._restart_connection, daemon=True
            )
            restart_thread.start()
            time.sleep(0.5)  # restartがqueue.getでブロックされるのを待つ

            # メインスレッドからstop()を呼ぶ
            stop_start = time.monotonic()
            writer.stop()
            stop_elapsed = time.monotonic() - stop_start

            # 5秒以内にstop()が完了すること(デッドロックしない)
            assert stop_elapsed < 5.0, \
                f"stop()が{stop_elapsed:.1f}秒かかりました(デッドロックの可能性)"

            # restart_threadも終了していることを確認
            restart_thread.join(timeout=3.0)
            assert not restart_thread.is_alive(), "再接続スレッドが終了していません"

    def test_container_forced_close_triggers_reconnection(self):
        """シナリオ8: containerが外部から強制close->mux失敗->再接続"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_forced_close.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 3.0
            writer._restart_wait_seconds = 0.1

            # フレーム送信してスレッド起動
            for i in range(10):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, i * 3000)
                )

            time.sleep(2.0)
            assert writer._thread is not None
            assert writer._thread.is_alive()
            old_thread = writer._thread

            # 外部からcontainerを強制close
            if writer.container is not None:
                try:
                    writer.container.close()
                except Exception:
                    pass

            # mux失敗が継続するようフレームを供給し続ける
            for i in range(20):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, (i + 10) * 3000)
                )
                time.sleep(0.05)

            # 再接続後のフレーム供給
            def delayed_feed():
                time.sleep(5.0)
                for i in range(10):
                    writer.enqueue_video_frame(
                        self._create_pts_video_frame(640, 480, i * 3000)
                    )
                    time.sleep(0.05)

            feeder = threading.Thread(target=delayed_feed, daemon=True)
            feeder.start()

            time.sleep(9.0)

            # 再接続が発生したことを検証
            assert writer._thread is not None
            assert writer._thread.is_alive()
            assert writer._thread is not old_thread, "スレッドが再作成されていません"

            writer.stop()

    def test_output_valid_after_reconnection(self):
        """シナリオ10: 再接続後の出力ファイルが有効"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_valid_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=640,
                height=480,
                fps=30,
            )
            writer._restart_threshold = 999.0
            writer._restart_threshold_max = 9999.0
            writer._restart_wait_seconds = 0.1

            # 正常フレームを送信
            writer.enqueue_video_frame(
                self._create_pts_video_frame(640, 480, 0)
            )
            time.sleep(2.0)

            # 再接続を直接呼び出し
            def feed_for_restart():
                time.sleep(1.0)
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, 0)
                )

            feeder = threading.Thread(target=feed_for_restart, daemon=True)
            feeder.start()
            writer._restart_connection()

            # 再接続後に新PTS系列でフレーム送信
            for i in range(30):
                writer.enqueue_video_frame(
                    self._create_pts_video_frame(640, 480, (i + 1) * 3000)
                )
                time.sleep(0.033)

            time.sleep(2.0)
            writer.stop()

            # 出力ファイルが有効であることを確認
            assert output_file.exists(), "出力ファイルが存在しません"
            assert output_file.stat().st_size > 0, "出力ファイルのサイズが0です"

            # ffprobeでPTSが単調増加していることを確認
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "packet=pts",
                    "-of", "csv=p=0",
                    str(output_file),
                ],
                capture_output=True,
                text=True,
            )

            if result.stdout.strip():
                pts_values = []
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if line and line != "N/A":
                        try:
                            pts_values.append(int(line))
                        except ValueError:
                            continue

                if len(pts_values) > 1:
                    # PTSが単調増加(非減少)であることを確認
                    for i in range(1, len(pts_values)):
                        assert pts_values[i] >= pts_values[i - 1], \
                            f"PTSが単調増加していません: [{i-1}]={pts_values[i-1]}, [{i}]={pts_values[i]}"
