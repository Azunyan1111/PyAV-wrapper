import tempfile
import time
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
        listener = StreamListener(srt_source_url)

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
        listener = StreamListener(srt_source_url)

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

        listener = StreamListener(srt_source_url)
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


class TestStreamWriterRealtime:
    """StreamWriterリアルタイムペーシング機能のテスト"""

    def test_init_realtime_clock_sets_origin(self):
        """_init_realtime_clockが基準時刻を正しく設定する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )
            writer.stop()

            # PTSとtime_baseを設定したフレームで基準時刻を初期化
            from fractions import Fraction

            frame = av.VideoFrame(1280, 720, "yuv420p")
            frame.pts = 90000
            frame.time_base = Fraction(1, 90000)

            before = time.monotonic()
            writer._init_realtime_clock(frame)
            after = time.monotonic()

            # _wall_clock_originがbefore〜afterの範囲内であること
            assert before <= writer._wall_clock_origin <= after
            # _pts_originが1.0秒であること（90000 * 1/90000）
            assert abs(writer._pts_origin - 1.0) < 0.001

    def test_init_realtime_clock_pts_none(self):
        """_init_realtime_clockでPTSがNoneの場合、_pts_originが0.0になる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )
            writer.stop()

            frame = av.VideoFrame(1280, 720, "yuv420p")
            frame.pts = None

            writer._init_realtime_clock(frame)
            assert writer._pts_origin == 0.0

    def test_pace_frame_returns_immediately_when_pts_none(self):
        """PTSがNoneの場合、_pace_frameが即座にreturnする"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )
            writer.stop()

            frame = av.VideoFrame(1280, 720, "yuv420p")
            frame.pts = None

            start = time.monotonic()
            writer._pace_frame(frame)
            elapsed = time.monotonic() - start

            assert elapsed < 0.01

    def test_pace_frame_sleeps_when_ahead(self):
        """フレームが壁時計より先行している場合、_pace_frameがスリープする"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )
            writer.stop()

            from fractions import Fraction

            # 基準フレーム（PTS=0）で初期化
            first_frame = av.VideoFrame(1280, 720, "yuv420p")
            first_frame.pts = 0
            first_frame.time_base = Fraction(1, 90000)
            writer._init_realtime_clock(first_frame)

            # 0.2秒先のフレーム（PTS=18000, time_base=1/90000 → 0.2秒）
            future_frame = av.VideoFrame(1280, 720, "yuv420p")
            future_frame.pts = 18000
            future_frame.time_base = Fraction(1, 90000)

            start = time.monotonic()
            writer._pace_frame(future_frame)
            elapsed = time.monotonic() - start

            # 約0.2秒スリープすること（許容: 0.15〜0.35秒）
            assert 0.15 <= elapsed <= 0.35

    def test_pace_frame_max_sleep_capped_at_1_second(self):
        """PTS不連続の場合、スリープが最大1秒に制限される"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_output.ts"
            writer = StreamWriter(
                url=str(output_file),
                width=1280,
                height=720,
                fps=30,
            )
            writer.stop()

            from fractions import Fraction

            # 基準フレーム（PTS=0）
            first_frame = av.VideoFrame(1280, 720, "yuv420p")
            first_frame.pts = 0
            first_frame.time_base = Fraction(1, 90000)
            writer._init_realtime_clock(first_frame)

            # 100秒先のフレーム（PTS=9000000）
            far_future_frame = av.VideoFrame(1280, 720, "yuv420p")
            far_future_frame.pts = 9000000
            far_future_frame.time_base = Fraction(1, 90000)

            start = time.monotonic()
            writer._pace_frame(far_future_frame)
            elapsed = time.monotonic() - start

            # 最大1秒に制限されること（許容: 0.9〜1.2秒）
            assert 0.9 <= elapsed <= 1.2


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
