import os
import tempfile
import time
from pathlib import Path

import av
import numpy as np
import pytest
from dotenv import load_dotenv

from pyav_wrapper import (
    StreamListener,
    StreamWriter,
    WrappedAudioFrame,
    WrappedVideoFrame,
)

load_dotenv()

SRT_URL = os.getenv("SRT_URL")
SRT_OUTPUT_URL = os.getenv("SRT_OUTPUT_URL")


def check_srt_available() -> bool:
    """SRTプロトコルが利用可能か確認"""
    if SRT_URL is None:
        return False
    try:
        container = av.open(SRT_URL, timeout=15.0)
        container.close()
        return True
    except Exception as e:
        print(f"SRT check failed: {type(e).__name__}: {e}")
        return False


SRT_AVAILABLE = check_srt_available()


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

            assert writer.running is False


@pytest.mark.skipif(
    SRT_OUTPUT_URL is None, reason="SRT_OUTPUT_URL環境変数が設定されていません"
)
class TestStreamWriterSRTIntegration:
    """SRT出力統合テスト"""

    def test_srt_output_video_only(self):
        """映像のみをSRTへ送信"""
        writer = StreamWriter(
            url=SRT_OUTPUT_URL,
            width=1024,
            height=576,
            fps=30,
        )

        print(f"\nSRT送信開始: {SRT_OUTPUT_URL}")

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

    def test_srt_output_video_and_audio(self):
        """映像と音声をSRTへ送信"""
        writer = StreamWriter(
            url=SRT_OUTPUT_URL,
            width=1024,
            height=576,
            fps=30,
        )

        print(f"\nSRT送信開始: {SRT_OUTPUT_URL}")

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


@pytest.mark.skipif(
    not SRT_AVAILABLE, reason="SRTストリームに接続できません（SRT_URL未設定またはプロトコル未対応）"
)
@pytest.mark.skipif(
    SRT_OUTPUT_URL is None, reason="SRT_OUTPUT_URL環境変数が設定されていません"
)
class TestStreamWriterWithListenerIntegration:
    """StreamListener受信→グレースケール変換→StreamWriter送信の統合テスト"""

    def test_listener_to_writer_grayscale_stream(self):
        """StreamListenerで受信→グレースケール変換→StreamWriterでSRTへ送信"""
        duration = 30.0

        # StreamListenerでSRTストリームを受信
        listener = StreamListener(SRT_URL)

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
            url=SRT_OUTPUT_URL,
            width=width,
            height=height,
            fps=30,
        )

        start_time = time.time()
        video_frame_count = 0
        audio_frame_count = 0

        print(f"\nSRT送信開始: {SRT_OUTPUT_URL}")
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
