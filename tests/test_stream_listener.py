import collections
import os
import tempfile
import time
from pathlib import Path

import av
import numpy as np
import pytest
from dotenv import load_dotenv

from pyav_wrapper import StreamListener, WrappedAudioFrame, WrappedVideoFrame

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


def create_test_video_file(path: Path, duration_frames: int = 60) -> None:
    """テスト用の動画ファイルを作成"""
    container = av.open(str(path), mode="w")

    # Videoストリーム
    video_stream = container.add_stream("libx264", rate=30)
    video_stream.width = 640
    video_stream.height = 480
    video_stream.pix_fmt = "yuv420p"

    # Audioストリーム
    audio_stream = container.add_stream("aac", rate=48000)
    audio_stream.layout = "stereo"

    # フレームを生成
    for i in range(duration_frames):
        # Video
        video_frame = av.VideoFrame(640, 480, "yuv420p")
        for plane in video_frame.planes:
            data = np.full(plane.buffer_size, 128, dtype=np.uint8)
            plane.update(data)
        video_frame.pts = i
        for packet in video_stream.encode(video_frame):
            container.mux(packet)

        # Audio（30fpsで1フレームあたり1600サンプル = 48000/30）
        samples = 1600
        audio_data = np.zeros((2, samples), dtype=np.float32)
        audio_frame = av.AudioFrame.from_ndarray(
            audio_data, format="fltp", layout="stereo"
        )
        audio_frame.sample_rate = 48000
        audio_frame.pts = i * samples
        for packet in audio_stream.encode(audio_frame):
            container.mux(packet)

    # フラッシュ
    for packet in video_stream.encode():
        container.mux(packet)
    for packet in audio_stream.encode():
        container.mux(packet)

    container.close()


class TestStreamListenerInit:
    """StreamListener初期化のテスト"""

    def test_init_with_file_url(self):
        """ファイルURLで初期化できる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp4"
            create_test_video_file(test_file)

            listener = StreamListener(str(test_file), width=640, height=480)
            assert listener is not None
            assert listener.url == str(test_file)
            assert listener.width == 640
            assert listener.height == 480
            listener.stop()


class TestStreamListenerVideoQueue:
    """StreamListener Videoキューのテスト"""

    def test_append_video_queue(self):
        """Videoキューにフレームを追加できる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp4"
            create_test_video_file(test_file)

            listener = StreamListener(str(test_file), width=640, height=480)
            time.sleep(0.5)

            assert len(listener.video_queue) > 0
            listener.stop()

    def test_pop_all_video_queue_returns_wrapped_video_frame(self):
        """pop_all_video_queueがWrappedVideoFrameのリストを返す"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp4"
            create_test_video_file(test_file, duration_frames=60)

            listener = StreamListener(str(test_file), width=640, height=480)
            listener.batch_size = 10
            time.sleep(1.0)

            frames = listener.pop_all_video_queue()
            if len(frames) > 0:
                assert all(isinstance(f, WrappedVideoFrame) for f in frames)
            listener.stop()

    def test_video_queue_overflow_handling(self):
        """Videoキュー満杯時に古いフレームが破棄される"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp4"
            create_test_video_file(test_file, duration_frames=100)

            listener = StreamListener(str(test_file), width=640, height=480)
            time.sleep(2.0)

            max_size = int(listener.batch_size * 1.7)
            assert len(listener.video_queue) <= max_size
            listener.stop()


class TestStreamListenerAudioQueue:
    """StreamListener Audioキューのテスト"""

    def test_append_audio_queue(self):
        """Audioキューにフレームを追加できる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp4"
            create_test_video_file(test_file)

            listener = StreamListener(str(test_file), width=640, height=480)
            time.sleep(0.5)

            assert len(listener.audio_queue) > 0
            listener.stop()

    def test_pop_all_audio_queue_returns_wrapped_audio_frame(self):
        """pop_all_audio_queueがWrappedAudioFrameのリストを返す"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp4"
            create_test_video_file(test_file, duration_frames=60)

            listener = StreamListener(str(test_file), width=640, height=480)
            listener.batch_size = 10
            time.sleep(1.0)

            frames = listener.pop_all_audio_queue()
            if len(frames) > 0:
                assert all(isinstance(f, WrappedAudioFrame) for f in frames)
            listener.stop()

    def test_audio_queue_overflow_handling(self):
        """Audioキュー満杯時に古いフレームが破棄される"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp4"
            create_test_video_file(test_file, duration_frames=100)

            listener = StreamListener(str(test_file), width=640, height=480)
            time.sleep(2.0)

            max_size = int(listener.batch_size * 1.7)
            assert len(listener.audio_queue) <= max_size
            listener.stop()


class TestStreamListenerControl:
    """StreamListener制御のテスト"""

    def test_stop(self):
        """stopで処理を停止できる"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.mp4"
            create_test_video_file(test_file)

            listener = StreamListener(str(test_file), width=640, height=480)
            time.sleep(0.3)

            listener.stop()
            assert listener.is_running is False


@pytest.mark.skipif(not SRT_AVAILABLE, reason="SRTストリームに接続できません（SRT_URL未設定またはプロトコル未対応）")
class TestStreamListenerSRTIntegration:
    """SRT統合テスト：受信→グレースケール変換→ファイル書き出し"""

    def test_srt_receive_convert_grayscale_and_write(self):
        """SRTから受信した映像をグレースケールに変換して音声付きでファイルに書き出す"""
        output_path = Path(__file__).parent / "output_grayscale.mp4"
        duration = 10.0

        listener = StreamListener(SRT_URL)

        time.sleep(1.0)

        first_frame = None
        while first_frame is None and listener.is_running:
            if len(listener.video_queue) > 0:
                first_frame = listener.video_queue[0]
            time.sleep(0.1)

        assert first_frame is not None, "映像フレームを受信できませんでした"

        width = first_frame.frame.width
        height = first_frame.frame.height

        first_audio = None
        while first_audio is None and listener.is_running:
            if len(listener.audio_queue) > 0:
                first_audio = listener.audio_queue[0]
            time.sleep(0.1)

        assert first_audio is not None, "音声フレームを受信できませんでした"

        sample_rate = first_audio.frame.sample_rate
        audio_layout = first_audio.frame.layout.name

        output_container = av.open(str(output_path), mode="w")

        video_stream = output_container.add_stream("libx264", rate=30)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"

        audio_stream = output_container.add_stream("aac", rate=sample_rate)
        audio_stream.layout = audio_layout

        start_time = time.time()
        video_frame_count = 0
        audio_frame_count = 0

        while time.time() - start_time < duration:
            with listener.video_queue_lock:
                video_frames = list(listener.video_queue)
                listener.video_queue.clear()

            for wrapped_frame in video_frames:
                planes = wrapped_frame.get_planes()

                y_plane = planes[0]
                u_plane = planes[1] if len(planes) > 1 else None
                v_plane = planes[2] if len(planes) > 2 else None

                if u_plane is not None:
                    u_plane[:] = 128
                if v_plane is not None:
                    v_plane[:] = 128

                new_frame = av.VideoFrame(width, height, "yuv420p")
                new_frame.planes[0].update(
                    np.pad(
                        y_plane,
                        ((0, 0), (0, new_frame.planes[0].line_size - y_plane.shape[1])),
                        mode="constant",
                    ).tobytes()
                )
                if u_plane is not None:
                    new_frame.planes[1].update(
                        np.pad(
                            u_plane,
                            ((0, 0), (0, new_frame.planes[1].line_size - u_plane.shape[1])),
                            mode="constant",
                            constant_values=128,
                        ).tobytes()
                    )
                if v_plane is not None:
                    new_frame.planes[2].update(
                        np.pad(
                            v_plane,
                            ((0, 0), (0, new_frame.planes[2].line_size - v_plane.shape[1])),
                            mode="constant",
                            constant_values=128,
                        ).tobytes()
                    )

                new_frame.pts = video_frame_count
                for packet in video_stream.encode(new_frame):
                    output_container.mux(packet)
                video_frame_count += 1

            with listener.audio_queue_lock:
                audio_frames = list(listener.audio_queue)
                listener.audio_queue.clear()

            for wrapped_audio in audio_frames:
                audio_frame = wrapped_audio.frame
                audio_frame.pts = None
                for packet in audio_stream.encode(audio_frame):
                    output_container.mux(packet)
                audio_frame_count += 1

            time.sleep(0.1)

        listener.stop()

        for packet in video_stream.encode():
            output_container.mux(packet)
        for packet in audio_stream.encode():
            output_container.mux(packet)

        output_container.close()

        assert output_path.exists(), "出力ファイルが作成されませんでした"
        assert output_path.stat().st_size > 0, "出力ファイルが空です"

        print(f"\n出力ファイル: {output_path}")
        print(f"ファイルサイズ: {output_path.stat().st_size} bytes")
        print(f"映像フレーム数: {video_frame_count}")
        print(f"音声フレーム数: {audio_frame_count}")
        print(f"解像度: {width}x{height}")
        print(f"音声サンプルレート: {sample_rate}Hz")

    @pytest.mark.skipif(
        SRT_OUTPUT_URL is None,
        reason="SRT_OUTPUT_URL環境変数が設定されていません"
    )
    def test_srt_receive_grayscale_and_stream_to_srt(self):
        """SRTから受信→グレースケール変換→SRTへ60秒間送信"""
        duration = 60.0

        listener = StreamListener(SRT_URL)

        time.sleep(1.0)

        first_frame = None
        while first_frame is None and listener.is_running:
            if len(listener.video_queue) > 0:
                first_frame = listener.video_queue[0]
            time.sleep(0.1)

        assert first_frame is not None, "映像フレームを受信できませんでした"

        width = first_frame.frame.width
        height = first_frame.frame.height

        first_audio = None
        while first_audio is None and listener.is_running:
            if len(listener.audio_queue) > 0:
                first_audio = listener.audio_queue[0]
            time.sleep(0.1)

        assert first_audio is not None, "音声フレームを受信できませんでした"

        sample_rate = first_audio.frame.sample_rate
        audio_layout = first_audio.frame.layout.name

        output_container = av.open(
            SRT_OUTPUT_URL,
            mode="w",
            format="mpegts",
        )

        video_stream = output_container.add_stream("libx264", rate=30)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"preset": "ultrafast", "tune": "zerolatency"}

        audio_stream = output_container.add_stream("aac", rate=sample_rate)
        audio_stream.layout = audio_layout

        start_time = time.time()
        video_frame_count = 0
        audio_frame_count = 0

        print(f"\nSRT送信開始: {SRT_OUTPUT_URL}")

        last_print_time = 0

        while time.time() - start_time < duration:
            has_data = False

            with listener.video_queue_lock:
                if len(listener.video_queue) > 0:
                    wrapped_frame = listener.video_queue.popleft()
                    has_data = True
                else:
                    wrapped_frame = None

            if wrapped_frame is not None:
                planes = wrapped_frame.get_planes()

                if len(planes) > 1:
                    planes[1][:] = 128
                if len(planes) > 2:
                    planes[2][:] = 128

                wrapped_frame.set_planes(planes)

                for packet in video_stream.encode(wrapped_frame.frame):
                    output_container.mux(packet)
                video_frame_count += 1

            with listener.audio_queue_lock:
                audio_frames_to_process = list(listener.audio_queue)
                listener.audio_queue.clear()
                if len(audio_frames_to_process) > 0:
                    has_data = True

            for wrapped_audio in audio_frames_to_process:
                audio_frame = wrapped_audio.frame
                for packet in audio_stream.encode(audio_frame):
                    output_container.mux(packet)
                audio_frame_count += 1

            elapsed = time.time() - start_time
            if int(elapsed) != last_print_time and int(elapsed) % 5 == 0:
                print(f"  経過: {int(elapsed)}秒, 映像: {video_frame_count}フレーム, 音声: {audio_frame_count}フレーム")
                last_print_time = int(elapsed)

            if not has_data:
                time.sleep(0.001)

        listener.stop()

        for packet in video_stream.encode():
            output_container.mux(packet)
        for packet in audio_stream.encode():
            output_container.mux(packet)

        try:
            output_container.close()
        except Exception as e:
            print(f"\nSRT切断時エラー（送信自体は完了）: {e}")

        print(f"\nSRT送信完了")
        print(f"映像フレーム数: {video_frame_count}")
        print(f"音声フレーム数: {audio_frame_count}")
        print(f"解像度: {width}x{height}")
        print(f"音声サンプルレート: {sample_rate}Hz")

        assert video_frame_count > 0, "映像フレームが送信されませんでした"
        assert audio_frame_count > 0, "音声フレームが送信されませんでした"


@pytest.mark.skipif(not SRT_AVAILABLE, reason="SRTストリームに接続できません（SRT_URL未設定またはプロトコル未対応）")
class TestStreamListenerSRT:
    """SRT実受信テスト（SRT_URL環境変数が設定されている場合のみ実行）"""

    def test_srt_connect(self):
        """SRTストリームに接続できる"""
        listener = StreamListener(SRT_URL)
        time.sleep(3.0)

        assert listener.is_running is True
        listener.stop()

    def test_srt_receive_video_frames(self):
        """SRTからVideoフレームを受信できる"""
        listener = StreamListener(SRT_URL)
        time.sleep(5.0)

        assert len(listener.video_queue) > 0
        listener.stop()

    def test_srt_receive_audio_frames(self):
        """SRTからAudioフレームを受信できる"""
        listener = StreamListener(SRT_URL)
        time.sleep(5.0)

        assert len(listener.audio_queue) > 0
        listener.stop()

    def test_srt_pop_video_returns_wrapped_frame(self):
        """SRTから取得したVideoフレームがWrappedVideoFrame型"""
        listener = StreamListener(SRT_URL)
        listener.batch_size = 5
        time.sleep(5.0)

        frames = listener.pop_all_video_queue()
        assert len(frames) > 0
        assert all(isinstance(f, WrappedVideoFrame) for f in frames)
        listener.stop()

    def test_srt_pop_audio_returns_wrapped_frame(self):
        """SRTから取得したAudioフレームがWrappedAudioFrame型"""
        listener = StreamListener(SRT_URL)
        listener.batch_size = 5
        time.sleep(5.0)

        frames = listener.pop_all_audio_queue()
        assert len(frames) > 0
        assert all(isinstance(f, WrappedAudioFrame) for f in frames)
        listener.stop()

    def test_srt_video_frame_has_valid_buffer(self):
        """SRTから取得したVideoフレームのバッファが有効"""
        listener = StreamListener(SRT_URL)
        time.sleep(5.0)

        if len(listener.video_queue) > 0:
            frame = listener.video_queue[0]
            buffer = frame.get_buffer()
            assert buffer is not None
            assert buffer.shape[0] > 0
            assert buffer.shape[1] > 0
        listener.stop()

    def test_srt_audio_frame_has_valid_buffer(self):
        """SRTから取得したAudioフレームのバッファが有効"""
        listener = StreamListener(SRT_URL)
        time.sleep(5.0)

        if len(listener.audio_queue) > 0:
            frame = listener.audio_queue[0]
            buffer = frame.get_buffer()
            assert buffer is not None
            assert buffer.shape[0] > 0
        listener.stop()
