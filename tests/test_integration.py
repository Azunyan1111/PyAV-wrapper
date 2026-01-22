import tempfile
from pathlib import Path

import av
import numpy as np

from pyav_wrapper import WrappedAudioFrame, WrappedVideoFrame


class TestVideoIntegration:
    """VideoFrame統合テスト"""

    def test_video_read_modify_encode(self):
        """VideoFrame取得→加工→エンコードが動作する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            # 出力コンテナを作成
            output_container = av.open(str(output_path), mode="w")
            output_stream = output_container.add_stream("libx264", rate=30)
            output_stream.width = 640
            output_stream.height = 480
            output_stream.pix_fmt = "yuv420p"

            # ダミーフレームを作成して加工
            for i in range(10):
                frame = av.VideoFrame(640, 480, "yuv420p")

                # Y planeを初期化
                y_plane = frame.planes[0]
                y_data = np.full(y_plane.buffer_size, 128, dtype=np.uint8)
                y_plane.update(y_data)

                # U, V planeを初期化
                for plane in frame.planes[1:]:
                    data = np.full(plane.buffer_size, 128, dtype=np.uint8)
                    plane.update(data)

                # ラッパーで加工
                wrapped = WrappedVideoFrame(frame)
                buffer = wrapped.get_buffer()
                # 明るさを変更（Y値を増加）
                buffer = np.clip(buffer + 50, 0, 255).astype(np.uint8)
                wrapped.set_buffer(buffer)

                # エンコード
                frame.pts = i
                for packet in output_stream.encode(wrapped.frame):
                    output_container.mux(packet)

            # フラッシュ
            for packet in output_stream.encode():
                output_container.mux(packet)

            output_container.close()

            # 出力ファイルが存在することを確認
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_frame_metadata_preserved(self, dummy_video_frame: av.VideoFrame):
        """pts等のメタデータが保持される"""
        dummy_video_frame.pts = 12345

        wrapped = WrappedVideoFrame(dummy_video_frame)
        buffer = wrapped.get_buffer()
        new_buffer = np.zeros_like(buffer)
        wrapped.set_buffer(new_buffer)

        # ptsが保持されていることを確認
        assert wrapped.frame.pts == 12345


class TestAudioIntegration:
    """AudioFrame統合テスト"""

    def test_audio_read_modify_encode(self):
        """AudioFrame取得→加工→エンコードが動作する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.aac"

            # 出力コンテナを作成
            output_container = av.open(str(output_path), mode="w")
            output_stream = output_container.add_stream("aac", rate=48000)
            output_stream.layout = "stereo"

            pts = 0
            # ダミーフレームを作成して加工
            for i in range(10):
                # サイン波を生成
                samples = 1024
                t = np.linspace(0, 0.02, samples, dtype=np.float32)
                audio_data = np.stack([
                    np.sin(2 * np.pi * 440 * t),  # 左チャネル: 440Hz
                    np.sin(2 * np.pi * 880 * t),  # 右チャネル: 880Hz
                ])

                frame = av.AudioFrame.from_ndarray(
                    audio_data, format="fltp", layout="stereo"
                )
                frame.sample_rate = 48000

                # ラッパーで加工
                wrapped = WrappedAudioFrame(frame)
                buffer = wrapped.get_buffer()
                # 音量を半分に
                buffer = buffer * 0.5
                wrapped.set_buffer(buffer.astype(np.float32))

                # エンコード
                wrapped.frame.pts = pts
                for packet in output_stream.encode(wrapped.frame):
                    output_container.mux(packet)
                pts += samples

            # フラッシュ
            for packet in output_stream.encode():
                output_container.mux(packet)

            output_container.close()

            # 出力ファイルが存在することを確認
            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_audio_metadata_preserved(self, dummy_audio_frame: av.AudioFrame):
        """pts等のメタデータが保持される"""
        dummy_audio_frame.pts = 54321

        wrapped = WrappedAudioFrame(dummy_audio_frame)
        buffer = wrapped.get_buffer()
        new_buffer = buffer * 0.5
        wrapped.set_buffer(new_buffer.astype(np.float32))

        # ptsが保持されていることを確認
        assert wrapped.frame.pts == 54321
