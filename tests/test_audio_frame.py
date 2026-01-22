import av
import numpy as np

from pyav_wrapper import WrappedAudioFrame


class TestWrappedAudioFrameBasic:
    """WrappedAudioFrame基本機能のテスト"""

    def test_wrap_audio_frame(self, dummy_audio_frame: av.AudioFrame):
        """AudioFrameをラップできる"""
        wrapped = WrappedAudioFrame(dummy_audio_frame)
        assert wrapped is not None

    def test_get_original_frame(self, dummy_audio_frame: av.AudioFrame):
        """元のAVFrameを取得できる"""
        wrapped = WrappedAudioFrame(dummy_audio_frame)
        assert wrapped.frame is dummy_audio_frame

    def test_get_buffer(self, dummy_audio_frame: av.AudioFrame):
        """バッファを取得できる"""
        wrapped = WrappedAudioFrame(dummy_audio_frame)
        buffer = wrapped.get_buffer()
        assert isinstance(buffer, np.ndarray)

    def test_get_buffer_correct_shape(self, dummy_audio_frame: av.AudioFrame):
        """取得したバッファの形状が正しい（channels, samples）"""
        wrapped = WrappedAudioFrame(dummy_audio_frame)
        buffer = wrapped.get_buffer()
        # fltp形式: (channels, samples)
        assert buffer.shape == (2, 1024)

    def test_set_buffer(self, dummy_audio_frame: av.AudioFrame):
        """バッファを上書きできる"""
        wrapped = WrappedAudioFrame(dummy_audio_frame)
        original_buffer = wrapped.get_buffer().copy()
        new_buffer = np.ones_like(original_buffer) * 0.5
        wrapped.set_buffer(new_buffer)
        modified_buffer = wrapped.get_buffer()
        assert not np.array_equal(original_buffer, modified_buffer)


class TestWrappedAudioFramePlanes:
    """WrappedAudioFrame Plane操作のテスト"""

    def test_get_planes(self, dummy_audio_frame: av.AudioFrame):
        """各チャネルを個別に取得できる"""
        wrapped = WrappedAudioFrame(dummy_audio_frame)
        planes = wrapped.get_planes()
        assert isinstance(planes, list)
        assert all(isinstance(p, np.ndarray) for p in planes)

    def test_get_planes_stereo(self, dummy_audio_frame: av.AudioFrame):
        """ステレオで2つのplaneが取得できる"""
        wrapped = WrappedAudioFrame(dummy_audio_frame)
        planes = wrapped.get_planes()
        assert len(planes) == 2
        # 各チャネル: 1024 samples
        assert planes[0].shape == (1024,)
        assert planes[1].shape == (1024,)

    def test_set_planes(self, dummy_audio_frame: av.AudioFrame):
        """各チャネルを個別に上書きできる"""
        wrapped = WrappedAudioFrame(dummy_audio_frame)
        original_planes = wrapped.get_planes()
        original_ch0 = original_planes[0].copy()

        # 新しいplaneデータを作成
        new_planes = [np.ones_like(p) * 0.5 for p in original_planes]
        wrapped.set_planes(new_planes)

        modified_planes = wrapped.get_planes()
        assert not np.array_equal(original_ch0, modified_planes[0])
