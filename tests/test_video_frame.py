import av
import numpy as np

from pyav_wrapper import WrappedVideoFrame


class TestWrappedVideoFrameBasic:
    """WrappedVideoFrame基本機能のテスト"""

    def test_wrap_video_frame(self, dummy_video_frame: av.VideoFrame):
        """VideoFrameをラップできる"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        assert wrapped is not None

    def test_get_original_frame(self, dummy_video_frame: av.VideoFrame):
        """元のAVFrameを取得できる"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        assert wrapped.frame is dummy_video_frame

    def test_get_buffer(self, dummy_video_frame: av.VideoFrame):
        """バッファ（numpy配列）を取得できる"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        buffer = wrapped.get_buffer()
        assert isinstance(buffer, np.ndarray)

    def test_get_buffer_correct_shape(self, dummy_video_frame: av.VideoFrame):
        """取得したバッファの形状が正しい"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        buffer = wrapped.get_buffer()
        # YUV420pの場合、to_ndarrayはrgb24に変換するとheight x width x 3になる
        # ここではYUV形式のまま取得することを想定
        assert buffer.shape[0] == dummy_video_frame.height
        assert buffer.shape[1] == dummy_video_frame.width

    def test_set_buffer(self, dummy_video_frame: av.VideoFrame):
        """バッファを上書きできる"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        original_buffer = wrapped.get_buffer().copy()
        new_buffer = np.zeros_like(original_buffer)
        wrapped.set_buffer(new_buffer)
        # 例外が発生しなければOK

    def test_set_buffer_modifies_frame(self, dummy_video_frame: av.VideoFrame):
        """上書き後、フレーム内容が変更されている"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        original_buffer = wrapped.get_buffer().copy()
        new_buffer = np.full_like(original_buffer, 255)
        wrapped.set_buffer(new_buffer)
        modified_buffer = wrapped.get_buffer()
        assert not np.array_equal(original_buffer, modified_buffer)


class TestWrappedVideoFramePlanes:
    """WrappedVideoFrame Plane操作のテスト"""

    def test_get_planes(self, dummy_video_frame: av.VideoFrame):
        """各planeを個別に取得できる"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        planes = wrapped.get_planes()
        assert isinstance(planes, list)
        assert all(isinstance(p, np.ndarray) for p in planes)

    def test_get_planes_yuv420p(self, dummy_video_frame: av.VideoFrame):
        """YUV420pで3つのplaneが取得できる"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        planes = wrapped.get_planes()
        assert len(planes) == 3
        # Y plane: full resolution
        assert planes[0].shape == (480, 640)
        # U, V planes: half resolution
        assert planes[1].shape == (240, 320)
        assert planes[2].shape == (240, 320)

    def test_set_planes(self, dummy_video_frame: av.VideoFrame):
        """各planeを個別に上書きできる"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        original_planes = wrapped.get_planes()
        original_y = original_planes[0].copy()

        # 新しいplaneデータを作成（全て0に）
        new_planes = [np.zeros_like(p) for p in original_planes]
        wrapped.set_planes(new_planes)

        modified_planes = wrapped.get_planes()
        assert not np.array_equal(original_y, modified_planes[0])
