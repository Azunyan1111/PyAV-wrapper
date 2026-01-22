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


class TestWrappedVideoFrameCrop:
    """WrappedVideoFrame クロップ機能のテスト"""

    def test_crop_center_default_ratio(self, dummy_video_frame: av.VideoFrame):
        """デフォルト比率（80%）で中央クロップできる"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        cropped = wrapped.crop_center()

        # 640x480の80% = 512x384
        assert cropped.frame.width == 512
        assert cropped.frame.height == 384

    def test_crop_center_custom_ratio(self, dummy_video_frame: av.VideoFrame):
        """カスタム比率で中央クロップできる"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        cropped = wrapped.crop_center(ratio=0.5)

        # 640x480の50% = 320x240
        assert cropped.frame.width == 320
        assert cropped.frame.height == 240

    def test_crop_center_preserves_pts(self, dummy_video_frame: av.VideoFrame):
        """クロップ後もPTSが保持される"""
        dummy_video_frame.pts = 12345
        wrapped = WrappedVideoFrame(dummy_video_frame)
        cropped = wrapped.crop_center()

        assert cropped.frame.pts == 12345

    def test_crop_center_preserves_time_base(self, dummy_video_frame: av.VideoFrame):
        """クロップ後もtime_baseが保持される"""
        from fractions import Fraction

        dummy_video_frame.time_base = Fraction(1, 30)
        wrapped = WrappedVideoFrame(dummy_video_frame)
        cropped = wrapped.crop_center()

        assert cropped.frame.time_base == Fraction(1, 30)

    def test_crop_center_plane_sizes(self, dummy_video_frame: av.VideoFrame):
        """クロップ後のplaneサイズが正しい（YUV420p）"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        cropped = wrapped.crop_center()  # 512x384

        planes = cropped.get_planes()
        assert len(planes) == 3
        # Y plane: full resolution
        assert planes[0].shape == (384, 512)
        # U, V planes: half resolution
        assert planes[1].shape == (192, 256)
        assert planes[2].shape == (192, 256)

    def test_crop_center_invalid_ratio_zero(self, dummy_video_frame: av.VideoFrame):
        """ratio=0でValueErrorが発生する"""
        import pytest

        wrapped = WrappedVideoFrame(dummy_video_frame)
        with pytest.raises(ValueError):
            wrapped.crop_center(ratio=0.0)

    def test_crop_center_invalid_ratio_over_one(self, dummy_video_frame: av.VideoFrame):
        """ratio>1でValueErrorが発生する"""
        import pytest

        wrapped = WrappedVideoFrame(dummy_video_frame)
        with pytest.raises(ValueError):
            wrapped.crop_center(ratio=1.5)

    def test_crop_center_returns_new_frame(self, dummy_video_frame: av.VideoFrame):
        """クロップは新しいフレームを返し、元のフレームは変更されない"""
        wrapped = WrappedVideoFrame(dummy_video_frame)
        original_width = wrapped.frame.width
        original_height = wrapped.frame.height

        cropped = wrapped.crop_center()

        # 元のフレームは変更されていない
        assert wrapped.frame.width == original_width
        assert wrapped.frame.height == original_height
        # 新しいフレームは異なるサイズ
        assert cropped.frame.width != original_width
        assert cropped.frame.height != original_height

    def test_crop_center_pixel_data_correct(self):
        """クロップ後のピクセルデータが正しい位置から取得されている"""
        # 特定のパターンを持つフレームを作成
        width, height = 640, 480
        frame = av.VideoFrame(width, height, "yuv420p")

        # Y planeに位置を特定できるパターンを設定
        y_plane = frame.planes[0]
        y_data = np.zeros(y_plane.buffer_size, dtype=np.uint8)
        y_2d = y_data.reshape(height, y_plane.line_size)

        # 中央部分だけ255にする
        center_y = height // 2
        center_x = width // 2
        y_2d[center_y - 10 : center_y + 10, center_x - 10 : center_x + 10] = 255
        y_plane.update(y_data)

        # U, V planeを初期化
        for plane in frame.planes[1:]:
            data = np.full(plane.buffer_size, 128, dtype=np.uint8)
            plane.update(data)

        wrapped = WrappedVideoFrame(frame)
        cropped = wrapped.crop_center(ratio=0.5)  # 320x240

        # クロップ後の中央部分に255のデータがあることを確認
        cropped_planes = cropped.get_planes()
        cropped_y = cropped_planes[0]
        cropped_center_y = cropped_y.shape[0] // 2
        cropped_center_x = cropped_y.shape[1] // 2

        # 中央付近の値が255であることを確認
        assert cropped_y[cropped_center_y, cropped_center_x] == 255
