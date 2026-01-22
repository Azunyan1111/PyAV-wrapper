import av
import numpy as np


class WrappedVideoFrame:
    """PyAVのVideoFrameをラップし、バッファ操作メソッドを追加するクラス"""

    def __init__(self, frame: av.VideoFrame):
        self._frame = frame
        self._is_bad_frame = False

    @property
    def frame(self) -> av.VideoFrame:
        """元のAVFrameを取得"""
        return self._frame

    @property
    def is_bad_frame(self) -> bool:
        """フレームが不正かどうかを取得"""
        return self._is_bad_frame

    @is_bad_frame.setter
    def is_bad_frame(self, value: bool) -> None:
        """フレームが不正かどうかを設定"""
        self._is_bad_frame = value

    def get_buffer(self) -> np.ndarray:
        """フレーム全体のバッファをnumpy配列として取得

        YUV420p等のplanar形式の場合、Y planeのみを返す。
        全planeが必要な場合はget_planes()を使用する。
        """
        plane = self._frame.planes[0]
        height = plane.height
        line_size = plane.line_size
        width = plane.width

        # memoryviewからnumpy配列を作成
        buffer = np.frombuffer(plane, dtype=np.uint8)
        # line_sizeにはパディングが含まれる可能性があるため、reshapeして必要部分を切り出す
        buffer = buffer.reshape(height, line_size)[:, :width]
        return buffer.copy()

    def set_buffer(self, data: np.ndarray) -> None:
        """フレーム全体のバッファを上書き

        YUV420p等のplanar形式の場合、Y planeのみを上書きする。
        全planeを上書きする場合はset_planes()を使用する。
        """
        plane = self._frame.planes[0]
        height = plane.height
        line_size = plane.line_size
        width = plane.width

        # line_sizeに合わせてパディングを追加
        if data.shape == (height, width):
            padded = np.zeros((height, line_size), dtype=np.uint8)
            padded[:, :width] = data
            plane.update(padded.tobytes())
        else:
            plane.update(data.tobytes())

    def get_planes(self) -> list[np.ndarray]:
        """各plane（Y, U, V等）を個別にnumpy配列として取得"""
        planes = []
        for plane in self._frame.planes:
            height = plane.height
            line_size = plane.line_size
            width = plane.width

            buffer = np.frombuffer(plane, dtype=np.uint8)
            buffer = buffer.reshape(height, line_size)[:, :width]
            planes.append(buffer.copy())
        return planes

    def set_planes(self, planes: list[np.ndarray]) -> None:
        """各planeを個別に上書き"""
        for i, data in enumerate(planes):
            plane = self._frame.planes[i]
            height = plane.height
            line_size = plane.line_size
            width = plane.width

            if data.shape == (height, width):
                padded = np.zeros((height, line_size), dtype=np.uint8)
                padded[:, :width] = data
                plane.update(padded.tobytes())
            else:
                plane.update(data.tobytes())

    def crop_center(self, ratio: float = 0.8) -> "WrappedVideoFrame":
        """中央からratioの割合でクロップした新しいWrappedVideoFrameを返す

        thread_ffmpeg_streamer.pyのcrop_center_resize_fast相当の処理。
        元のフレームのメタデータ（pts, time_base等）は新しいフレームにコピーされる。

        Args:
            ratio: クロップ比率（0.0〜1.0）。デフォルト0.8で80%の領域を切り出す。

        Returns:
            クロップされた新しいWrappedVideoFrame
        """
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be between 0.0 and 1.0, got {ratio}")

        original_width = self._frame.width
        original_height = self._frame.height

        # クロップ後のサイズを計算
        crop_width = int(original_width * ratio)
        crop_height = int(original_height * ratio)

        # YUV420pの場合、幅と高さは2の倍数である必要がある
        crop_width = crop_width - (crop_width % 2)
        crop_height = crop_height - (crop_height % 2)

        # クロップ開始位置（中央からクロップ）
        x0 = (original_width - crop_width) // 2
        y0 = (original_height - crop_height) // 2

        # 新しいフレームを作成
        new_frame = av.VideoFrame(crop_width, crop_height, self._frame.format.name)

        # メタデータをコピー
        new_frame.pts = self._frame.pts
        if self._frame.time_base is not None:
            new_frame.time_base = self._frame.time_base

        # 各planeをクロップしてコピー
        planes = self.get_planes()
        new_planes = []

        for i, plane_data in enumerate(planes):
            # YUV420pの場合、U/VプレーンはY planeの半分のサイズ
            if i == 0:
                # Y plane
                px0, py0 = x0, y0
                pw, ph = crop_width, crop_height
            else:
                # U/V planes（半分のサイズ）
                px0, py0 = x0 // 2, y0 // 2
                pw, ph = crop_width // 2, crop_height // 2

            cropped = plane_data[py0 : py0 + ph, px0 : px0 + pw].copy()
            new_planes.append(cropped)

        # 新しいフレームにplaneデータを書き込む
        wrapped_new = WrappedVideoFrame(new_frame)
        wrapped_new.set_planes(new_planes)

        return wrapped_new
