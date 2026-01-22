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
