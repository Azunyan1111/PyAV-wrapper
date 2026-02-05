import av
import numpy as np
from typing import Any


class WrappedAudioFrame:
    """PyAVのAudioFrameをラップし、バッファ操作メソッドを追加するクラス"""

    def __init__(self, frame: av.AudioFrame):
        self._frame = frame
        self._serialized_payload: dict[str, Any] | None = None

    @property
    def frame(self) -> av.AudioFrame:
        """元のAVFrameを取得"""
        return self._frame

    def set_serialized_payload(self, payload: dict[str, Any] | None) -> None:
        """元のシリアライズ済みpayloadを保持する"""
        self._serialized_payload = payload

    def get_serialized_payload(self) -> dict[str, Any] | None:
        """保持しているシリアライズ済みpayloadを取得する"""
        return self._serialized_payload

    def get_buffer(self) -> np.ndarray:
        """フレーム全体のバッファをnumpy配列として取得

        Planar形式の場合、(channels, samples)の形状で返す。
        """
        return self._frame.to_ndarray()

    def set_buffer(self, data: np.ndarray) -> None:
        """フレーム全体のバッファを上書き

        Planar形式の場合、dataは(channels, samples)の形状である必要がある。
        """
        format_name = self._frame.format.name
        layout_name = self._frame.layout.name
        sample_rate = self._frame.sample_rate
        pts = self._frame.pts

        # 新しいフレームを作成してデータをコピー
        new_frame = av.AudioFrame.from_ndarray(data, format=format_name, layout=layout_name)
        new_frame.sample_rate = sample_rate
        new_frame.pts = pts

        # 各planeのデータを元のフレームにコピー
        for i, plane in enumerate(new_frame.planes):
            self._frame.planes[i].update(bytes(plane))
        self._serialized_payload = None

    def get_planes(self) -> list[np.ndarray]:
        """各チャネルを個別にnumpy配列として取得（Planar形式用）"""
        planes = []
        format_info = self._frame.format
        samples = self._frame.samples

        for plane in self._frame.planes:
            buffer = np.frombuffer(plane, dtype=self._get_numpy_dtype())
            buffer = buffer[:samples].copy()
            planes.append(buffer)
        return planes

    def set_planes(self, planes: list[np.ndarray]) -> None:
        """各チャネルを個別に上書き"""
        for i, data in enumerate(planes):
            self._frame.planes[i].update(data.tobytes())
        self._serialized_payload = None

    def _get_numpy_dtype(self) -> np.dtype:
        """AudioFrameのフォーマットに対応するnumpyのdtypeを取得"""
        format_name = self._frame.format.name
        dtype_map = {
            "u8": np.uint8,
            "u8p": np.uint8,
            "s16": np.int16,
            "s16p": np.int16,
            "s32": np.int32,
            "s32p": np.int32,
            "flt": np.float32,
            "fltp": np.float32,
            "dbl": np.float64,
            "dblp": np.float64,
        }
        return dtype_map.get(format_name, np.float32)
