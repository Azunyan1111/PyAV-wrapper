import av
import numpy as np
import pytest


@pytest.fixture
def dummy_video_frame() -> av.VideoFrame:
    """YUV420p形式のダミーVideoFrameを生成する"""
    width = 640
    height = 480
    frame = av.VideoFrame(width, height, "yuv420p")

    # Y plane: 全て128（グレー）
    y_plane = frame.planes[0]
    y_data = np.full(y_plane.buffer_size, 128, dtype=np.uint8)
    y_plane.update(y_data)

    # U plane: 全て128（無彩色）
    u_plane = frame.planes[1]
    u_data = np.full(u_plane.buffer_size, 128, dtype=np.uint8)
    u_plane.update(u_data)

    # V plane: 全て128（無彩色）
    v_plane = frame.planes[2]
    v_data = np.full(v_plane.buffer_size, 128, dtype=np.uint8)
    v_plane.update(v_data)

    return frame


@pytest.fixture
def dummy_audio_frame() -> av.AudioFrame:
    """fltp形式（float planar）のダミーAudioFrameを生成する"""
    samples = 1024
    channels = 2
    sample_rate = 48000

    # fltp形式: (channels, samples) の形状
    audio_data = np.zeros((channels, samples), dtype=np.float32)

    frame = av.AudioFrame.from_ndarray(audio_data, format="fltp", layout="stereo")
    frame.sample_rate = sample_rate

    return frame
