import socket
import subprocess
import time
from pathlib import Path

import av
import numpy as np
import pytest

MOVIE_FILE = Path(__file__).parent.parent / "deps" / "test" / "movie.mp4"


def _find_free_port() -> int:
    """空いているポートを見つける"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _start_ffmpeg_process(args: list[str]) -> subprocess.Popen:
    """ffmpegプロセスをバックグラウンドで起動する"""
    return subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _terminate_process(proc: subprocess.Popen) -> None:
    """プロセスを安全に終了する"""
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def srt_source_url():
    """ffmpegでテスト動画をlocalhost SRTで配信するフィクスチャ

    ffmpegがSRT listenerモードで起動し、テストコードはcallerとして接続する。
    外部サーバー不要で完全にローカルで動作する。

    Yields:
        str: SRT接続URL（例: "srt://localhost:12345?mode=caller"）
    """
    if not MOVIE_FILE.exists():
        pytest.skip(f"テスト動画が見つかりません: {MOVIE_FILE}")

    port = _find_free_port()

    proc = _start_ffmpeg_process([
        "ffmpeg",
        "-re",
        "-stream_loop", "-1",
        "-i", str(MOVIE_FILE),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "30",
        "-tune", "zerolatency",
        "-c:a", "aac",
        "-b:a", "48k",
        "-f", "mpegts",
        f"srt://0.0.0.0:{port}?mode=listener",
    ])

    # ffmpegのSRT listener起動を待機
    time.sleep(3.0)

    yield f"srt://localhost:{port}?mode=caller"

    _terminate_process(proc)


@pytest.fixture
def srt_output_url():
    """SRT出力を受信するffmpegフィクスチャ

    ffmpegがSRT listenerモードで受信待機し、テストコードはcallerとして接続・送信する。
    受信データは破棄される。外部サーバー不要。

    Yields:
        str: SRT出力URL（例: "srt://localhost:12346?mode=caller"）
    """
    port = _find_free_port()

    proc = _start_ffmpeg_process([
        "ffmpeg",
        "-y",
        "-i", f"srt://0.0.0.0:{port}?mode=listener",
        "-c", "copy",
        "-f", "null",
        "-",
    ])

    # ffmpegのSRT listener起動を待機
    time.sleep(2.0)

    yield f"srt://localhost:{port}?mode=caller"

    _terminate_process(proc)


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
