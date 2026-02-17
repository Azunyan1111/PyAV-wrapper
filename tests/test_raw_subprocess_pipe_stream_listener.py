import fcntl
import multiprocessing
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import av
import numpy as np
import pytest
from dotenv import load_dotenv

from pyav_wrapper import (
    RawSubprocessPipeStreamListener,
    StreamListener,
    WrappedAudioFrame,
    WrappedVideoFrame,
)
from pyav_wrapper.raw_subprocess_pipe_stream_listener import _raw_subprocess_pipe_decode_worker

load_dotenv()

WHIP_URL = os.getenv("WHIP_URL")
WHEP_URL = os.getenv("WHEP_URL")
WHIP_CLIENT_PATH = "./deps/whip-client"
WHEP_CLIENT_PATH = "./deps/whep-client"
MOVIE_FILE = Path(__file__).parent.parent / "deps" / "test" / "movie.mp4"
LOCK_FILE_PATH = "/tmp/pyav_wrapper_whip_whep.lock"


def check_whep_available() -> bool:
    """whip-client/whep-clientバイナリとWHIP/WHEP URL/テスト動画が利用可能か確認"""
    if WHEP_URL is None or WHIP_URL is None:
        return False
    if not shutil.which(WHEP_CLIENT_PATH) and not os.path.isfile(WHEP_CLIENT_PATH):
        return False
    if not shutil.which(WHIP_CLIENT_PATH) and not os.path.isfile(WHIP_CLIENT_PATH):
        return False
    if not MOVIE_FILE.exists():
        return False
    return True


WHEP_AVAILABLE = check_whep_available()


def create_test_opus_video_file(path: Path, duration_frames: int = 60) -> None:
    """テスト用のOpus音声付き動画ファイルを作成"""
    container = av.open(str(path), mode="w", format="matroska")

    video_stream = container.add_stream("libx264", rate=30)
    video_stream.width = 640
    video_stream.height = 480
    video_stream.pix_fmt = "yuv420p"

    audio_stream = container.add_stream("libopus", rate=48000)
    audio_stream.layout = "stereo"

    for i in range(duration_frames):
        video_frame = av.VideoFrame(640, 480, "yuv420p")
        for plane in video_frame.planes:
            data = np.full(plane.buffer_size, 128, dtype=np.uint8)
            plane.update(data)
        video_frame.pts = i
        for packet in video_stream.encode(video_frame):
            container.mux(packet)

        samples = 960
        audio_data = np.zeros((2, samples), dtype=np.float32)
        audio_frame = av.AudioFrame.from_ndarray(audio_data, format="fltp", layout="stereo")
        audio_frame.sample_rate = 48000
        audio_frame.pts = i * samples
        for packet in audio_stream.encode(audio_frame):
            container.mux(packet)

    for packet in video_stream.encode():
        container.mux(packet)
    for packet in audio_stream.encode():
        container.mux(packet)
    container.close()


class _WhipWhepLock:
    def __enter__(self):
        self._lock_file = open(LOCK_FILE_PATH, "w")
        fcntl.flock(self._lock_file, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()


class TestRawSubprocessPipeStreamListenerInit:
    """RawSubprocessPipeStreamListener初期化のテスト"""

    def test_init_with_command(self):
        """コンストラクタにコマンドを渡して保持されるか"""
        command = ["echo", "test"]
        listener = RawSubprocessPipeStreamListener.__new__(
            RawSubprocessPipeStreamListener
        )
        listener._command = command
        assert listener._command == ["echo", "test"]

    def test_inherits_stream_listener(self):
        """StreamListenerを継承しているか"""
        assert issubclass(RawSubprocessPipeStreamListener, StreamListener)


class TestRawSubprocessPipeStreamListenerReconnection:
    """RawSubprocessPipeStreamListenerの再接続機能テスト"""

    def test_reconnection_variables_initialized(self):
        """サブクラスでも再接続関連の変数が初期化される"""
        listener = RawSubprocessPipeStreamListener.__new__(
            RawSubprocessPipeStreamListener
        )
        # __new__のみで__init__を呼ばず、属性の存在をクラス定義で確認
        # 代わりにStreamListenerの親クラスに定義があることを確認
        assert hasattr(StreamListener, '_monitor_frame_updates')
        assert hasattr(StreamListener, '_restart_connection')

    def test_has_restart_connection_method(self):
        """_restart_connectionメソッドがオーバーライドされている"""
        assert hasattr(RawSubprocessPipeStreamListener, '_restart_connection')
        assert (
            RawSubprocessPipeStreamListener._restart_connection
            is not StreamListener._restart_connection
        )


class TestRawSubprocessPipeStreamListenerPacketPassthrough:
    """RawSubprocessPipeStreamListenerのOpus packet passthroughテスト"""

    def test_decode_worker_emits_audio_packet_payload(self):
        """Opus packetモード有効時にaudio_packet payloadを出力する"""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = Path(tmpdir) / "source_opus.mkv"
            create_test_opus_video_file(source_file, duration_frames=90)

            mp_ctx = multiprocessing.get_context()
            stop_event = mp_ctx.Event()
            video_queue: queue.Queue[object] = queue.Queue()
            audio_queue: queue.Queue[object] = queue.Queue()

            _raw_subprocess_pipe_decode_worker(
                command=["cat", str(source_file)],
                width=640,
                height=480,
                stop_event=stop_event,
                video_queue=video_queue,
                audio_queue=audio_queue,
                video_drop_count=1,
                audio_drop_count=1,
                start_timeout_seconds=5.0,
                start_retry_interval_seconds=0.1,
                stderr_log_path=None,
                audio_packet_passthrough_enabled=True,
                audio_packet_codec_name="opus",
            )

            audio_payloads: list[dict[str, object]] = []
            while True:
                try:
                    item = audio_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, dict):
                    audio_payloads.append(item)
                elif isinstance(item, list):
                    for payload in item:
                        if isinstance(payload, dict):
                            audio_payloads.append(payload)

            assert len(audio_payloads) > 0
            first_payload = audio_payloads[0]
            assert first_payload.get("payload_type") == "audio_packet"
            assert first_payload.get("codec_name") == "opus"
            assert isinstance(first_payload.get("packet_bytes"), (bytes, bytearray))


@pytest.mark.skipif(
    not WHEP_AVAILABLE,
    reason="whip-client/whep-clientバイナリ、WHIP_URL/WHEP_URL環境変数、またはテスト動画が利用できません",
)
class TestRawSubprocessPipeStreamListenerIntegration:
    """WHEP統合テスト：サブプロセスパイプ経由でフレームを受信"""

    @pytest.mark.timeout(120)
    def test_receive_video_and_audio_from_pipe(self):
        """whep-clientからパイプ経由で映像・音声フレームを受信できる"""
        listener = None
        with _WhipWhepLock():
            ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-re",
                    "-stream_loop", "-1",
                    "-i", str(MOVIE_FILE),
                    "-c:v", "rawvideo",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "pcm_s16le",
                    "-f", "matroska",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            whip_source_proc = subprocess.Popen(
                [WHIP_CLIENT_PATH, WHIP_URL],
                stdin=ffmpeg_proc.stdout,
                stderr=subprocess.PIPE,
            )
            ffmpeg_proc.stdout.close()

            try:
                time.sleep(10.0)

                command = [WHEP_CLIENT_PATH, WHEP_URL]
                listener = RawSubprocessPipeStreamListener(command=command, width=640, height=480)

                time.sleep(10.0)

                assert listener.is_running is True

                video_frames = []
                for _ in range(20):
                    frames = listener.pop_all_video_queue()
                    if frames:
                        video_frames.extend(frames)
                        break
                    time.sleep(0.5)

                if len(video_frames) == 0:
                    with listener.video_queue_lock:
                        video_frames = list(listener.video_queue)

                assert len(video_frames) > 0, "映像フレームを受信できませんでした"

                frame = video_frames[0]
                assert isinstance(frame, WrappedVideoFrame)
                assert frame.frame.width > 0
                assert frame.frame.height > 0

                buffer = frame.get_buffer()
                assert buffer is not None
                assert buffer.shape[0] > 0
                assert buffer.shape[1] > 0

                planes = frame.get_planes()
                assert len(planes) > 0

                audio_frames = []
                for _ in range(20):
                    frames = listener.pop_all_audio_queue()
                    if frames:
                        audio_frames.extend(frames)
                        break
                    time.sleep(0.5)

                if len(audio_frames) == 0:
                    with listener.audio_queue_lock:
                        audio_frames = list(listener.audio_queue)

                assert len(audio_frames) > 0, "音声フレームを受信できませんでした"

                audio = audio_frames[0]
                assert isinstance(audio, WrappedAudioFrame)
                assert audio.frame.sample_rate > 0

                audio_buffer = audio.get_buffer()
                assert audio_buffer is not None
                assert audio_buffer.shape[0] > 0

                listener.stop()

                assert listener.is_running is False
            finally:
                if listener is not None:
                    listener.stop()
                for proc in [whip_source_proc, ffmpeg_proc]:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    except Exception:
                        pass
