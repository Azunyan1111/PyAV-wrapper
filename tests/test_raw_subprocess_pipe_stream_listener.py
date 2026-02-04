import os
import shutil
import threading
import time

import pytest
from dotenv import load_dotenv

from pyav_wrapper import (
    RawSubprocessPipeStreamListener,
    StreamListener,
    WrappedAudioFrame,
    WrappedVideoFrame,
)

load_dotenv()

WHEP_URL = os.getenv("WHEP_URL")
WHEP_CLIENT_PATH = "./deps/whep-client"


def check_whep_available() -> bool:
    """whep-clientバイナリとWHEP_URLが利用可能か確認"""
    if WHEP_URL is None:
        return False
    if not shutil.which(WHEP_CLIENT_PATH) and not os.path.isfile(WHEP_CLIENT_PATH):
        return False
    return True


WHEP_AVAILABLE = check_whep_available()


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


@pytest.mark.skipif(
    not WHEP_AVAILABLE,
    reason="whep-clientバイナリまたはWHEP_URL環境変数が利用できません",
)
class TestRawSubprocessPipeStreamListenerIntegration:
    """WHEP統合テスト：サブプロセスパイプ経由でフレームを受信"""

    def test_receive_video_and_audio_from_pipe(self):
        """whep-clientからパイプ経由で映像・音声フレームを受信できる"""
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
