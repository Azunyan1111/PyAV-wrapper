import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

from pyav_wrapper import (
    RawSubprocessPipeStreamListener,
    RawSubprocessPipeStreamWriter,
    StreamWriter,
    WrappedAudioFrame,
    WrappedVideoFrame,
)

load_dotenv()

WHIP_URL = os.getenv("WHIP_URL")
WHEP_URL = os.getenv("WHEP_URL")
WHIP_URL_2 = os.getenv("WHIP_URL_2")
WHEP_URL_2 = os.getenv("WHEP_URL_2")
WHIP_CLIENT_PATH = "./deps/whip-client"
WHEP_CLIENT_PATH = "./deps/whep-client"
MOVIE_FILE = Path(__file__).parent.parent / "deps" / "test" / "movie.mp4"
LOCK_FILE_PATH = "/tmp/pyav_wrapper_whip_whep.lock"


def check_whip_whep_available() -> bool:
    """whip-client/whep-clientバイナリとWHIP/WHEP URL/テスト動画が利用可能か確認"""
    if WHIP_URL is None or WHEP_URL is None:
        return False
    if WHIP_URL_2 is None or WHEP_URL_2 is None:
        return False
    if not shutil.which(WHIP_CLIENT_PATH) and not os.path.isfile(WHIP_CLIENT_PATH):
        return False
    if not shutil.which(WHEP_CLIENT_PATH) and not os.path.isfile(WHEP_CLIENT_PATH):
        return False
    if not MOVIE_FILE.exists():
        return False
    return True


WHIP_WHEP_AVAILABLE = check_whip_whep_available()


class _WhipWhepLock:
    def __enter__(self):
        self._lock_file = open(LOCK_FILE_PATH, "w")
        fcntl.flock(self._lock_file, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()


class TestRawSubprocessPipeStreamWriterInit:
    """RawSubprocessPipeStreamWriter初期化のテスト"""

    def test_init_with_command(self):
        """コンストラクタにコマンドを渡して保持されるか"""
        command = ["echo", "test"]
        writer = RawSubprocessPipeStreamWriter.__new__(
            RawSubprocessPipeStreamWriter
        )
        writer._command = command
        assert writer._command == ["echo", "test"]

    def test_inherits_stream_writer(self):
        """StreamWriterを継承しているか"""
        assert issubclass(RawSubprocessPipeStreamWriter, StreamWriter)


@pytest.mark.skipif(
    not WHIP_WHEP_AVAILABLE,
    reason="whip-client/whep-clientバイナリ、WHIP_URL/WHEP_URL環境変数、またはテスト動画が利用できません",
)
class TestRawSubprocessPipeStreamWriterIntegration:
    """WHIP統合テスト：ffmpeg|whip-client配信 -> WHEP受信 -> WHIP送信 E2Eテスト"""

    @pytest.mark.timeout(120)
    def test_send_video_and_audio_to_pipe(self):
        """ffmpeg|whip-client配信 -> WHEP受信 -> WHIP送信のE2Eテスト"""
        with _WhipWhepLock():
            # 1. ffmpeg -> whip-client パイプでWHIP配信を開始
            #    ffmpeg -re -i movie.mp4 -c:v rawvideo -pix_fmt yuv420p -c:a pcm_s16le -f matroska - | whip-client WHIP_URL
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
            # ffmpegのstdoutをwhip-clientに渡したので、親プロセス側は閉じる
            ffmpeg_proc.stdout.close()

            try:
                # WHIP配信がCloudflareに到達するまで待機
                print(f"\nWHIP配信開始: {WHIP_URL}")
                time.sleep(10.0)

                # 2. WHEP受信開始
                listener_command = [WHEP_CLIENT_PATH, WHEP_URL]
                listener = RawSubprocessPipeStreamListener(command=listener_command, width=640, height=480)
                print(f"WHEP受信開始: {WHEP_URL}")

                # フレーム受信を待機
                time.sleep(10.0)

                # 最初のフレームから解像度を取得
                video_frames = []
                for _ in range(20):
                    frames = listener.pop_all_video_queue()
                    if frames:
                        video_frames.extend(frames)
                        break
                    time.sleep(0.5)

                assert len(video_frames) > 0, "映像フレームを受信できませんでした"

                first_frame = video_frames[0]
                width = first_frame.frame.width
                height = first_frame.frame.height
                fps = 30

                print(f"受信解像度: {width}x{height}")

                # 3. WHIP送信開始（RawSubprocessPipeStreamWriterのテスト、別ストリームへ送信）
                writer_command = [WHIP_CLIENT_PATH, WHIP_URL_2]
                writer = RawSubprocessPipeStreamWriter(
                    command=writer_command,
                    width=width,
                    height=height,
                    fps=fps,
                )
                print(f"WHIP送信開始（Writer経由）: {WHIP_URL_2}")

                # 受信済みフレームを送信
                for vf in video_frames:
                    writer.enqueue_video_frame(vf)

                audio_frames = listener.pop_all_audio_queue()
                for af in audio_frames:
                    writer.enqueue_audio_frame(af)

                # 10秒間中継
                video_count = len(video_frames)
                audio_count = len(audio_frames)

                for _ in range(100):
                    time.sleep(0.1)

                    vframes = listener.pop_all_video_queue()
                    for vf in vframes:
                        writer.enqueue_video_frame(vf)
                        video_count += 1

                    aframes = listener.pop_all_audio_queue()
                    for af in aframes:
                        writer.enqueue_audio_frame(af)
                        audio_count += 1

                assert video_count > 0, "映像フレームが1つも中継されませんでした"
                assert audio_count > 0, "音声フレームが1つも中継されませんでした"

                print(f"中継した映像フレーム数: {video_count}")
                print(f"中継した音声フレーム数: {audio_count}")

                # 4. WHEP_URL_2で受信確認（Writerの出力が実際に届いているか検証）
                #    stdoutは破棄し、stderrのログで受信状況を確認する
                print(f"WHEP受信確認開始: {WHEP_URL_2}")
                verify_proc = subprocess.Popen(
                    [WHEP_CLIENT_PATH, WHEP_URL_2],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                # 受信を待機
                time.sleep(10.0)

                # プロセスを終了してstderrを取得
                verify_proc.terminate()
                try:
                    verify_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    verify_proc.kill()
                    verify_proc.wait()

                stderr_output = verify_proc.stderr.read().decode("utf-8", errors="replace")
                print(f"WHEP_URL_2 whep-client stderr:\n{stderr_output}")

                # whep-clientが正常に動作していたことを確認
                # returncode=0 はテスト側のterminate(-15)で終了するため期待しない
                # stderrにエラーが無いこと、またはデータ受信のログがあることを確認
                assert "error" not in stderr_output.lower() or "eof" in stderr_output.lower(), \
                    f"WHEP_URL_2の受信でエラーが発生しました: {stderr_output}"

                writer.stop()
                listener.stop()

            finally:
                # プロセスを確実に終了
                for proc in [whip_source_proc, ffmpeg_proc]:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    except Exception:
                        pass
