import os
import time
from pyav_wrapper import RawSubprocessPipeStreamListener, RawSubprocessPipeStreamWriter
import threading

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env.v2")


def _load_env_file(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


try:
    _env = _load_env_file(ENV_PATH)
except FileNotFoundError as e:
    raise SystemExit(f"{ENV_PATH} が見つかりません") from e

_required_keys = ("WHEP_CLIENT", "WHIP_CLIENT", "WHEP_URL", "WHIP_URL")
_missing_keys = [key for key in _required_keys if not _env.get(key)]
if _missing_keys:
    raise SystemExit(f".env.v2 に必須キーがありません: {', '.join(_missing_keys)}")

WHEP_CLIENT = _env["WHEP_CLIENT"]
WHIP_CLIENT = _env["WHIP_CLIENT"]
WHEP_URL = _env["WHEP_URL"]
WHIP_URL = _env["WHIP_URL"]

def main() -> None:
    # 1. WHEP受信開始
    print(f"Starting WHEP listener for {WHEP_URL}...")
    listener = RawSubprocessPipeStreamListener(command=[WHEP_CLIENT, WHEP_URL], width=1600, height=900)
    print("WHEP listener started.")

    # 2. WHIP送信開始
    print(f"Starting WHIP writer to {WHIP_URL}...")
    writer = RawSubprocessPipeStreamWriter(
        command=[WHIP_CLIENT, WHIP_URL],
        width=1600,
        height=900,
        fps=30,
        stats_enabled=True,
    )
    print("WHIP writer started.")

    # 音声は非同期で即座に送信する
    def audio_sending_loop() -> None:
        while True:
            audio_frames = listener.pop_all_audio_queue()
            for af in audio_frames:
                writer.enqueue_audio_frame(af)
            time.sleep(1)
            if len(audio_frames) > 0:
                print(f"First audio frame: {audio_frames[0].frame.pts}")
    audio_thread = threading.Thread(target=audio_sending_loop, daemon=True)
    audio_thread.start()

    # 4. 継続的に中継（300秒間）
    t = time.time()


    for i in range(60*30):  # 1分間実行
        # print(f"Sending frames...")  # 1秒ごとに溜まったフレームを両方書き込む
        video_frames = listener.pop_all_video_queue()
        for vf in video_frames:
            writer.enqueue_video_frame(vf)
        if len(video_frames) > 0:
            # 両方の一番最初のフレームのタイムスタンプを表示
            first_frame = video_frames[0]
            print(f"First frame: {first_frame.frame.pts}")
            print(f"Sent frames in {time.time() - t:.4f} seconds.")
            t = time.time()
            print(f"Send Video Frame size: {len(video_frames)}")
        # ここで一秒間GILに激しい処理を入れる
        # start = time.perf_counter()
        # while time.perf_counter() - start < 1.0:
        #     pass
        # 99% of the time holding GIL, 1% releasing GIL
        end = time.perf_counter() + 1.0
        while time.perf_counter() < end:
            # 約10ms GILを握る
            start = time.perf_counter()
            while time.perf_counter() - start < 0.01:
                pass
            # 約1msだけGILを解放
            time.sleep(1/1000)
        # time.sleep(1/30)

    # 6. 終了
    writer.stop()
    listener.stop()


if __name__ == "__main__":
    main()
