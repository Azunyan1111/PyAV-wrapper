import os
import time
from pyav_wrapper import RawSubprocessPipeStreamListener, RawSubprocessPipeStreamWriter

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
# WHIP_URL = _env["WHIP_URL"]
WHIP_URL = f"http://localhost:8889/{int(time.time() * 1000)}/whip"


def main() -> None:
    # 1. WHEP受信開始
    print(f"Starting WHEP listener for {WHEP_URL}...")
    listener = RawSubprocessPipeStreamListener(
        command=[WHEP_CLIENT, WHEP_URL],
        width=1600,
        height=900,
        batch_size=5,
    )
    print("WHEP listener started.")

    # 2. WHIP送信開始
    print(f"Starting WHIP writer to {WHIP_URL}...")
    writer = RawSubprocessPipeStreamWriter(
        command=[WHIP_CLIENT, WHIP_URL],
        width=1280,
        height=720,
        fps=30,
        crop_ratio=0.8,
        stats_enabled=True,
        video_queue_maxlen=int(5*1.7),
    )
    print("WHIP writer started.")

    # 映像は受信payloadをそのままwriterへ直送する（デシリアライズ不要）
    # listener.forward_video_to_writer(writer, forward_only=True)
    # print("Video direct forwarding enabled.")

    # 音声は受信payloadをそのままwriterへ直送する（デシリアライズ不要）
    listener.forward_audio_to_writer(writer, forward_only=True)
    print("Audio direct forwarding enabled.")

    # 4. 継続的に中継（300秒間）
    for i in range(15 * 240):  # 1分間実行
        frames = listener.pop_all_video_queue()
        if len(frames) != 0:
            for frame in frames:
                # グレースケールする
                gray_frame = frame.frame.to_ndarray(format="gray")
                frame.set_buffer(gray_frame)
            writer.enqueue_video_frames(frames)


        # 直送モードでは待機のみ
        time.sleep(1 / 240)

    # 6. 終了
    writer.stop()
    listener.stop()


if __name__ == "__main__":
    main()
