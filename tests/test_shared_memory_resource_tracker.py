import os
import subprocess
import sys
from pathlib import Path


class TestSharedMemoryResourceTracker:
    """shared_memory解放漏れ警告の回帰テスト"""

    def _run_child_script(self, child_script: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        source_root = repo_root / "src"
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{source_root}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = str(source_root)

        return subprocess.run(
            [sys.executable, "-c", child_script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

    def _assert_no_shared_memory_leak_warning(self, result: subprocess.CompletedProcess[str]) -> None:
        assert result.returncode == 0, (
            f"child process failed: returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert "ok" in result.stdout
        assert "leaked shared_memory objects" not in result.stderr

    def test_stream_writer_does_not_emit_leaked_shared_memory_warning(self):
        child_script = """
import multiprocessing as mp
import os
import tempfile

import av
import numpy as np

from pyav_wrapper.stream_writer import StreamWriter
from pyav_wrapper.video_frame import WrappedVideoFrame

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

writer = StreamWriter(
    url=output_path,
    width=640,
    height=480,
    fps=30,
)

for _ in range(120):
    frame = av.VideoFrame(640, 480, "yuv420p")
    wrapped = WrappedVideoFrame(frame)
    y = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    u = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    v = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    wrapped.set_planes([y, u, v])
    writer.enqueue_video_frame(wrapped)

writer.stop()
print("ok")

try:
    os.unlink(output_path)
except Exception:
    pass
"""

        result = self._run_child_script(child_script)
        self._assert_no_shared_memory_leak_warning(result)

    def test_stream_writer_drop_stress_does_not_emit_warning(self):
        child_script = """
import multiprocessing as mp
import os
import tempfile

import av
import numpy as np

from pyav_wrapper.stream_writer import StreamWriter
from pyav_wrapper.video_frame import WrappedVideoFrame

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
writer = StreamWriter(
    url=output_path,
    width=640,
    height=480,
    fps=30,
)

for _ in range(800):
    frame = av.VideoFrame(640, 480, "yuv420p")
    wrapped = WrappedVideoFrame(frame)
    y = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    u = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    v = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
    wrapped.set_planes([y, u, v])
    writer.enqueue_video_frame(wrapped)

writer.stop()
print("ok")

try:
    os.unlink(output_path)
except Exception:
    pass
"""

        result = self._run_child_script(child_script)
        self._assert_no_shared_memory_leak_warning(result)

    def test_stream_writer_video_audio_shm_does_not_emit_warning(self):
        child_script = """
import multiprocessing as mp
import os
import tempfile

import av
import numpy as np

from pyav_wrapper.audio_frame import WrappedAudioFrame
from pyav_wrapper.stream_writer import StreamWriter
from pyav_wrapper.video_frame import WrappedVideoFrame

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
writer = StreamWriter(
    url=output_path,
    width=640,
    height=480,
    fps=30,
    sample_rate=48000,
)

audio_data = np.zeros((2, 32768), dtype=np.int16)
for i in range(200):
    video = av.VideoFrame(640, 480, "yuv420p")
    wrapped_video = WrappedVideoFrame(video)
    wrapped_video.set_planes(
        [
            np.random.randint(0, 255, (480, 640), dtype=np.uint8),
            np.random.randint(0, 255, (240, 320), dtype=np.uint8),
            np.random.randint(0, 255, (240, 320), dtype=np.uint8),
        ]
    )
    writer.enqueue_video_frame(wrapped_video)

    if i % 2 == 0:
        audio = av.AudioFrame.from_ndarray(audio_data, format="s16p", layout="stereo")
        audio.sample_rate = 48000
        writer.enqueue_audio_frame(WrappedAudioFrame(audio))

writer.stop()
print("ok")

try:
    os.unlink(output_path)
except Exception:
    pass
"""

        result = self._run_child_script(child_script)
        self._assert_no_shared_memory_leak_warning(result)

    def test_shared_memory_unlink_race_does_not_leave_tracker_registration(self):
        child_script = """
import multiprocessing as mp
from multiprocessing import shared_memory

from pyav_wrapper.stream_writer import (
    _close_and_unlink_shared_memory,
    _unregister_created_shared_memory,
)

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

seed = shared_memory.SharedMemory(create=True, size=16)
seed_name = seed.name
seed.close()
_unregister_created_shared_memory(seed_name)

attached_event = mp.Event()
go_event = mp.Event()

def late_cleanup(name, attached_evt, go_evt):
    shm = shared_memory.SharedMemory(name=name)
    attached_evt.set()
    go_evt.wait(timeout=10)
    _close_and_unlink_shared_memory(shm)

def early_cleanup(name, attached_evt, go_evt):
    attached_evt.wait(timeout=10)
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()
    go_evt.set()

p_late = mp.Process(target=late_cleanup, args=(seed_name, attached_event, go_event))
p_early = mp.Process(target=early_cleanup, args=(seed_name, attached_event, go_event))

p_late.start()
p_early.start()

p_late.join(timeout=30)
p_early.join(timeout=30)

if p_late.is_alive():
    p_late.terminate()
    p_late.join(timeout=5)
if p_early.is_alive():
    p_early.terminate()
    p_early.join(timeout=5)

if p_late.exitcode != 0 or p_early.exitcode != 0:
    raise RuntimeError(f"child exits: late={p_late.exitcode}, early={p_early.exitcode}")

print("ok")
"""

        result = self._run_child_script(child_script)
        self._assert_no_shared_memory_leak_warning(result)

    def test_stream_listener_pack_cleanup_does_not_emit_warning(self):
        child_script = """
import multiprocessing as mp

from pyav_wrapper import stream_listener as sl

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

chunks = [b"x" * 2048, b"y" * 4096, b"z" * 8192]
for _ in range(500):
    packed = sl._pack_bytes_to_shared_memory(chunks, threshold_bytes=1)
    if packed is None:
        raise RuntimeError("shared memory packing failed")
    shm_name, _ = packed
    payload = {"storage": "shm", "shm_name": shm_name}
    sl._cleanup_payload_shared_memory(payload)
    sl._cleanup_payload_shared_memory(payload)

print("ok")
"""

        result = self._run_child_script(child_script)
        self._assert_no_shared_memory_leak_warning(result)

    def test_stream_writer_cross_process_cleanup_does_not_emit_warning(self):
        child_script = """
import multiprocessing as mp

from pyav_wrapper import stream_writer as sw

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

def worker_cleanup(name: str) -> None:
    sw._cleanup_payload_shared_memory({"storage": "shm", "shm_name": name})

for _ in range(200):
    packed = sw._pack_bytes_to_shared_memory([b"a" * 8192, b"b" * 8192], threshold_bytes=1)
    if packed is None:
        raise RuntimeError("shared memory packing failed")
    shm_name, _ = packed
    proc = mp.Process(target=worker_cleanup, args=(shm_name,))
    proc.start()
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3)
        raise RuntimeError("cleanup process timeout")
    if proc.exitcode != 0:
        raise RuntimeError(f"cleanup process failed: {proc.exitcode}")

print("ok")
"""

        result = self._run_child_script(child_script)
        self._assert_no_shared_memory_leak_warning(result)
