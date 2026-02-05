import os
import subprocess
import sys
from pathlib import Path


class TestSharedMemoryResourceTracker:
    """shared_memory解放漏れ警告の回帰テスト"""

    def _run_child_script(
        self,
        child_script: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        source_root = repo_root / "src"
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{source_root}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = str(source_root)
        if extra_env:
            env.update(extra_env)

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

    def test_shared_memory_tracking_writes_to_stderr(self):
        child_script = """
import pyav_wrapper
from multiprocessing import shared_memory

shm = shared_memory.SharedMemory(create=True, size=16)
shm.close()
shm.unlink()
print("ok")
"""
        result = self._run_child_script(
            child_script,
            extra_env={
                "PYAV_WRAPPER_TRACE_SHM": "1",
                "PYAV_WRAPPER_TRACE_SHM_STACK": "1",
            },
        )
        assert result.returncode == 0, (
            f"child process failed: returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert "ok" in result.stdout
        assert "[pyav_wrapper][shm-trace] tracking enabled" in result.stderr
        assert "[pyav_wrapper][shm-trace] init name=" in result.stderr
        assert "[pyav_wrapper][shm-trace] creation_stack name=" in result.stderr
        assert "[pyav_wrapper][shm-trace] close name=" in result.stderr
        assert "[pyav_wrapper][shm-trace] unlink name=" in result.stderr

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

    def test_stream_writer_force_terminate_restart_stress_does_not_emit_warning(self):
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

for _cycle in range(12):
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = StreamWriter(
        url=output_path,
        width=640,
        height=480,
        fps=30,
        sample_rate=48000,
    )

    audio_data = np.zeros((2, 32768), dtype=np.int16)
    for i in range(90):
        frame = av.VideoFrame(640, 480, "yuv420p")
        wrapped = WrappedVideoFrame(frame)
        wrapped.set_planes(
            [
                np.random.randint(0, 255, (480, 640), dtype=np.uint8),
                np.random.randint(0, 255, (240, 320), dtype=np.uint8),
                np.random.randint(0, 255, (240, 320), dtype=np.uint8),
            ]
        )
        writer.enqueue_video_frame(wrapped)
        if i % 3 == 0:
            audio = av.AudioFrame.from_ndarray(audio_data, format="s16p", layout="stereo")
            audio.sample_rate = 48000
            writer.enqueue_audio_frame(WrappedAudioFrame(audio))

    proc = getattr(writer, "_write_process", None)
    if proc is not None and proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)

    writer.stop()
    try:
        os.unlink(output_path)
    except Exception:
        pass

print("ok")
"""

        result = self._run_child_script(child_script)
        self._assert_no_shared_memory_leak_warning(result)

    def test_parallel_pack_cleanup_race_stress_does_not_emit_warning(self):
        child_script = """
import multiprocessing as mp

from pyav_wrapper import stream_writer as sw

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

def producer(out_q: mp.Queue, rounds: int) -> None:
    for _ in range(rounds):
        packed = sw._pack_bytes_to_shared_memory(
            [b"a" * 32768, b"b" * 32768, b"c" * 32768],
            threshold_bytes=1,
        )
        if packed is None:
            continue
        name, _ = packed
        out_q.put(name)
    out_q.put(None)

def consumer(in_q: mp.Queue, producer_count: int) -> None:
    finished = 0
    while finished < producer_count:
        item = in_q.get(timeout=30)
        if item is None:
            finished += 1
            continue
        sw._cleanup_payload_shared_memory({"storage": "shm", "shm_name": item})
        sw._cleanup_payload_shared_memory({"storage": "shm", "shm_name": item})

q = mp.Queue(maxsize=256)
producer_count = 4
producers = [
    mp.Process(target=producer, args=(q, 250))
    for _ in range(producer_count)
]
consumers = [
    mp.Process(target=consumer, args=(q, producer_count // 2))
    for _ in range(2)
]

for p in producers:
    p.start()
for c in consumers:
    c.start()

for p in producers:
    p.join(timeout=60)
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
    if p.exitcode != 0:
        raise RuntimeError(f"producer failed: {p.exitcode}")

q.put(None)
q.put(None)

for c in consumers:
    c.join(timeout=60)
    if c.is_alive():
        c.terminate()
        c.join(timeout=5)
    if c.exitcode != 0:
        raise RuntimeError(f"consumer failed: {c.exitcode}")

print("ok")
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

    def test_pack_exception_path_does_not_emit_warning(self):
        child_script = """
import multiprocessing as mp

from pyav_wrapper import stream_writer as sw

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

# len()は通るがbuffer代入で失敗する値を混ぜて、shm書き込み例外経路を通す
for _ in range(300):
    packed = sw._pack_bytes_to_shared_memory([b"ok", [1, 2, 3]], threshold_bytes=1)
    if packed is not None:
        raise RuntimeError("expected None for invalid chunk")

print("ok")
"""

        result = self._run_child_script(child_script)
        self._assert_no_shared_memory_leak_warning(result)
