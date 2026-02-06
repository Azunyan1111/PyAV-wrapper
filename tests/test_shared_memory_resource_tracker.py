import os
import subprocess
import sys
from pathlib import Path


class TestSharedMemoryResourceTracker:
    """shared_memory解放漏れ警告の回帰テスト"""

    def test_stream_writer_does_not_emit_leaked_shared_memory_warning(self):
        repo_root = Path(__file__).resolve().parents[1]
        source_root = repo_root / "src"

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

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{source_root}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = str(source_root)

        result = subprocess.run(
            [sys.executable, "-c", child_script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert result.returncode == 0, (
            f"child process failed: returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert "ok" in result.stdout
        assert "leaked shared_memory objects" not in result.stderr
