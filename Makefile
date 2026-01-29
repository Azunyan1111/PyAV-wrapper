.PHONY: test test-srt test-stream test-output test-writer-stream test-whep

test:
	uv run pytest

test-srt:
	uv run pytest -m srt -v -s -n 0 --timeout=120

test-stream:
	uv run pytest tests/test_stream_listener.py::TestStreamListenerSRTIntegration::test_srt_receive_grayscale_and_stream_to_srt -v -s -n 0 -m srt --timeout=120

test-output:
	uv run pytest tests/test_stream_listener.py::TestStreamListenerSRTIntegration::test_srt_receive_convert_grayscale_and_write -v -s -n 0 -m srt --timeout=120

test-writer-stream:
	uv run pytest tests/test_stream_writer.py::TestStreamWriterWithListenerIntegration::test_listener_to_writer_grayscale_stream -v -s -n 0 -m srt --timeout=120

test-whep:
	uv run pytest tests/test_raw_subprocess_pipe_stream_listener.py -v -s --timeout 60
