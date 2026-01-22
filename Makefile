.PHONY: test test-stream test-output test-writer-stream

test:
	uv run pytest

test-stream:
	uv run pytest tests/test_stream_listener.py::TestStreamListenerSRTIntegration::test_srt_receive_grayscale_and_stream_to_srt -v -s

test-output:
	uv run pytest tests/test_stream_listener.py::TestStreamListenerSRTIntegration::test_srt_receive_convert_grayscale_and_write -v -s

test-writer-stream:
	uv run pytest tests/test_stream_writer.py::TestStreamWriterWithListenerIntegration::test_listener_to_writer_grayscale_stream -v -s -n 0 --timeout=120
