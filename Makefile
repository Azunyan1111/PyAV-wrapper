.PHONY: test test-stream test-output

test:
	uv run pytest

test-stream:
	uv run pytest tests/test_stream_listener.py::TestStreamListenerSRTIntegration::test_srt_receive_grayscale_and_stream_to_srt -v -s

test-output:
	uv run pytest tests/test_stream_listener.py::TestStreamListenerSRTIntegration::test_srt_receive_convert_grayscale_and_write -v -s
