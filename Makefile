.PHONY: test test-srt test-stream test-output test-writer-stream test-whep docker-build docker-run release

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

test-whip:
	uv run pytest tests/test_raw_subprocess_pipe_stream_writer.py -v -s --timeout 120

release:
	$(eval CUR := $(shell grep '^version = ' pyproject.toml | sed 's/version = "//;s/"//'))
	$(eval NEXT := $(shell echo $(CUR) | awk -F. '{print $$1"."$$2"."$$3+1}'))
	sed -i '' 's/^version = ".*"/version = "$(NEXT)"/' pyproject.toml
	git add pyproject.toml
	git commit -m "update version to $(NEXT)"
	git tag v$(NEXT)
	git push origin main
	git push origin v$(NEXT)
	gh release create v$(NEXT) --generate-notes

docker-build:
	docker build -t pyav-wrapper -f Dockerfile .

docker-run: docker-build
	docker context use rtx4090
	docker stop pyav-wrapper || true
	docker rm pyav-wrapper || true
	docker run --rm -t --gpus=all --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --name pyav-wrapper pyav-wrapper
