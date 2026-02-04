# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `src/pyav_wrapper/`, which contains the core wrappers and stream components (e.g., `video_frame.py`, `audio_frame.py`, `stream_listener.py`, `stream_writer.py`). Tests are in `tests/` and follow a `test_*.py` naming scheme, with some integration outputs (e.g., `.mp4`, `.ts`) stored alongside the tests. Root-level scripts like `stream_listener.py` and `raw_subprocess_pipe_stream_writer.py` are used for direct execution or integration experiments. Design notes and background are in `avframe-direct-manipulation.md`.

## Build, Test, and Development Commands
Use `uv` to run the test suite and resolve dependencies. Common targets are provided via `make`:
`make test` runs the standard test suite (`uv run pytest`).
`make test-srt` runs SRT end-to-end tests with longer timeouts.
Targeted commands like `make test-whep` and `make test-whip` run specific raw-subprocess pipe tests.
Python packaging uses `uv_build` (see `pyproject.toml`), and the package requires Python 3.12+.

## Coding Style & Naming Conventions
The codebase uses 4-space indentation, type hints, and short docstrings for public methods. Keep module and function names in `snake_case` and class names in `CamelCase` (e.g., `WrappedVideoFrame`). No formatter or linter is configured in `pyproject.toml`, so match the existing style in nearby files.

## Testing Guidelines
Tests run with `pytest`, and SRT-related tests are marked with `srt`. By default, pytest options exclude the SRT marker (`-m 'not srt'`) and use parallel execution (`-n auto`). For SRT tests, use the Makefile targets that set `-m srt` and longer timeouts.

## Commit & Pull Request Guidelines
This repository does not document a specific commit message convention. Keep commits small and descriptive, and include a clear summary and testing notes in PRs. If an established convention exists in current project discussions, align with it.
