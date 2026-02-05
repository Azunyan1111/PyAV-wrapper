# 主要コマンド
- `make test`: `uv run pytest` で通常テスト実行
- `make test-srt`: SRT E2Eテスト（`-m srt`、timeout延長）
- `make test-stream`: stream_listener のSRT統合テストを実行
- `make test-output`: SRT受信→変換→書き出しの統合テスト
- `make test-writer-stream`: listener→writerの統合テスト
- `make test-whep`: raw_subprocess_pipe_stream_listener のテスト
- `make test-whip`: raw_subprocess_pipe_stream_writer のテスト
- pytest 既定: `-v -n auto -m 'not srt'`, timeout=30, testpaths=tests
- 注意: gitコマンドの実行は本セッションの指示で禁止されている