# 構成
- `src/pyav_wrapper/`: ライブラリ本体（audio_frame.py, video_frame.py, stream_listener.py, stream_writer.py, raw_subprocess_pipe_stream_listener.py, raw_subprocess_pipe_stream_writer.py, __init__.py, py.typed）
- `tests/`: pytest テスト群（test_*.py とテスト用出力ファイル）
- ルート直下: 実行用/実験用スクリプト（stream_listener.py, thread_ffmpeg_streamer.py, raw_subprocess_pipe_stream_writer.py など）
- ドキュメント: avframe-direct-manipulation.md