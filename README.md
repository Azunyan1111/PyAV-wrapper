# PyAV-wrapper

## 概要
PyAVの`av.VideoFrame`/`av.AudioFrame`をラップし、バッファ操作を簡素化するライブラリです。
SRT等のストリーム受信・加工・送信を、PTS/time_baseなどのメタデータを維持したまま行うことを目的としています。

## 主要コンポーネント

### WrappedVideoFrame
- `get_buffer`/`set_buffer`: Y planeの取得・更新
- `get_planes`/`set_planes`: 各plane（Y/U/V等）の取得・更新
- `crop_center`: 中央クロップした新しいフレームを返却（元のPTS/time_baseをコピー）

### WrappedAudioFrame
- `get_buffer`/`set_buffer`: フレーム全体の取得・更新（planarは`(channels, samples)`）
- `get_planes`/`set_planes`: 各チャネルの取得・更新

### StreamListener
- URL（`srt://`, `rtmp://`, `udp://`等）からVideo/Audioフレームをデコードしてキューへ格納
- 指定された`width`/`height`に合わせてリサイズ
- フレーム更新監視と自動再接続

### StreamWriter
- `WrappedVideoFrame`/`WrappedAudioFrame`をSRTまたはファイルへ送信
- 元のPTS/time_baseを維持したままエンコード
- 再接続（ペーシングは外部で実施することを想定）

### RawSubprocessPipeStreamListener / RawSubprocessPipeStreamWriter
- サブプロセスのstdout/stdinパイプをMatroska（MKV）として扱う派生クラス
- rawvideo + PCMの読み書きに対応

## 使い方（例）

```python
from pyav_wrapper import StreamListener, StreamWriter

listener = StreamListener("srt://host:port?mode=caller&latency=120", width=640, height=480)
writer = StreamWriter("srt://host:port?mode=caller&latency=120", width=640, height=480)

while listener.is_running:
    video_frames = listener.pop_all_video_queue()
    for wrapped_frame in video_frames:
        planes = wrapped_frame.get_planes()
        if len(planes) > 1:
            planes[1][:] = 128
        if len(planes) > 2:
            planes[2][:] = 128
        wrapped_frame.set_planes(planes)
        writer.enqueue_video_frame(wrapped_frame)

    audio_frames = listener.pop_all_audio_queue()
    for wrapped_audio in audio_frames:
        writer.enqueue_audio_frame(wrapped_audio)
```

## 設計上の注意
- フレーム加工時は元のフレームを直接操作してください。
- PTS/time_baseを手動で変更しないでください（同期が崩れる原因になります）。

## 必要条件
- Python 3.12+
- `av>=14.0.0`
- `numpy>=1.20.0`

## テスト

```bash
make test
make test-srt
make test-stream
make test-output
make test-writer-stream
make test-whep
make test-whip
```

## 環境変数（.env）
- `SRT_URL`: SRT入力ストリームURL
- `SRT_OUTPUT_URL`: SRT出力ストリームURL

## 構成
- `src/pyav_wrapper/`: ライブラリ本体
- `tests/`: テストコード
- `avframe-direct-manipulation.md`: 設計メモ
