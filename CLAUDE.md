# PyAV-wrapper

## プロジェクト概要
PyAVのav.VideoFrame/av.AudioFrameをラップし、バッファ操作を簡素化するライブラリ。
SRTストリームの受信・加工・送信を、フレーム情報を維持したまま行うことができる。

## 主要クラス

### WrappedVideoFrame / WrappedAudioFrame
PyAVのフレームをラップし、シンプルなバッファ操作APIを提供する。

**存在意義:**
- 複雑なパディング処理をカプセル化
- 元のav.VideoFrame/av.AudioFrameを内部に保持
- PTS、time_base、フォーマット情報など全てのメタデータを維持
- ストリーム処理で同期情報を失わずに加工可能

### StreamListener
PyAVでストリームを受信し、Video/Audioフレームをバッファリングするクラス。

## 重要な設計原則

### フレーム加工時は元のフレームを直接操作する

**絶対にやってはいけないこと:**
```python
# NG: 新しいフレームを作成して情報を破棄
planes = wrapped_frame.get_planes()
new_frame = av.VideoFrame(width, height, "yuv420p")  # 元フレームの情報が全て失われる
new_frame.planes[0].update(...)
new_frame.pts = wrapped_frame.frame.pts  # PTSだけコピーしてもtime_base等が失われる
```

**正しい実装:**
```python
# OK: 元のフレームを直接加工
planes = wrapped_frame.get_planes()

# グレースケール化（U, Vプレーンを128に）
if len(planes) > 1:
    planes[1][:] = 128
if len(planes) > 2:
    planes[2][:] = 128

# 元のフレームに書き戻す
wrapped_frame.set_planes(planes)

# 元のフレームをそのままエンコード（全情報維持）
for packet in video_stream.encode(wrapped_frame.frame):
    output_container.mux(packet)
```

### 音声フレームも元のフレームをそのまま使用
```python
audio_frame = wrapped_audio.frame
for packet in audio_stream.encode(audio_frame):
    output_container.mux(packet)
```

## テストコマンド

```bash
make test          # 全テスト実行
make test-stream   # SRT送信テスト（60秒間グレースケール+音声送信）
make test-output   # ファイル書き出しテスト（10秒間グレースケール+音声）
```

## 環境変数（.env）

- `SRT_URL`: SRT入力ストリームURL
- `SRT_OUTPUT_URL`: SRT出力ストリームURL
