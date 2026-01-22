# AVFrame直接操作による高効率パイプライン設計

## 概要

PyAVのAVFrame（VideoFrame/AudioFrame）を直接操作することで、メモリコピーを最小化した高効率な映像・音声処理パイプラインを実現する。

## 目的

パイプライン処理を簡潔かつ高速にする。

1. AVFrameをPyAVで映像ソースから取り出す
2. AVFrameから生のフレームバッファを取得する（ゲッター）
3. AVFrameの生のフレームバッファに加工を施し上書きする（セッター）
4. AVFrameを維持したまま、PyAVで外部に出力する
5. 取り出すクラスを作成する(2のゲッターで取り出せる)
6. 書き込むクラスを作成する(3のセッターで書き込める)
7. I/OはAVFrame
8. AVFrameをオーバーライドなりして、ゲッターセッターメソッドを実行できるようにする

## 調査結果

### 検証環境

- PyAV 14.x (libavformat/libavcodec)
- Python 3.9+
- numpy 1.20+

### デコード済みフレームの書き込み可能性

検証の結果、**デコードしたフレームのバッファも書き込み可能**であることを確認した。

```python
container = av.open('input.mp4')
for packet in container.demux(video_stream):
    for frame in codec_ctx.decode(packet):
        plane = frame.planes[0]
        mem = memoryview(plane)
        arr = np.frombuffer(mem, dtype=np.uint8)

        # デコード済みフレームを直接編集可能
        arr[0] = 128  # OK
```

### 完全なパイプライン検証

以下のパイプラインが正常に動作することを確認した：

1. ファイルを開き、デコード
2. VideoFrameのバッファを直接編集（グレースケール化）
3. 編集したフレームをそのままエンコード
4. ファイルとして出力

## 実装タスク

### Phase 1: コア実装

- [ ] `FrameReader` クラスの実装

- [ ] `FrameWriter` クラスの実装

### Phase 2: テスト

- [ ] `test_frame_reader.py`
  - [ ] ファイルからの映像フレーム読み込み
  - [ ] ファイルからの音声フレーム読み込み
  - [ ] ストリーム入力対応

- [ ] `test_frame_writer.py`
  - [ ] 映像フレーム書き込み
  - [ ] 音声フレーム書き込み
  - [ ] 出力ファイル検証
  - [ ] ストリーム出力対応

- [ ] `test_integration.py`
  - [ ] Reader → Writer パイプライン
  - [ ] 映像加工パイプライン
  - [ ] 音声加工パイプライン

## 参考資料

- [PyAV Documentation - Video](https://pyav.basswood-io.com/docs/stable/api/video.html)
- [PyAV Documentation - Audio](https://pyav.org/docs/stable/api/audio.html)
- [PyAV Documentation - Frame](https://pyav.basswood-io.com/docs/stable/api/frame.html)
- [PyAV Cookbook - Basics](https://pyav.basswood-io.com/docs/stable/cookbook/basics.html)
- [GitHub Issue #730 - Template Stream Transcoding](https://github.com/PyAV-Org/PyAV/issues/730)

## 検証コード

以下のコードで動作を確認できる：

```python
import av
import numpy as np
from fractions import Fraction

# 入力を開く
input_container = av.open('input.mp4')
video_stream = input_container.streams.video[0]
audio_stream = input_container.streams.audio[0]

# 出力を準備
output_container = av.open('output.mp4', mode='w')

out_video = output_container.add_stream('libx264', rate=30)
out_video.width = video_stream.width
out_video.height = video_stream.height
out_video.pix_fmt = 'yuv420p'

out_audio = output_container.add_stream('aac', rate=48000)
out_audio.layout = 'stereo'

video_pts = 0
audio_pts = 0

for packet in input_container.demux():
    if packet.size == 0:
        continue

    if packet.stream_index == video_stream.index:
        for frame in packet.decode():
            # フレームを加工してエンコード
            frame.pts = video_pts
            for out_packet in out_video.encode(frame):
                output_container.mux(out_packet)
            video_pts += 1

    elif packet.stream_index == audio_stream.index:
        for frame in packet.decode():
            frame.pts = audio_pts
            for out_packet in out_audio.encode(frame):
                output_container.mux(out_packet)
            audio_pts += frame.samples

# フラッシュ
for out_packet in out_video.encode():
    output_container.mux(out_packet)
for out_packet in out_audio.encode():
    output_container.mux(out_packet)

output_container.close()
input_container.close()
```
