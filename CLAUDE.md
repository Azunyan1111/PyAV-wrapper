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
make test-stream   # SRT受信→グレースケール→SRT送信の統合テスト
make test-output   # SRT受信→グレースケール→ファイル書き出しの統合テスト
make test-writer-stream  # Listener→Writer 統合テスト（グレースケール+SRT送信）
make test-whep     # raw-subprocess pipe の WHEP リスナーテスト
make test-whip     # raw-subprocess pipe の WHIP ライターテスト
```

## 環境変数（.env）

- `SRT_URL`: SRT入力ストリームURL
- `SRT_OUTPUT_URL`: SRT出力ストリームURL

---

## トラブルシューティング: 音声・映像同期問題

### 問題の症状

StreamWriter経由で出力した映像・音声の同期がずれる。具体的には：
- 再生速度が異常（例: 1秒の音が1.4秒かかる）
- 再生開始タイミングがずれる
- ファイル出力時のdurationが異常に短い（例: 330フレーム出力したのにduration=0.016秒）

### 根本原因: PTSとtime_baseの不一致

PyAVでは、フレームの再生タイミングは**PTS（Presentation Time Stamp）**と**time_base**の組み合わせで決定される。

```
実際の再生時刻（秒） = PTS × time_base
```

例:
- `PTS=3000, time_base=1/90000` → 3000 × (1/90000) = 0.0333秒
- `PTS=1, time_base=1/30` → 1 × (1/30) = 0.0333秒

**問題が発生するケース:**

SRTストリーム（MPEG-TS形式）から受信したフレームは`time_base=1/90000`を持つ。
このフレームのPTSを手動で`0, 1, 2, ...`と連番に書き換えると：

```python
# NG: PTSを手動で連番に設定
wrapped_frame.frame.pts = frame_count  # 0, 1, 2, ...
```

- 入力: `PTS=1, time_base=1/90000` → 0.000011秒
- 期待: `PTS=1, time_base=1/30` → 0.0333秒

結果として全フレームがほぼ同時刻（0秒付近）に再生される。

### 正しい実装: 元のPTS/time_baseを維持

```python
# OK: 元のPTSとtime_baseをそのまま使用
for packet in video_stream.encode(wrapped_frame.frame):
    container.mux(packet)

# OK: 音声も同様
for packet in audio_stream.encode(wrapped_audio.frame):
    container.mux(packet)
```

**絶対にやってはいけないこと:**

```python
# NG: PTSを手動で設定
wrapped_frame.frame.pts = video_pts
video_pts += 1

# NG: PTSをNoneに設定（音声の場合、一見動くが同期がずれる）
wrapped_audio.frame.pts = None

# NG: time_baseを変更
wrapped_frame.frame.time_base = Fraction(1, 30)
```

### 調査方法

#### 1. ffprobeでパケットのPTSを確認

```bash
ffprobe -v error -show_packets -select_streams v:0 output.ts | head -60
```

正常な出力例（PTSが増加している）:
```
[PACKET]
pts=0
pts_time=0.000000
[/PACKET]
[PACKET]
pts=3000
pts_time=0.033333
[/PACKET]
[PACKET]
pts=6000
pts_time=0.066667
[/PACKET]
```

異常な出力例（PTSが全て同じ）:
```
[PACKET]
pts=0
pts_time=0.000000
[/PACKET]
[PACKET]
pts=0
pts_time=0.000000
[/PACKET]
```

#### 2. ffprobeでストリーム情報を確認

```bash
ffprobe -v error -show_format -show_streams output.ts
```

確認ポイント:
- `duration`: 期待される長さか
- `time_base`: 映像は通常`1/90000`（MPEG-TS）または`1/30`等
- `sample_rate`: 音声のサンプルレート

#### 3. Pythonコードでデバッグ出力

```python
# フレームの情報を確認
print(f"frame.pts={frame.pts}, frame.time_base={frame.time_base}")

# エンコード後のパケット情報を確認
for packet in video_stream.encode(frame):
    print(f"packet.pts={packet.pts}, packet.time_base={packet.time_base}")
```

### time_baseの典型的な値

| コンテナ/形式 | 映像time_base | 音声time_base |
|--------------|---------------|---------------|
| MPEG-TS (SRT) | 1/90000 | 1/90000 |
| MP4 | 1/fps (例: 1/30) | 1/sample_rate |
| エンコーダー出力 | 1/fps | 1/sample_rate |

### 音声速度が1.4倍遅い問題の計算

入力: 48000Hz音声、time_base=1/90000
- 1秒分のサンプル数: 48000
- 1秒のPTS増分: 48000 × (90000/48000) = 90000

出力のtime_baseが1/48000の場合:
- PTS=90000で再生時刻 = 90000 / 48000 = 1.875秒

→ 1秒の音声が約1.875秒（≒1.4秒）で再生される

### StreamWriterの設計原則

1. **入力フレームのPTS/time_baseは絶対に変更しない**
2. **PyAVのエンコーダーが自動的にtime_base変換を行う**
3. **手動でPTS管理が必要なのは、新規フレームを作成する場合のみ**

### 関連ファイル

- `src/pyav_wrapper/stream_writer.py`: StreamWriterの実装
- `tests/test_stream_writer.py`: 統合テスト
- `tests/test_stream_listener.py`: 比較用（正常動作するテスト）

### 参考: crop_center()でのPTS維持

`WrappedVideoFrame.crop_center()`は新しいフレームを作成するが、元のPTS/time_baseをコピーしている:

```python
new_frame = av.VideoFrame(crop_width, crop_height, self._frame.format.name)
new_frame.pts = self._frame.pts  # PTSをコピー
if self._frame.time_base is not None:
    new_frame.time_base = self._frame.time_base  # time_baseもコピー
```

これにより、クロップ後も元の再生タイミングが維持される。
