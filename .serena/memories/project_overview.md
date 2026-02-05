# プロジェクト概要
- PyAVのAVFrame（VideoFrame/AudioFrame）を直接操作し、メモリコピーを最小化した高効率な映像・音声処理パイプラインを目指す。
- 目的は「パイプライン処理を簡潔かつ高速にする」こと。
- ラップクラス（WrappedVideoFrame/WrappedAudioFrame）やストリーム入出力コンポーネントを提供する。