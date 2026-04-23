## 概要
ゲーム配信の相方として動作するAI（PNGTuber）です。
音声入力からリアクションを生成し、音声とキャラクターとして出力します。

## システム構成
マイク
↓
STT（Whisper）
↓
LLM（Ollama）
↓
TTS（AivisSpeech）
↓
Voicemeeter
↓
Veadotube
↓
OBS

## 必要環境
- Python 3.10+
- GPU（推奨）
- Ollama
- AivisSpeech
- Voicemeeter
- OBS

## 制約
- 応答に約3秒の遅延あり
- STTは発話に依存（ゆっくり話す必要あり）
- VRAM使用量が高い（約7.5GB）

## 起動方法

### 1. 初回セットアップ

```bash
cd E:\AI\shizuku\aituber

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 事前起動
#### LLM（Ollama）
```bash
ollama run phi4-mini
```
※ 初回のみモデルダウンロードあり
※ 起動後はバックグラウンドで常駐

#### TTS（AivisSpeech）

AivisSpeechを起動する（アプリを開くだけ）

### 3. デバイス確認（venv上）
#### 一覧表示
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```
#### INPUT_DEVICE（マイク）

以下の条件で選ぶ

* (2 in, 0 out) など input がある
* マイク名が一致

例：
```bash
4 マイク (Anker PowerCast M300), MME (2 in, 0 out)
```
設定：
```python
INPUT_DEVICE = 4
OUTPUT_DEVICE（音声出力）
```
以下の条件で選ぶ

* (0 in, 2 out) または (0 in, 8 out)
* Voicemeeter を使用する場合は以下を選択

例：
```bash
25 Voicemeeter AUX Input, MME (0 in, 8 out)
```
設定：
```python
OUTPUT_DEVICE = 25
```
### 4. 起動(venv上)
```bash
python src\shizuku_aituber.py
```
### 5. 正常時ログ
```bash
=== 月野しずく AITuber ===
マイク待機中…（話しかけてください）
```
### 6. 終了
```
Ctrl + C
```
### 7. トラブルシュート
#### マイクが反応しない
* INPUT_DEVICEが間違っている
#### 音が出ない
* OUTPUT_DEVICEが間違っている
* Voicemeeter設定を確認
#### LLMが応答しない
```bash
ollama list
```
でモデルを確認

### 8. フル起動手順（まとめ）
```
① venv activate
② Ollama起動
③ AivisSpeech起動
④ python 実行
```