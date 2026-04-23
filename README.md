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

## 無音時リアクション機能

一定時間無音が続いた際に、配信の間を保つため、しずくが短い定型文を自発的に発話する軽量な機能です。
LLMを使用せず直接TTS（音声合成）を呼び出すため、通常会話を阻害しません。

### 動作仕様
- マイク録音 (`record_utterance`) のタイムアウトを契機に無音判定を行うため、実際の発火時刻は設定値（interval）ピッタリではなく、タイムアウトの粒度に依存します。
- LLMを介さず、設定された定型文から直接発話します。
- 同じ文言が連続して選ばれない仕組みになっています。
- ユーザーとの通常会話が発生した場合は、無音タイマーがリセットされ、通常会話が常に優先されます。

### 設定項目 (`config/config.py`)
| 設定名 | 初期値 | 役割・変更例 |
| --- | --- | --- |
| `SILENT_REACTION_ENABLED` | `True` | 機能のON/OFF。<br>無効にする場合は `False` に変更します。 |
| `SILENT_REACTION_INTERVAL_SEC` | `60.0` | 無音と判定するまでの秒数。<br>長くする場合は `120.0` (2分) などに変更します。 |
| `SILENT_REACTION_PHRASES` | `("見ていますよ。", ...)` | ランダムに選ばれる定型文のリスト。<br>お好みのセリフを追加・変更できます。 |

### 注意事項
- `SILENT_REACTION_PHRASES` を空 `()` にした場合は、機能が有効でも発話しません。
- `INTERVAL_SEC` を短くしすぎると配信の邪魔になる可能性があるため、適切な間隔を設定してください。
- 配信のノイズにならないよう、文言は控えめなものを推奨します。

### 動作確認手順
1. **アプリ起動**: `python src/shizuku_aituber.py` でアプリを通常通り起動します。
2. **無音発火の確認**: マイクに何も話さず、約60秒（初期設定）経過後に定型文が再生されるか確認します。
3. **LLM非使用の確認**: 無音リアクション発火時のコンソールログに `[LLM]` の通信ログが出力されていないことを確認します。
4. **タイマーリセットの確認**: 無音リアクション後、普通に話しかけて応答をもらい、そこからさらに60秒以上経過しないと次の無音リアクションが発火しないことを確認します。
5. **ON/OFFの確認**: `config.py` で `SILENT_REACTION_ENABLED = False` に変更して再起動し、無音のまま60秒経過しても発火しないことを確認します。


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