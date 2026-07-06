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

> [!WARNING]
> `config/config.py` には Twitch のアクセストークンなどの秘匿情報が含まれるため、Git の管理から除外（`.gitignore` に追加）されています。
> 新規セットアップ時は、`config/config.example.py` をコピーして `config/config.py` を作成してください。

| 設定名 | 初期値 | 役割・変更例 |
| --- | --- | --- |
| `SILENT_REACTION_ENABLED` | `True` | 機能のON/OFF。<br>無効にする場合は `False` に変更します。 |
| `SILENT_REACTION_INTERVAL_SEC` | `180.0` | 無音と判定するまでの秒数。<br>長くする場合は `120.0` (2分) などに変更します。 |
| `SILENT_REACTION_PHRASES` | `("見ていますよ。", ...)` | ランダムに選ばれる定型文のリスト。<br>お好みのセリフを追加・変更できます。 |

### 注意事項
- `SILENT_REACTION_PHRASES` を空 `()` にした場合は、機能が有効でも発話しません。
- `INTERVAL_SEC` を短くしすぎると配信の邪魔になる可能性があるため、適切な間隔を設定してください。
- 配信のノイズにならないよう、文言は控えめなものを推奨します。

### 動作確認手順
1. **アプリ起動**: `python src/shizuku_aituber.py` でアプリを通常通り起動します。
2. **無音発火の確認**: マイクに何も話さず、約180秒（初期設定）経過後に定型文が再生されるか確認します。
3. **LLM非使用の確認**: 無音リアクション発火時のコンソールログに `[LLM]` の通信ログが出力されていないことを確認します。
4. **タイマーリセットの確認**: 無音リアクション後、普通に話しかけて応答をもらい、そこからさらに180秒以上経過しないと次の無音リアクションが発火しないことを確認します。
5. **ON/OFFの確認**: `config.py` で `SILENT_REACTION_ENABLED = False` に変更して再起動し、無音のまま180秒経過しても発火しないことを確認します。


## 会話頻度制御

しずくが配信者の独り言すべてに反応しないよう、AI発話にクールダウンと反応率制御を入れています。視聴者コメントは有効時に優先処理され、「しずく」と呼ばれた発話は配信者音声でも優先して返答します。

| 設定名 | 初期値 | 役割・変更例 |
| --- | --- | --- |
| `AI_SPEECH_COOLDOWN_SEC` | `8.0` | AIが連続発話しすぎないための最短間隔です。 |
| `STREAMER_RESPONSE_PROBABILITY` | `0.25` | 配信者音声へ反応する確率です。独り言への反応を減らします。 |
| `SHIZUKU_CALL_KEYWORDS` | `("しずく", "シズク", "雫")` | 呼びかけとして扱うキーワードです。 |
| `STREAMER_FORCE_REPLY_KEYWORDS` | `("しずく", "どう思う", "見て", "聞いて")` | 反応率制御を超えて返答するキーワードです。 |

## Twitchコメント優先

`TWITCH_COMMENT_ENABLED = True` の場合、Twitchコメントreaderを起動し、コメントqueueをメインループで確認します。コメントがある場合は音声入力より先にLLM/TTSへ渡します。Twitchを無効にしている場合は従来通り音声入力だけで動作します。

| 設定名 | 初期値 | 役割・変更例 |
| --- | --- | --- |
| `TWITCH_COMMENT_ENABLED` | `False` | Twitchコメント取得のON/OFFです。 |
| `TWITCH_COMMENT_PRIORITY` | `True` | Twitchコメントを音声入力より優先します。 |
| `TWITCH_COMMENT_COOLDOWN_SEC` | `3.0` | Twitchコメント応答の最短間隔です。 |

## GAME_MODE

`GAME_MODE` で通常配信と対戦ゲーム中の応答方針を切り替えます。変更後はアプリを再起動してください。

| 設定名 | 初期値 | 役割・変更例 |
| --- | --- | --- |
| `GAME_MODE` | `"normal"` | `"normal"` は落ち着いた通常配信、`"battle"` は短い応援・共感・状況リアクション寄りです。 |

## 謝罪抑制

強い言葉への返答が「すみません」「ごめんなさい」だけに寄った場合、短い応援や切り替えの言葉へ置換します。危険な内容や個人情報などに触れない既存の安全方針は維持します。

| 設定名 | 初期値 | 役割・変更例 |
| --- | --- | --- |
| `APOLOGY_SUPPRESSION_ENABLED` | `True` | 謝罪連発の抑制を有効にします。 |
| `APOLOGY_REPLACEMENT_PHRASES` | `("落ち着いていきましょう。", ...)` | 謝罪の代わりに使う短文候補です。 |


## 画面イベント検出（実験機能）

スプラトゥーンの死亡表示を軽量に検出して、短い固定文で反応する実験機能です。Phase 1では中央付近のcrop領域、`やられた!` テンプレート、白文字分布、黒カード量から判定します。Phase 2では任意で死亡表示周辺だけOCRできます。Twitchコメントより優先されず、発話連発は専用クールダウンで抑制します。

### 設定項目 (`config/config.py`)

| 設定名 | 初期値 | 役割・変更例 |
| --- | --- | --- |
| `SCREEN_EVENT_ENABLED` | `False` | 画面イベント検出のON/OFFです。まず動作確認時だけ `True` にします。 |
| `SCREEN_EVENT_MODE` | `"death_detect_only"` | `"death_detect_only"` または `"death_ocr"`。OCRは死亡検出後だけ実行します。 |
| `SCREEN_CAPTURE_INTERVAL_SEC` | `0.25` | 画面を確認する間隔です。短時間の死亡表示を拾うため、Phase 1では `0.25` を標準にします。 |
| `SCREEN_CAPTURE_MONITOR_INDEX` | `1` | `mss` のモニター番号です。画面が違う場合に変更します。 |
| `SCREEN_CAPTURE_WIDTH` / `SCREEN_CAPTURE_HEIGHT` | `640` / `360` | 判定用に縮小するサイズです。 |
| `SCREEN_CAPTURE_REGION` | `None` | 指定モニター内の一部だけを見る場合に `(left, top, width, height)` で指定します。 |
| `SCREEN_EVENT_DEBUG_LOG` | `False` | 検出スコアや処理時間の詳細ログを出します。 |
| `SCREEN_EVENT_LOG_EVERY_SEC` | `10.0` | debug無効時の低スコアログ間隔です。 |
| `DEATH_EVENT_COOLDOWN_SEC` | `20.0` | 死亡イベント反応の最短間隔です。検出頻度とは別です。 |
| `DEATH_EVENT_MIN_TEMPLATE_SCORE` | `0.55` | 主判定のテンプレート一致しきい値です。 |
| `DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE` | `0.40` | 補助判定でも必要な最低テンプレート一致度です。 |
| `DEATH_EVENT_WHITE_RATIO_MIN` / `MAX` | `0.015` / `0.18` | 白文字量の下限/上限です。リザルトなど白文字過多を除外します。 |
| `DEATH_EVENT_SHAPE_WHITE_RATIO_MIN` / `MAX` | `0.04` / `0.14` | 補助判定用の白文字量です。ロビーのリザルト表示の誤検知を抑えます。 |
| `DEATH_EVENT_ROI` | `(0.32, 0.21, 0.67, 0.54)` | 死亡表示全体を見る正規化cropです。 |
| `DEATH_EVENT_TEXT_ROI` | `(0.42, 0.33, 0.59, 0.46)` | `やられた!` 付近を見る正規化cropです。 |
| `DEATH_EVENT_TEMPLATE_PATH` | `"assets/templates/splatoon_death_yarareta.png"` | 死亡表示テンプレート画像です。 |
| `DEATH_EVENT_OCR_ROI` | `(0.34, 0.25, 0.66, 0.48)` | OCR対象の正規化cropです。死亡表示周辺だけを読みます。 |
| `DEATH_EVENT_OCR_LANG` | `"jpn+eng"` | TesseractのOCR言語です。日本語データが必要です。 |
| `DEATH_EVENT_OCR_CONFIG` | `"--psm 6"` | Tesseractのページ分割設定です。 |
| `DEATH_EVENT_OCR_MIN_CONFIDENCE` | `0.0` | OCR結果をログ採用する最低confidenceです。 |
| `DEATH_EVENT_OCR_SCALE` | `3.0` | OCR前にcropを拡大する倍率です。 |
| `DEATH_EVENT_OCR_PREPROCESS_MODE` | `"default"` | OCR前処理です。`"threshold"`, `"adaptive"`, `"invert_threshold"`, `"sharpen_threshold"` も試せます。 |
| `DEATH_EVENT_OCR_TESSERACT_CMD` | `""` | Tesseract実行ファイルのパスです。PATHが通っていない場合に指定します。 |
| `DEATH_EVENT_OCR_SAVE_DEBUG_IMAGES` | `False` | OCR用cropと前処理後画像を保存します。通常配信では `False` にします。 |
| `DEATH_EVENT_OCR_DEBUG_DIR` | `"debug/ocr"` | OCR debug画像の保存先です。 |
| `DEATH_EVENT_USE_CATEGORY_REACTIONS` | `True` | OCR結果を補助情報としてカテゴリ別固定文を選びます。LLMは使いません。 |
| `DEATH_EVENT_OCR_CATEGORY_MIN_CONFIDENCE` | `60.0` | 武器カテゴリ推定に使う最低OCR confidenceです。低い場合は `unknown` にします。 |
| `DEATH_EVENT_CATEGORY_DEBUG_LOG` | `False` | カテゴリ推定の詳細ログを出します。 |
| `DEATH_EVENT_WEAPON_KEYWORDS` | `None` | `None` の場合は標準の武器カテゴリ辞書を使います。 |
| `DEATH_EVENT_REACTIONS_BY_CATEGORY` | `None` | `None` の場合は標準のカテゴリ別固定文を使います。 |
| `DEATH_EVENT_REACTION_PHRASES` | `("今のはきついですね。", ...)` | 検出時にランダム再生する固定文です。 |

標準設定:

```python
SCREEN_CAPTURE_INTERVAL_SEC = 0.25
```

`0.25` は `SCREEN_EVENT_DEBUG_LOG = True` で `capture elapsed` が安定して `0.25s` 未満であることを確認してください。負荷が気になる場合は `0.5` に戻して、動画検証で取り逃しが許容できるか確認します。

### サンプル検証

誤検知した画面は `negative`、未検知だった死亡画面は `positive/death` に追加します。

```text
assets/screen_samples/
  positive/death/
  negative/normal/
  negative/result/
  negative/lobby/
  negative/map/
  negative/respawn/
  negative/special_ui/
  unknown/
```

一括検証:

```bash
python tools\evaluate_screen_events.py
```

CSV出力:

```bash
python tools\evaluate_screen_events.py --csv screen_event_eval.csv
```

出力には `final_score`, `template_score`, `shape_score`, `dark_ratio`, `white_ratio`, `cols_with_white`, `rows_with_white`, `reason` が含まれます。

### OCR検証（Phase 2）

OCRは死亡検出が成立した後だけ実行されます。固定文リアクションは維持し、OCR結果はログとイベントdetailsに入るだけです。LLMには渡しません。

```python
SCREEN_EVENT_MODE = "death_ocr"
```

`SCREEN_EVENT_MODE = "death_detect_only"` の場合は OCR を実行しないため、Tesseract本体、日本語言語データ、`pytesseract` が無くても従来どおり死亡検出だけで動作します。

OCRには `pytesseract` と Tesseract本体が必要です。Tesseract本体にPATHが通っていない場合は、次のように指定します。

```python
DEATH_EVENT_OCR_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

日本語を読む場合は Tesseract の `jpn` 言語データも必要です。環境が無い場合、死亡検出と固定文リアクションは継続し、OCRだけ `[SCREEN] OCR disabled` または `[SCREEN] OCR skip` になります。

単体画像のOCR確認でも `config/config.py` の `DEATH_EVENT_OCR_TESSERACT_CMD` が使われます。実際に使われるTesseract実行ファイルを確認する場合:

```bash
python src\screen_event_detector.py E:\capture\death_03.png --ocr --debug-ocr
```

一時的に別のTesseractを使う場合は `--tesseract-cmd` で上書きできます。

OCR crop画像を保存してROIを確認:

```bash
python src\screen_event_detector.py E:\capture\death_03.png --ocr --debug-ocr --save-ocr-debug
```

保存先は既定で `debug/ocr/` です。`*_crop.png` でOCR対象領域がズレていないかを先に確認し、ズレている場合は `DEATH_EVENT_OCR_ROI` を調整します。単体確認では `--ocr-roi` で一時的にROIを上書きできます。

```bash
python src\screen_event_detector.py E:\capture\death_03.png --ocr --debug-ocr --save-ocr-debug --ocr-preprocess-mode sharpen_threshold --ocr-roi 0.42,0.34,0.59,0.47
```

ROI調整の目安:

- 黒カード外の背景や余計なUIが入っている場合は、ROIを狭めます。
- `やられた!` の白文字が切れている場合は、ROIを少し広げます。
- 武器名やプレイヤー名まで読みたい場合は、上段も含めます。
- Phase3で使う場合も、まずは `やられた!` 検出補助として扱い、名前や武器名は信頼しすぎないでください。

複数ROIを一括比較する場合は `--compare-ocr-rois` を使います。ROI自体がカンマ区切りなので、複数ROIはセミコロンで区切ります。

```bash
python tools\evaluate_screen_events.py --samples E:\capture --ocr --compare-ocr-modes sharpen_threshold --compare-ocr-rois "0.34,0.25,0.66,0.48;0.42,0.34,0.59,0.47;0.38,0.28,0.63,0.48" --save-ocr-debug --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" --csv screen_event_ocr_roi_eval.csv
```

動画でも同じROI比較ができます。

```bash
python tools\evaluate_screen_event_video.py C:\Users\y-aka\Videos\splatoon_battle_02.mp4 --compare 0.25 --expected-times 133,183,333,373 --ocr --compare-ocr-modes sharpen_threshold --compare-ocr-rois "0.34,0.25,0.66,0.48;0.42,0.34,0.59,0.47;0.38,0.28,0.63,0.48" --save-ocr-debug --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" --csv screen_event_video_ocr_roi_eval.csv
```

ROIが合っているのに誤読する場合は、前処理モードを比較します。

```bash
python tools\evaluate_screen_events.py --samples E:\capture --ocr --compare-ocr-modes default,threshold,adaptive,invert_threshold,sharpen_threshold --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" --csv screen_event_ocr_modes_eval.csv
```

CSVには `ocr_preprocess_mode`, `ocr_roi`, `ocr_scale`, `ocr_text`, `ocr_confidence`, `ocr_reason` が含まれます。OCR結果が不安定、または誤読が多い場合、Phase3では `ocr_text` をそのままLLMへ渡さず、短いイベント種別だけを使うか、OCR結果を補助情報扱いにしてください。

静止画でOCR列まで確認:

```bash
python tools\evaluate_screen_events.py --samples assets\screen_samples --ocr --csv screen_event_ocr_eval.csv
```

動画でOCR列まで確認:

```bash
python tools\evaluate_screen_event_video.py C:\Users\y-aka\Videos\splatoon_battle_02.mp4 --compare 0.25 --expected-times 133,183,333,373 --ocr --csv screen_event_video_ocr_eval.csv
```

CSVには `ocr_text`, `ocr_confidence`, `ocr_reason` が含まれます。OCR品質は次のように見ます。

- 読めた: `ocr_reason=ok` で `ocr_text` に武器名や `やられた` が自然に含まれる
- 一部読めた: `ocr_reason=ok` だが文字欠けや一部だけ
- 誤読: `ocr_reason=ok` だが実画面と違う
- 空: `ocr_reason=empty`
- OCR未実行/skip: `ocr_reason=disabled`, `pytesseract_unavailable`, `ocr_error`, `low_confidence`

### カテゴリ別固定文リアクション（Phase 3）

Phase 3では、死亡イベントにLLMを使わず、OCR結果を補助情報として武器カテゴリを推定し、カテゴリ別の固定文を選びます。低遅延・低負荷で、OCR誤読をそのまま発話しにくくし、LLMリソースをTwitchコメントや「しずく」呼びかけに残すためです。

OCR結果は信頼度つきの補助情報です。`DEATH_EVENT_OCR_CATEGORY_MIN_CONFIDENCE` 未満、OCR失敗、キーワード不一致の場合は `unknown` に落とし、汎用固定文で反応します。武器名やプレイヤー名を断定して発話しません。

カテゴリ辞書や固定文を調整したい場合は、`config/config.py` に `DEATH_EVENT_WEAPON_KEYWORDS` または `DEATH_EVENT_REACTIONS_BY_CATEGORY` を追加します。未指定または `None` の場合は `src/screen_event_reactions.py` の標準辞書を使います。

評価CSVには以下の列が追加されます。

- `normalized_ocr_text`
- `weapon_category`
- `emotion_category`
- `selected_reaction`
- `reaction_source`
- `matched_keywords`
- `category_reason`

`unknown` の低頻度LLM利用は将来拡張です。`DEATH_EVENT_UNKNOWN_USE_LLM` はデフォルト `False` で、Phase 3ではLLMを呼びません。

### 入力取得の設計メモ

画面イベント検出は、将来OBS入力へ差し替えやすいように責務を分けています。

- `FrameSource`: フレーム取得だけを担当します。
- `MSSFrameSource`: 現在のディスプレイキャプチャ実装です。
- `DeathDetector`: 渡された画像フレームから死亡検出とOCRを行います。入力元がmssかOBSかは知りません。
- `ScreenEventDetector`: intervalごとに `FrameSource` からフレームを読み、`DeathDetector` に渡し、検出時にevent queueへ投入します。

静止画検証、動画検証、mss live capture は同じ `detect_death_event()` と `ocr_death_text()` を使います。

将来の `FrameSource` 候補:

- `MSSFrameSource`: 現在のディスプレイキャプチャ
- `VideoFileFrameSource`: 検証動画用
- `OBSVirtualCameraFrameSource`: Phase4候補
- `OBSWindowOrProjectorFrameSource`: OBSプレビュー/プロジェクターをmssで読む代替候補
- `OBSSourceFrameSource`: OBS WebSocketやOBS拡張による将来候補

### Phase4 OBS入力方針

現在はディスプレイキャプチャ前提のため、ゲーム画面を表示しておく必要があります。将来的な理想はOBS映像ソースから直接フレームを取得し、画面表示に依存せず検出できることです。

Phase4の候補:

1. OBS Virtual CameraをOpenCV `VideoCapture` で読む
2. OBSプレビュー/プロジェクターを `mss` で読む
3. OBS WebSocketまたはOBS拡張でソースフレームを取得する

最初の候補は OBS Virtual Camera が現実的です。理由は、死亡検出/OCR/queue投入の処理を変えず、`FrameSource` だけ差し替えればよいためです。Phase4ではOBS連携を追加しますが、Phase2では実装しません。

### 動画検証

静止画ではなく、録画した配信映像から指定間隔ごとにフレームを抽出して検出結果を比較できます。本番の `mss` キャプチャ処理には影響しません。

1.5秒、0.5秒、0.25秒をまとめて比較:

```bash
python tools\evaluate_screen_event_video.py C:\Users\y-aka\Videos\splatoon_battle_01.mp4 --compare 1.5,0.5,0.25
```

CSV出力:

```bash
python tools\evaluate_screen_event_video.py C:\Users\y-aka\Videos\splatoon_battle_01.mp4 --compare 1.5,0.5,0.25 --csv screen_event_video_eval.csv
```

検出フレームを保存:

```bash
python tools\evaluate_screen_event_video.py C:\Users\y-aka\Videos\splatoon_battle_01.mp4 --compare 1.5,0.5,0.25 --save-detected-dir screen_event_frames\detected
```

誤検知候補の確認用に、高スコアだが非検出だったフレームを保存:

```bash
python tools\evaluate_screen_event_video.py C:\Users\y-aka\Videos\splatoon_battle_01.mp4 --compare 1.5,0.5,0.25 --save-high-score-dir screen_event_frames\review --high-score-threshold 0.30
```

CSVには `timestamp_sec`, `frame_index`, `detected`, `emitted`, `final_score`, `template_score`, `dark_ratio`, `white_ratio`, `cols_with_white`, `rows_with_white`, `reason` が含まれます。`detected` は生の検出結果、`emitted` は `DEATH_EVENT_COOLDOWN_SEC` 相当のクールダウンを通した場合に発話イベントとして扱われるかを示します。

目検で検出すべき時刻が分かっている場合:

```bash
python tools\evaluate_screen_event_video.py C:\Users\y-aka\Videos\splatoon_battle_02.mp4 --compare 1.5,0.5,0.25 --expected-times 133,183,333,373 --expected-tolerance-sec 3
```

`expected_check` には、各intervalで目検時刻を拾えたか、目検リスト外で `emitted` になった候補があるかが出力されます。

### 動作確認手順

1. `pip install -r requirements.txt` で `mss` と `opencv-python` を入れます。
2. `config.py` で `SCREEN_EVENT_ENABLED = True` にします。
3. `python src\shizuku_aituber.py` を起動します。
4. 起動時に `[SCREEN] enabled` と `[SCREEN] capture started` が出ることを確認します。
5. 死亡表示が出たときに `[SCREEN] detected type=death` と `[EVENT] source=screen_event` が出て、固定文が再生されることを確認します。
6. Twitchコメントがある場合は `[SKIP] screen_event priority lower than twitch_comment` となり、コメント優先になることを確認します。
7. `SCREEN_EVENT_ENABLED = False` に戻すと `[SCREEN] disabled` だけになり、画面検出は停止します。

### 注意事項

- ゲーム画面がモニター全体を占めていない場合は、`SCREEN_CAPTURE_REGION` でゲーム映像部分だけを指定してください。
- Phase 1ではOCRしないため、倒された相手名や武器名は読み取りません。
- `SCREEN_CAPTURE_INTERVAL_SEC` は検出頻度、`DEATH_EVENT_COOLDOWN_SEC` は発話頻度です。
- 画面全体の常時OCR、毎フレーム解析、Twitchコメントより画面イベントを優先する設計は行いません。


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
