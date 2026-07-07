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

配信者の短い感情発話は、LLMではなく固定文で短く返せます。Twitchコメントと「しずく」呼びかけはLLMを優先し、`きつい`、`惜しい`、`ナイス`、`ふざけんな` などの感情リアクションは固定文中心にすることで、謝罪誤爆や長文返答を減らします。`あー`、`うーん` のような短い独り言は基本的にスキップします。

「しずく」と呼びかけられた場合は、通常の短文制限とは別枠で少し長めにLLM応答できます。自己紹介や初見さんへの挨拶などの定型要求は、LLMではなく固定文で安定して返します。

| 設定名 | 初期値 | 役割・変更例 |
| --- | --- | --- |
| `AI_SPEECH_COOLDOWN_SEC` | `8.0` | AIが連続発話しすぎないための最短間隔です。 |
| `STREAMER_RESPONSE_PROBABILITY` | `0.25` | 配信者音声へ反応する確率です。独り言への反応を減らします。 |
| `SHIZUKU_CALL_KEYWORDS` | `("しずく", "シズク", "雫")` | 呼びかけとして扱うキーワードです。 |
| `STREAMER_FORCE_REPLY_KEYWORDS` | `("しずく", "どう思う", "見て", "聞いて")` | 反応率制御を超えて返答するキーワードです。 |
| `STREAMER_FIXED_RESPONSE_ENABLED` | `True` | 配信者の感情発話に固定文で返します。 |
| `STREAMER_LLM_ON_ADDRESS_ONLY` | `True` | 配信者音声では、呼びかけ以外のLLM利用を抑えます。 |
| `STREAMER_FIXED_RESPONSE_RATE` | `0.5` | 固定文候補にする発話へ実際に返す確率です。 |
| `STREAMER_SHORT_NOISE_SKIP` | `True` | 短い独り言や相づちはスキップします。 |
| `STREAMER_FIXED_RESPONSE_COOLDOWN_SEC` | `15.0` | 配信者固定文応答の最短間隔です。 |
| `STREAMER_UTTERANCE_KEYWORDS` | `None` | `None` の場合は標準分類キーワードを使います。 |
| `STREAMER_FIXED_RESPONSE_PHRASES` | `None` | `None` の場合は標準固定文を使います。 |
| `SHIZUKU_ADDRESS_STRONG_KEYWORDS` | `("しずく", "雫", ...)` | 単独で呼びかけとして扱う強一致キーワードです。 |
| `SHIZUKU_ADDRESS_WEAK_KEYWORDS` | `("しず", "静岡", ...)` | 文脈語がある場合だけ呼びかけとして扱う弱一致キーワードです。 |
| `SHIZUKU_ADDRESS_CONTEXT_WORDS` | `("どう", "今の", ...)` | 弱一致を呼びかけ扱いにする文脈語です。 |
| `LLM_ADDRESS_RESPONSE_MAX_CHARS` | `160` | 呼びかけLLM応答だけに使う最大文字数です。 |
| `LLM_ADDRESS_RESPONSE_MAX_SENTENCES` | `4` | 呼びかけLLM応答の最大文数です。 |
| `STREAMER_KEYWORD_FIXED_RESPONSE_ENABLED` | `True` | 自己紹介などの定型キーワード固定応答を有効にします。 |
| `STREAMER_KEYWORD_FIXED_RESPONSES` | `{"self_intro": ...}` | 定型キーワードと固定文を設定します。 |
| `STREAMER_REACTION_DEBUG_LOG` | `False` | 呼びかけ判定や固定応答選択の詳細ログを出します。 |

配信者発話の分類:

- `addressed`: 「しずく」などの呼びかけ。定型キーワードがあれば固定文、なければ長めのLLMで返します。
- `short_noise`: 「あー」「うーん」など。基本スキップします。
- `frustration`: 「きつい」「無理」「やばい」など。固定文で返します。
- `close_call`: 「惜しい」「あと少し」など。固定文で返します。
- `success`: 「ナイス」「勝った」「いいね」など。固定文で返します。
- `angry`: 「ふざけんな」「それはない」など。謝罪せず短い共感・ツッコミで返します。
- `generic`: その他。低頻度で固定文またはスキップします。

呼びかけ判定の例:

| 発話 | 判定 |
| --- | --- |
| `しずく、今のどう？` | `addressed` |
| `雫、これ見て` | `addressed` |
| `しーちゃん、どう思う？` | `addressed` |
| `静岡、今のどう？` | `addressed` |
| `しず、これどう？` | `addressed` |
| `続く、どう思う？` | `addressed` |
| `静岡に行きたい` | 呼びかけにしない |
| `続くと思う` | 呼びかけにしない |

定型キーワード固定応答の例:

```text
しずく、自己紹介して
しずく、初見さんに挨拶して
しずく、あなた誰？
```

上記はLLMではなく `self_intro` の固定文で返します。固定文を変更したい場合は `STREAMER_KEYWORD_FIXED_RESPONSES` の `phrases` を編集してください。

応答優先度:

1. Twitchコメント
2. しずく呼びかけ + キーワード固定応答
3. しずく呼びかけ + 長めLLM応答
4. 配信者感情発話の固定文
5. 画面イベント固定文
6. 無音リアクション

Phase7の低頻度キャラ崩壊モードは将来拡張です。入れる場合も既定OFF、battleモード中だけ、固定文の口調セットを短時間だけ切り替える方針にします。LLMに自由に暴走させず、配信者や視聴者への攻撃、差別、下品、暴力肯定は禁止します。

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
| `SCREEN_FRAME_SOURCE` | `"mss"` | `"mss"` または `"obs_virtual_camera"`。画面イベント検出の入力元です。 |
| `SCREEN_CAPTURE_INTERVAL_SEC` | `0.25` | 画面を確認する間隔です。短時間の死亡表示を拾うため、Phase 1では `0.25` を標準にします。 |
| `SCREEN_CAPTURE_MONITOR_INDEX` | `1` | `mss` のモニター番号です。画面が違う場合に変更します。 |
| `SCREEN_CAPTURE_WIDTH` / `SCREEN_CAPTURE_HEIGHT` | `640` / `360` | 判定用に縮小するサイズです。 |
| `SCREEN_CAPTURE_REGION` | `None` | 指定モニター内の一部だけを見る場合に `(left, top, width, height)` で指定します。 |
| `SCREEN_EVENT_DEBUG_LOG` | `False` | 検出スコアや処理時間の詳細ログを出します。 |
| `SCREEN_EVENT_DEBUG_DIR` | `"debug/screen_event"` | 画面イベントdebug画像の保存先です。 |
| `SCREEN_EVENT_SAVE_DEBUG_FRAMES` | `False` | full frame、ROI crop、overlay、metricsを保存します。通常配信では `False` にします。 |
| `SCREEN_EVENT_SAVE_ROI_CROPS` | `True` | debug保存時にdeath/text ROI cropも保存します。 |
| `SCREEN_EVENT_SAVE_DETECTED_FRAMES` | `False` | raw検出が `detected=True` になった瞬間のフレームを保存します。テスト配信用です。 |
| `SCREEN_EVENT_DETECTED_FRAME_DIR` | `"debug/screen_events"` | raw検出フレームの保存先です。 |
| `SCREEN_EVENT_SAVE_DETECTED_FULL_FRAME` | `True` | raw検出保存時にfull frameを保存します。 |
| `SCREEN_EVENT_SAVE_DETECTED_OVERLAY` | `True` | raw検出保存時にROI overlayを保存します。 |
| `SCREEN_EVENT_SAVE_DETECTED_ROI` | `True` | raw検出保存時にdeath/text ROI cropを保存します。 |
| `SCREEN_EVENT_SAVE_DETECTED_METRICS` | `True` | raw検出保存時にmetrics jsonを保存します。 |
| `SCREEN_EVENT_MAX_DEBUG_FILES` | `200` | debugファイルの最大数です。超過時は古いファイルから削除します。 |
| `SCREEN_EVENT_LOG_EVERY_SEC` | `10.0` | debug無効時の低スコアログ間隔です。 |
| `OBS_VIRTUAL_CAMERA_INDEX` | `0` | OBS Virtual Cameraを読むOpenCVカメラ番号です。 |
| `OBS_VIRTUAL_CAMERA_WIDTH` / `HEIGHT` | `1920` / `1080` | OBS Virtual Cameraへ要求する解像度です。 |
| `OBS_VIRTUAL_CAMERA_FPS` | `30` | OBS Virtual Cameraへ要求するFPSです。 |
| `OBS_VIRTUAL_CAMERA_WARMUP_FRAMES` | `5` | 起動直後に読み捨てるフレーム数です。 |
| `DEATH_EVENT_COOLDOWN_SEC` | `20.0` | 死亡イベント反応の最短間隔です。検出頻度とは別です。 |
| `DEATH_EVENT_MIN_TEMPLATE_SCORE` | `0.55` | 主判定のテンプレート一致しきい値です。 |
| `DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE` | `0.40` | 補助判定でも必要な最低テンプレート一致度です。 |
| `DEATH_EVENT_WHITE_RATIO_MIN` / `MAX` | `0.015` / `0.18` | 白文字量の下限/上限です。リザルトなど白文字過多を除外します。 |
| `DEATH_EVENT_SHAPE_WHITE_RATIO_MIN` / `MAX` | `0.04` / `0.14` | 補助判定用の白文字量です。ロビーのリザルト表示の誤検知を抑えます。 |
| `DEATH_EVENT_ROI` | `(0.32, 0.21, 0.67, 0.54)` | 死亡表示全体を見る正規化cropです。 |
| `DEATH_EVENT_TEXT_ROI` | `(0.42, 0.33, 0.59, 0.46)` | `やられた!` 付近を見る正規化cropです。 |
| `DEATH_EVENT_TEMPLATE_PATH` | `"assets/templates/splatoon_death_yarareta.png"` | 死亡表示テンプレート画像です。 |
| `DEATH_EVENT_ROI_OBS` | `None` | OBS Virtual Camera入力時だけ死亡表示ROIを上書きします。 |
| `DEATH_EVENT_TEXT_ROI_OBS` | `None` | OBS Virtual Camera入力時だけ `やられた!` ROIを上書きします。 |
| `DEATH_EVENT_TEMPLATE_PATH_OBS` | `""` | OBS Virtual Camera入力時だけテンプレート画像を差し替えます。 |
| `DEATH_EVENT_MIN_TEMPLATE_SCORE_OBS` | `None` | OBS Virtual Camera入力時だけ主判定しきい値を上書きします。まずは変更しないでください。 |
| `DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE_OBS` | `None` | OBS Virtual Camera入力時だけ補助判定しきい値を上書きします。まずは変更しないでください。 |
| `DEATH_EVENT_OCR_ROI` | `(0.34, 0.25, 0.66, 0.48)` | OCR対象の正規化cropです。死亡表示周辺だけを読みます。 |
| `DEATH_EVENT_OCR_LANG` | `"jpn+eng"` | TesseractのOCR言語です。日本語データが必要です。 |
| `DEATH_EVENT_OCR_CONFIG` | `"--psm 6"` | Tesseractのページ分割設定です。 |
| `DEATH_EVENT_OCR_MIN_CONFIDENCE` | `0.0` | OCR結果をログ採用する最低confidenceです。 |
| `DEATH_EVENT_OCR_SCALE` | `3.0` | OCR前にcropを拡大する倍率です。 |
| `DEATH_EVENT_OCR_PREPROCESS_MODE` | `"default"` | OCR前処理です。`"threshold"`, `"adaptive"`, `"invert_threshold"`, `"sharpen_threshold"` も試せます。 |
| `DEATH_EVENT_OCR_TESSERACT_CMD` | `""` | Tesseract実行ファイルのパスです。PATHが通っていない場合に指定します。 |
| `DEATH_EVENT_OCR_SAVE_DEBUG_IMAGES` | `False` | OCR用cropと前処理後画像を保存します。通常配信では `False` にします。 |
| `DEATH_EVENT_OCR_DEBUG_DIR` | `"debug/ocr"` | OCR debug画像の保存先です。 |
| `DEATH_EVENT_USE_CONTEXT_REACTIONS` | `True` | OCRに依存しない死亡状況別固定文を選びます。標準のPhase3動作です。 |
| `DEATH_EVENT_USE_WEAPON_CATEGORY_REACTIONS` | `False` | OCR結果から武器カテゴリ分岐を試す実験機能です。通常はOFFにします。 |
| `DEATH_EVENT_OCR_CATEGORY_MIN_CONFIDENCE` | `60.0` | 武器カテゴリ推定に使う最低OCR confidenceです。低い場合は `unknown` にします。 |
| `DEATH_EVENT_CATEGORY_DEBUG_LOG` | `False` | カテゴリ推定の詳細ログを出します。 |
| `DEATH_EVENT_WEAPON_KEYWORDS` | `None` | 武器カテゴリ分岐をONにした場合だけ使います。 |
| `DEATH_EVENT_REACTIONS_BY_CATEGORY` | `None` | `None` の場合は標準の死亡状況別固定文を使います。 |
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

### 死亡状況別固定文リアクション（Phase 3）

Phase 3では、死亡イベントにLLMを使わず、死亡状況別の固定文を選びます。低遅延・低負荷で、Twitchコメントや「しずく」呼びかけにLLMリソースを残すためです。

標準では `DEATH_EVENT_USE_CONTEXT_REACTIONS = True`、`DEATH_EVENT_USE_WEAPON_CATEGORY_REACTIONS = False` です。OCR成功/失敗に関係なく、発話内容にはOCR文字列を使いません。OCR結果はログ、CSV、将来検証用として残します。

OCRによる武器カテゴリ分岐は実験機能です。現時点ではOCR品質が不安定なため既定OFFです。試す場合だけ `DEATH_EVENT_USE_WEAPON_CATEGORY_REACTIONS = True` にします。`DEATH_EVENT_OCR_CATEGORY_MIN_CONFIDENCE` 未満、OCR失敗、キーワード不一致の場合は `unknown` に落とし、死亡状況別固定文へフォールバックします。武器名やプレイヤー名を断定して発話しません。

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

### OBS Virtual Camera入力（Phase 4）

OBS Virtual Cameraを使うと、ディスプレイ全画面表示に依存せず、OBSの映像を画面イベント検出の入力元にできます。死亡検出、OCR、固定文リアクション、Twitch優先、配信者発話固定文は従来通りです。

設定例:

```python
SCREEN_EVENT_ENABLED = True
SCREEN_FRAME_SOURCE = "obs_virtual_camera"
OBS_VIRTUAL_CAMERA_INDEX = 2
OBS_VIRTUAL_CAMERA_WIDTH = 1920
OBS_VIRTUAL_CAMERA_HEIGHT = 1080
OBS_VIRTUAL_CAMERA_FPS = 30
```

OBS側では、配信画面を構成してから Virtual Camera を開始してください。OBSキャンバス解像度と `OBS_VIRTUAL_CAMERA_WIDTH` / `OBS_VIRTUAL_CAMERA_HEIGHT` を合わせると確認しやすくなります。実際の取得解像度は起動ログの `obs actual width=... height=... fps=...` を見て確認します。

1フレーム取得して保存:

```bash
python src\screen_event_detector.py --frame-source obs_virtual_camera --capture-once --save-frame debug\obs_frame.png
```

1フレーム取得して死亡検出に通す:

```bash
python src\screen_event_detector.py --frame-source obs_virtual_camera --capture-once --detect
```

検出しにくい場合は、ROI枠付き画像とスコアを保存して確認します。

```bash
python src\screen_event_detector.py --frame-source obs_virtual_camera --capture-once --detect --print-detection-metrics --save-debug-frames --debug-dir debug\obs_detect
```

保存される主なファイル:

- `*_full.png`: OBS Virtual Cameraから取得し、判定サイズへそろえたフレーム
- `*_overlay.png`: death ROIとtext ROIを重ねた確認用画像
- `*_death_roi.png`: 死亡表示全体のcrop
- `*_text_roi.png`: `やられた!` 付近のcrop
- `*_metrics.json`: ROI座標と検出スコア

カメラ番号を変えて確認:

```bash
python src\screen_event_detector.py --frame-source obs_virtual_camera --capture-once --camera-index 1 --save-frame debug\obs_frame_1.png
```

OBS Virtual Cameraを選んで開けない場合、MSSへ自動fallbackしません。意図しない画面を読んで混乱しないよう、画面イベント検出だけ停止し、Twitch/STT/LLM/TTSは継続します。

トラブルシュート:

- カメラが開けない: OBS Virtual Cameraを開始し、`OBS_VIRTUAL_CAMERA_INDEX` を変え、他アプリが占有していないか確認します。
- 黒画面になる: OBSのシーン、ソース表示、Virtual Camera開始状態を確認します。
- 別カメラを読んでいる: `--camera-index` を変え、`--save-frame` で保存画像を確認します。
- 解像度が想定と違う: 起動ログのactual width/heightを確認し、OBSキャンバスやVirtual Camera設定を見直します。
- 検出しない: `*_overlay.png` で死亡表示がROI内に入っているか確認します。ズレている場合は `DEATH_EVENT_ROI_OBS` と `DEATH_EVENT_TEXT_ROI_OBS` を調整します。
- ROIは合っているのに `template_score` が低い: OBS経由の映像から `やられた!` 部分を切り出し、`assets/templates/splatoon_death_yarareta_obs.png` を作成して `DEATH_EVENT_TEMPLATE_PATH_OBS` に指定します。
- `dark_ratio` や `white_ratio` がMSS時と大きく違う: OBSキャンバス内に黒帯、余白、縮小されたゲーム映像が入っていないか確認します。
- しきい値調整は最後に行います。単純に緩めるとリザルト、ロビー、復活画面の誤検知が増えるため、まずROIとOBS用テンプレートを合わせます。

### テスト配信中の誤検知確認

テスト配信中に死亡判定が誤検知した場合、raw検出された瞬間のフレームを保存できます。発話されたイベントだけでなく、screen event側のcooldownで発話されなかったraw検出も保存対象です。

```python
SCREEN_EVENT_SAVE_DETECTED_FRAMES = True
SCREEN_EVENT_DETECTED_FRAME_DIR = "debug/screen_events"
SCREEN_EVENT_MAX_DEBUG_FILES = 200
```

通常配信ではdebugファイルが増えるため `False` 推奨です。保存される主なファイルは `*_full.png`, `*_overlay.png`, `*_death_roi.png`, `*_text_roi.png`, `*_metrics.json` です。`*_overlay.png` では緑枠がdeath ROI、赤枠がtext ROIです。

`metrics.json` では以下を確認します。

- `template_score`
- `dark_ratio`
- `white_ratio`
- `confidence`
- `detection_reason`
- `screen_event_cooldown_active`
- `screen_event_cooldown_remaining`
- `emitted`

誤検知時の確認順:

1. `*_full.png` で入力元が正しいか確認します。
2. `*_overlay.png` でROI位置を確認します。
3. ROIがズレていれば `DEATH_EVENT_ROI_OBS` / `DEATH_EVENT_TEXT_ROI_OBS` を調整します。
4. ROIが合っていて `template_score` が低ければOBS専用テンプレートを作成します。
5. 最後にしきい値調整を検討します。

将来の候補:

- `OBSWindowOrProjectorFrameSource`: OBSプロジェクターを `mss` で読む代替。
- `OBSSourceFrameSource`: OBS WebSocketやOBS拡張による直接取得。
- `SCREEN_FRAME_SOURCE_FALLBACK`: 明示設定した場合だけMSS fallbackを許可。

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
