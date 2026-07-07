# --- Audio I/O ---
SAMPLE_RATE = 16000
CHANNELS = 1
INPUT_DEVICE = 1       # 例: 3 のように番号指定も可
OUTPUT_DEVICE = None
OUTPUT_DEVICE_NAME = "Voicemeeter AUX Input"

# --- VAD / 録音制御 ---
VAD_START_RMS = 0.012
VAD_END_RMS = 0.008
MAX_RECORD_SECONDS = 12.0
MIN_RECORD_SECONDS = 0.6
END_SILENCE_SECONDS = 0.7

# --- STT (faster-whisper) ---
WHISPER_MODEL_SIZE = "medium"   # small/medium
WHISPER_DEVICE = "cuda"        # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float16"

# --- LLM (Ollama) ---
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "phi4-mini"
LLM_TIMEOUT_SEC = 60

# --- TTS (AivisSpeech Engine / VOICEVOX互換) ---
TTS_BASE_URL = "http://127.0.0.1:10101"
TTS_SPEAKER = 1717361472             # Noneなら /speakers から自動選択
TTS_TIMEOUT_SEC = 60

# --- Character / System prompt ---
SYSTEM_PROMPT = """あなたは「月野しずく」というキャラクターです。
ゲーム配信者の隣で静かに配信を見守る相方です。

# キャラクター
- 落ち着いている
- 静かでやさしい
- 少しだけフレンドリー
- 控えめでふんわりした雰囲気
- 配信者を立てる
- 主役にはならない

# 立ち位置
- 配信者の隣で配信を見ている観察者
- 実況者ではない
- 解説者でもない
- 配信の空気をよくする存在

# 会話ルール
- 日本語で話す
- 返答は必ず短くする
- 1回の返答は1文のみ
- 最大40文字程度
- 句点で終わる
- 配信のテンポを邪魔しない

# 会話の方針
- 配信者の直前の発言に反応する
- 勝手に新しい話題を出さない
- 配信者が言っていないことを補わない
- 解説を始めない
- 指示や説教をしない
- 質問で返さない
- 長い説明をしない
- 配信の空気を優しく整える

# 話し方
- 落ち着いた優しい語り方
- フレンドリーだが控えめ
- テンションは高すぎない
- AIらしい説明口調を避ける
- 自然な短い感想を中心にする

# よく使う表現
- いいですね。
- 楽しそうです。
- いい流れですね。
- 惜しいですね。
- 落ち着いていていいですね。
- 今日はそんな感じなんですね。
- それ、好きです。

# 避ける表現
- では
- たとえば
- おすすめとしては
- 最新のニュース
- 科学
- 料理
- レベルデザイン
- キャラクター開発
- 長い解説
- 質問返し
- 固有名詞の言い換え

# 定番フレーズ

配信開始のとき
- こんばんは。ゆっくり見ていきますね。
- 始まりましたね。今日はどんな感じでしょう。
- 今日も始まりましたね。楽しみです。

配信終了のとき
- 今日もいい時間でしたね。おつかれさまでした。
- ゆっくり休んでくださいね。また次も楽しみです。
- 今日はここまでですね。いい配信でした。

視聴者が挨拶したとき
- こんばんは。来てくれてありがとうございます。
- いらっしゃいませ。ゆっくりしていってください。
- 来てくれて嬉しいです。こんばんは。

フォローされたとき
- フォローありがとうございます。嬉しいです。
- フォローしてくれたんですね。ありがとうございます。
- 応援してくれて嬉しいです。ありがとうございます。

# 安全
- 危険な内容
- 攻撃的な内容
- 政治や宗教
- 個人情報
には触れない
"""

# --- Response shaping ---
MAX_RESPONSE_CHARS = 120
ADD_SHORTENER_PROMPT = True

# --- Silent Reaction ---
SILENT_REACTION_ENABLED = True
SILENT_REACTION_INTERVAL_SEC = 180.0
SILENT_REACTION_PHRASES = (
    "見ていますよ。",
    "静かですね。",
    "大丈夫です、待機しています。",
    "ふふっ。",
    "ゆっくりしていってくださいね。"
)

# --- Conversation Control ---
AI_SPEECH_COOLDOWN_SEC = 8.0
STREAMER_RESPONSE_PROBABILITY = 0.25
SHIZUKU_CALL_KEYWORDS = ("しずく", "シズク", "雫")
STREAMER_FORCE_REPLY_KEYWORDS = ("しずく", "どう思う", "見て", "聞いて")
STREAMER_FIXED_RESPONSE_ENABLED = True
STREAMER_LLM_ON_ADDRESS_ONLY = True
STREAMER_FIXED_RESPONSE_RATE = 0.5
STREAMER_SHORT_NOISE_SKIP = True
STREAMER_FIXED_RESPONSE_COOLDOWN_SEC = 15.0
STREAMER_UTTERANCE_KEYWORDS = None  # None の場合は標準分類キーワードを使います。
STREAMER_FIXED_RESPONSE_PHRASES = None  # None の場合は標準固定文を使います。

# --- Game Mode ---
GAME_MODE = "normal"  # "normal" or "battle"

# --- Safety / Tone Control ---
APOLOGY_SUPPRESSION_ENABLED = True
APOLOGY_REPLACEMENT_PHRASES = (
    "落ち着いていきましょう。",
    "まだいけます。",
    "惜しいですね。",
    "切り替えていきましょう。",
)

# --- Twitch ---
TWITCH_COMMENT_ENABLED = False
TWITCH_COMMENT_PRIORITY = True
TWITCH_COMMENT_COOLDOWN_SEC = 3.0
TWITCH_CHANNEL_NAME = ""
TWITCH_BOT_USERNAME = ""  # 認証ユーザー名（NICK）
TWITCH_BROADCASTER_USER_ID = ""
TWITCH_USER_ID = ""
TWITCH_CLIENT_ID = ""
TWITCH_ACCESS_TOKEN = ""
TWITCH_DEBUG_LOG = False

COMMENT_MAX_LENGTH = 50
COMMENT_IGNORE_PREFIXES = ("!", "/")
COMMENT_IGNORE_URL = True

# --- Screen Event Detection ---
SCREEN_EVENT_ENABLED = False
SCREEN_EVENT_MODE = "death_detect_only"  # "death_detect_only" or "death_ocr"
SCREEN_FRAME_SOURCE = "mss"  # "mss" or "obs_virtual_camera"
SCREEN_CAPTURE_INTERVAL_SEC = 0.25
SCREEN_CAPTURE_MONITOR_INDEX = 1
SCREEN_CAPTURE_WIDTH = 640
SCREEN_CAPTURE_HEIGHT = 360
SCREEN_CAPTURE_REGION = None  # 例: (0, 50, 1920, 1080)
SCREEN_EVENT_DEBUG_LOG = False
SCREEN_EVENT_DEBUG_DIR = "debug/screen_event"
SCREEN_EVENT_SAVE_DEBUG_FRAMES = False
SCREEN_EVENT_SAVE_ROI_CROPS = True
SCREEN_EVENT_SAVE_DETECTED_FRAMES = False
SCREEN_EVENT_DETECTED_FRAME_DIR = "debug/screen_events"
SCREEN_EVENT_SAVE_DETECTED_FULL_FRAME = True
SCREEN_EVENT_SAVE_DETECTED_OVERLAY = True
SCREEN_EVENT_SAVE_DETECTED_ROI = True
SCREEN_EVENT_SAVE_DETECTED_METRICS = True
SCREEN_EVENT_MAX_DEBUG_FILES = 200
SCREEN_EVENT_LOG_EVERY_SEC = 10.0
OBS_VIRTUAL_CAMERA_INDEX = 2
OBS_VIRTUAL_CAMERA_WIDTH = 1920
OBS_VIRTUAL_CAMERA_HEIGHT = 1080
OBS_VIRTUAL_CAMERA_FPS = 30
OBS_VIRTUAL_CAMERA_WARMUP_FRAMES = 5

DEATH_EVENT_COOLDOWN_SEC = 20.0
DEATH_EVENT_USE_LLM = False
DEATH_EVENT_MIN_CONFIDENCE = 0.72
DEATH_EVENT_MIN_TEMPLATE_SCORE = 0.55
DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE = 0.40
DEATH_EVENT_WHITE_RATIO_MIN = 0.015
DEATH_EVENT_WHITE_RATIO_MAX = 0.18
DEATH_EVENT_SHAPE_WHITE_RATIO_MIN = 0.04
DEATH_EVENT_SHAPE_WHITE_RATIO_MAX = 0.14
DEATH_EVENT_ROI = (0.32, 0.21, 0.67, 0.54)
DEATH_EVENT_TEXT_ROI = (0.42, 0.33, 0.59, 0.46)
DEATH_EVENT_TEMPLATE_PATH = "assets/templates/splatoon_death_yarareta.png"
DEATH_EVENT_ROI_OBS = None  # OBS入力でROIがズレる場合だけ例: (0.32, 0.21, 0.67, 0.54)
DEATH_EVENT_TEXT_ROI_OBS = None
DEATH_EVENT_TEMPLATE_PATH_OBS = ""  # 例: "assets/templates/splatoon_death_yarareta_obs.png"
DEATH_EVENT_MIN_TEMPLATE_SCORE_OBS = None
DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE_OBS = None
DEATH_EVENT_OCR_ROI = (0.34, 0.25, 0.66, 0.48)
DEATH_EVENT_OCR_LANG = "jpn+eng"
DEATH_EVENT_OCR_CONFIG = "--psm 6"
DEATH_EVENT_OCR_MIN_CONFIDENCE = 0.0
DEATH_EVENT_OCR_SCALE = 3.0
DEATH_EVENT_OCR_PREPROCESS_MODE = "default"  # "default", "threshold", "adaptive", "invert_threshold", "sharpen_threshold"
DEATH_EVENT_OCR_TESSERACT_CMD = ""  # 例: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEATH_EVENT_OCR_DEBUG_LOG = False
DEATH_EVENT_OCR_SAVE_DEBUG_IMAGES = False
DEATH_EVENT_OCR_DEBUG_DIR = "debug/ocr"
DEATH_EVENT_USE_CONTEXT_REACTIONS = True
DEATH_EVENT_USE_WEAPON_CATEGORY_REACTIONS = False
DEATH_EVENT_CATEGORY_DEBUG_LOG = False
DEATH_EVENT_OCR_CATEGORY_MIN_CONFIDENCE = 60.0
DEATH_EVENT_WEAPON_KEYWORDS = None  # 武器カテゴリ分岐をONにした場合だけ使います。
DEATH_EVENT_REACTIONS_BY_CATEGORY = None  # None の場合は標準の死亡状況別固定文を使います。
DEATH_EVENT_UNKNOWN_USE_LLM = False  # Phase3では未使用。将来拡張用です。
DEATH_EVENT_UNKNOWN_LLM_RATE = 0.2
DEATH_EVENT_UNKNOWN_LLM_COOLDOWN_SEC = 60.0
DEATH_EVENT_REACTION_PHRASES = (
    "今のはきついですね。",
    "これは悔しいですね。",
    "相手、やってますね。",
    "今の詰め方は強いですね。",
    "それは声出ますね。",
)
