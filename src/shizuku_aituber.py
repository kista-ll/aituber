import sys
from pathlib import Path

# src/ の1つ上（プロジェクト直下）を import パスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import time
import queue
import threading
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

try:
    from twitch_comment_reader import TwitchCommentReader
except BaseException as e:
    TwitchCommentReader = None
    TWITCH_IMPORT_ERROR = e
else:
    TWITCH_IMPORT_ERROR = None

try:
    from screen_event_detector import ScreenEventDetector
except BaseException as e:
    ScreenEventDetector = None
    SCREEN_IMPORT_ERROR = e
else:
    SCREEN_IMPORT_ERROR = None


# =========================================
# config/config.py を読み込む
# =========================================
try:
    from config import config as CONFIG_MODULE
    from config.config import (
        SAMPLE_RATE, CHANNELS, INPUT_DEVICE, OUTPUT_DEVICE,
        VAD_START_RMS, VAD_END_RMS, MAX_RECORD_SECONDS, MIN_RECORD_SECONDS, END_SILENCE_SECONDS,
        WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
        OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TIMEOUT_SEC,
        TTS_BASE_URL, TTS_SPEAKER, TTS_TIMEOUT_SEC,
        SYSTEM_PROMPT,
        MAX_RESPONSE_CHARS, ADD_SHORTENER_PROMPT,
        OUTPUT_DEVICE_NAME,
        SILENT_REACTION_ENABLED, SILENT_REACTION_INTERVAL_SEC, SILENT_REACTION_PHRASES
    )
except Exception as e:
    raise RuntimeError(
        "config/config.py を読み込めませんでした。"
        "プロジェクト直下で `python src/shizuku_aituber.py` を実行しているか確認してください。"
    ) from e


def config_value(name, default):
    value = getattr(CONFIG_MODULE, name, default)
    if not hasattr(CONFIG_MODULE, name):
        setattr(CONFIG_MODULE, name, value)
    return value


AI_SPEECH_COOLDOWN_SEC = config_value("AI_SPEECH_COOLDOWN_SEC", 8.0)
STREAMER_RESPONSE_PROBABILITY = config_value("STREAMER_RESPONSE_PROBABILITY", 0.25)
SHIZUKU_CALL_KEYWORDS = config_value("SHIZUKU_CALL_KEYWORDS", ("しずく", "シズク", "雫"))
STREAMER_FORCE_REPLY_KEYWORDS = config_value("STREAMER_FORCE_REPLY_KEYWORDS", ("しずく", "どう思う", "見て", "聞いて"))

TWITCH_COMMENT_ENABLED = config_value("TWITCH_COMMENT_ENABLED", False)
TWITCH_COMMENT_PRIORITY = config_value("TWITCH_COMMENT_PRIORITY", True)
TWITCH_COMMENT_COOLDOWN_SEC = config_value("TWITCH_COMMENT_COOLDOWN_SEC", 3.0)

GAME_MODE = config_value("GAME_MODE", "normal")
APOLOGY_SUPPRESSION_ENABLED = config_value("APOLOGY_SUPPRESSION_ENABLED", True)
APOLOGY_REPLACEMENT_PHRASES = config_value(
    "APOLOGY_REPLACEMENT_PHRASES",
    ("落ち着いていきましょう。", "まだいけます。", "惜しいですね。", "切り替えていきましょう。"),
)

SCREEN_EVENT_ENABLED = config_value("SCREEN_EVENT_ENABLED", False)
SCREEN_EVENT_MODE = config_value("SCREEN_EVENT_MODE", "death_detect_only")
SCREEN_CAPTURE_INTERVAL_SEC = config_value("SCREEN_CAPTURE_INTERVAL_SEC", 0.25)
SCREEN_CAPTURE_MONITOR_INDEX = config_value("SCREEN_CAPTURE_MONITOR_INDEX", 1)
SCREEN_CAPTURE_WIDTH = config_value("SCREEN_CAPTURE_WIDTH", 640)
SCREEN_CAPTURE_HEIGHT = config_value("SCREEN_CAPTURE_HEIGHT", 360)
SCREEN_CAPTURE_REGION = config_value("SCREEN_CAPTURE_REGION", None)
SCREEN_EVENT_DEBUG_LOG = config_value("SCREEN_EVENT_DEBUG_LOG", False)
SCREEN_EVENT_LOG_EVERY_SEC = config_value("SCREEN_EVENT_LOG_EVERY_SEC", 10.0)

DEATH_EVENT_COOLDOWN_SEC = config_value("DEATH_EVENT_COOLDOWN_SEC", 20.0)
DEATH_EVENT_USE_LLM = config_value("DEATH_EVENT_USE_LLM", False)
DEATH_EVENT_MIN_CONFIDENCE = config_value("DEATH_EVENT_MIN_CONFIDENCE", 0.72)
DEATH_EVENT_MIN_TEMPLATE_SCORE = config_value("DEATH_EVENT_MIN_TEMPLATE_SCORE", 0.55)
DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE = config_value("DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE", 0.40)
DEATH_EVENT_WHITE_RATIO_MIN = config_value("DEATH_EVENT_WHITE_RATIO_MIN", 0.015)
DEATH_EVENT_WHITE_RATIO_MAX = config_value("DEATH_EVENT_WHITE_RATIO_MAX", 0.18)
DEATH_EVENT_ROI = config_value("DEATH_EVENT_ROI", (0.32, 0.21, 0.67, 0.54))
DEATH_EVENT_TEXT_ROI = config_value("DEATH_EVENT_TEXT_ROI", (0.42, 0.33, 0.59, 0.46))
DEATH_EVENT_TEMPLATE_PATH = config_value("DEATH_EVENT_TEMPLATE_PATH", "assets/templates/splatoon_death_yarareta.png")
DEATH_EVENT_REACTION_PHRASES = config_value(
    "DEATH_EVENT_REACTION_PHRASES",
    ("今のはきついですね。", "これは悔しいですね。", "相手、やってますね。", "今の詰め方は強いですね。", "それは声出ますね。"),
)


# =========================
# 設定（dataclassに集約）
# =========================

@dataclass
class Config:
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    input_device: Optional[int] = INPUT_DEVICE
    output_device: Optional[int] = OUTPUT_DEVICE
    output_device_name: Optional[str] = OUTPUT_DEVICE_NAME

    vad_start_rms: float = VAD_START_RMS
    vad_end_rms: float = VAD_END_RMS
    max_record_seconds: float = MAX_RECORD_SECONDS
    min_record_seconds: float = MIN_RECORD_SECONDS
    end_silence_seconds: float = END_SILENCE_SECONDS

    whisper_model_size: str = WHISPER_MODEL_SIZE
    whisper_device: str = WHISPER_DEVICE
    whisper_compute_type: str = WHISPER_COMPUTE_TYPE

    ollama_base_url: str = OLLAMA_BASE_URL
    ollama_model: str = OLLAMA_MODEL
    llm_timeout_sec: int = LLM_TIMEOUT_SEC

    tts_base_url: str = TTS_BASE_URL
    tts_speaker: Optional[int] = TTS_SPEAKER
    tts_timeout_sec: int = TTS_TIMEOUT_SEC

    system_prompt: str = SYSTEM_PROMPT
    max_response_chars: int = MAX_RESPONSE_CHARS
    add_shortener_prompt: bool = ADD_SHORTENER_PROMPT

    silent_reaction_enabled: bool = SILENT_REACTION_ENABLED
    silent_reaction_interval_sec: float = SILENT_REACTION_INTERVAL_SEC
    silent_reaction_phrases: tuple = SILENT_REACTION_PHRASES

    ai_speech_cooldown_sec: float = AI_SPEECH_COOLDOWN_SEC
    streamer_response_probability: float = STREAMER_RESPONSE_PROBABILITY
    shizuku_call_keywords: tuple = SHIZUKU_CALL_KEYWORDS
    streamer_force_reply_keywords: tuple = STREAMER_FORCE_REPLY_KEYWORDS

    twitch_comment_enabled: bool = TWITCH_COMMENT_ENABLED
    twitch_comment_priority: bool = TWITCH_COMMENT_PRIORITY
    twitch_comment_cooldown_sec: float = TWITCH_COMMENT_COOLDOWN_SEC

    game_mode: str = GAME_MODE
    apology_suppression_enabled: bool = APOLOGY_SUPPRESSION_ENABLED
    apology_replacement_phrases: tuple = APOLOGY_REPLACEMENT_PHRASES

    screen_event_enabled: bool = SCREEN_EVENT_ENABLED
    screen_event_mode: str = SCREEN_EVENT_MODE
    screen_capture_interval_sec: float = SCREEN_CAPTURE_INTERVAL_SEC
    screen_capture_monitor_index: int = SCREEN_CAPTURE_MONITOR_INDEX
    screen_capture_width: int = SCREEN_CAPTURE_WIDTH
    screen_capture_height: int = SCREEN_CAPTURE_HEIGHT
    screen_capture_region: Optional[tuple] = SCREEN_CAPTURE_REGION
    screen_event_debug_log: bool = SCREEN_EVENT_DEBUG_LOG
    screen_event_log_every_sec: float = SCREEN_EVENT_LOG_EVERY_SEC

    death_event_cooldown_sec: float = DEATH_EVENT_COOLDOWN_SEC
    death_event_use_llm: bool = DEATH_EVENT_USE_LLM
    death_event_min_confidence: float = DEATH_EVENT_MIN_CONFIDENCE
    death_event_min_template_score: float = DEATH_EVENT_MIN_TEMPLATE_SCORE
    death_event_shape_min_template_score: float = DEATH_EVENT_SHAPE_MIN_TEMPLATE_SCORE
    death_event_white_ratio_min: float = DEATH_EVENT_WHITE_RATIO_MIN
    death_event_white_ratio_max: float = DEATH_EVENT_WHITE_RATIO_MAX
    death_event_roi: tuple = DEATH_EVENT_ROI
    death_event_text_roi: tuple = DEATH_EVENT_TEXT_ROI
    death_event_template_path: str = DEATH_EVENT_TEMPLATE_PATH
    death_event_reaction_phrases: tuple = DEATH_EVENT_REACTION_PHRASES


CFG = Config()


# =========================
# ユーティリティ
# =========================

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def clamp_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def t() -> float:
    return time.perf_counter()


def log_time(label: str, dt: float) -> None:
    print(f"[TIME] {label:<5}: {dt:.2f}s", flush=True)


def contains_keyword(text: str, keywords: tuple) -> bool:
    normalized = text.lower()
    return any(str(keyword).lower() in normalized for keyword in keywords if keyword)


def contains_call_keyword(text: str, cfg: Config) -> bool:
    return contains_keyword(text, cfg.shizuku_call_keywords)


def is_cooldown_ready(now: float, last_time: float, cooldown_sec: float) -> bool:
    return (now - last_time) >= max(0.0, cooldown_sec)


def should_reply_to_streamer(text: str, now: float, state, cfg: Config) -> bool:
    if contains_call_keyword(text, cfg) or contains_keyword(text, cfg.streamer_force_reply_keywords):
        return True

    if not is_cooldown_ready(now, state.last_assistant_speech_time, cfg.ai_speech_cooldown_sec):
        print("[SKIP] AI speech cooldown", flush=True)
        return False

    probability = min(1.0, max(0.0, cfg.streamer_response_probability))
    if random.random() > probability:
        print("[SKIP] streamer speech probability", flush=True)
        return False

    return True


def get_next_twitch_comment(comment_queue: Optional[queue.Queue]) -> Optional[dict]:
    if comment_queue is None:
        return None
    try:
        return comment_queue.get_nowait()
    except queue.Empty:
        return None


def has_queued_item(item_queue: Optional[queue.Queue]) -> bool:
    return item_queue is not None and not item_queue.empty()


def get_next_screen_event(event_queue: Optional[queue.Queue]):
    if event_queue is None:
        return None
    try:
        return event_queue.get_nowait()
    except queue.Empty:
        return None


def format_twitch_comment_for_llm(comment: dict) -> str:
    username = str(comment.get("username", "視聴者")).strip() or "視聴者"
    message = str(comment.get("message", "")).strip()
    return f"{username}さんのコメント: {message}"


def build_mode_prompt(cfg: Config) -> str:
    mode = str(cfg.game_mode).strip().lower()
    if mode == "battle":
        return (
            "GAME_MODE: battle\n"
            "対戦ゲーム中です。短い応援、共感、状況リアクションを中心にしてください。"
            "謝罪や説教を避け、強い言葉には落ち着いた一言で返してください。"
        )
    return (
        "GAME_MODE: normal\n"
        "通常配信です。落ち着いた相方として、控えめで短い反応をしてください。"
    )


def suppress_apology_reply(reply: str, cfg: Config) -> str:
    apology_phrases = ("すみません", "ごめんなさい", "申し訳ありません", "申し訳ない", "ごめん")
    if not contains_keyword(reply, apology_phrases):
        return reply

    stripped = reply.strip(" 、。")
    apology_only = any(stripped == phrase for phrase in apology_phrases)
    apology_heavy = len(stripped) <= 24
    if not apology_only and not apology_heavy:
        return reply

    replacements = tuple(p for p in cfg.apology_replacement_phrases if p)
    if not replacements:
        return "落ち着いていきましょう。"
    return random.choice(replacements)


def resolve_output_device(cfg: Config) -> int:
    # 番号が明示されていればそれを使う
    if cfg.output_device is not None:
        dev = sd.query_devices(cfg.output_device)
        if dev["max_output_channels"] > 0:
            return cfg.output_device
        raise RuntimeError(
            f"OUTPUT_DEVICE={cfg.output_device} は出力デバイスではありません: "
            f"{dev['name']} (max_output_channels={dev['max_output_channels']})"
        )

    # 名前から探す
    if not cfg.output_device_name:
        raise RuntimeError("OUTPUT_DEVICE も OUTPUT_DEVICE_NAME も設定されていません。")

    candidates = []
    for i, dev in enumerate(sd.query_devices()):
        name = dev["name"]
        if cfg.output_device_name.lower() in name.lower() and dev["max_output_channels"] > 0:
            candidates.append((i, name, dev["max_output_channels"], dev["default_samplerate"]))

    if not candidates:
        raise RuntimeError(
            f"'{cfg.output_device_name}' を含む出力デバイスが見つかりません。"
        )

    # WASAPI優先、次にDirectSound、最後にMME
    def score(item):
        idx, name, max_out, default_sr = item
        lname = name.lower()
        if "wasapi" in lname:
            return 0
        if "directsound" in lname:
            return 1
        if "mme" in lname:
            return 2
        return 3

    candidates.sort(key=score)

    idx, name, max_out, default_sr = candidates[0]
    print(f"[DEVICE] resolved output device: {idx} / {name} / out={max_out} / sr={default_sr}", flush=True)
    return idx


# =========================
# 録音（簡易VAD）
# =========================

def record_utterance(cfg: Config) -> Optional[np.ndarray]:
    block_size = int(cfg.sample_rate * 0.05)  # 50ms
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    print("マイク待機中…（話しかけてください）", flush=True)
    started = False
    start_time = time.time()
    last_voice_time = None
    chunks: List[np.ndarray] = []

    with sd.InputStream(
        samplerate=cfg.sample_rate,
        channels=cfg.channels,
        dtype="float32",
        blocksize=block_size,
        device=cfg.input_device,
        callback=callback,
    ):
        while True:
            if time.time() - start_time > cfg.max_record_seconds and not started:
                print("タイムアウト（無音）", flush=True)
                return None

            indata = q.get()
            mono = indata[:, 0] if indata.ndim == 2 else indata
            level = rms(mono)

            if not started:
                if level >= cfg.vad_start_rms:
                    started = True
                    last_voice_time = time.time()
                    chunks.append(mono.copy())
                    print("録音開始", flush=True)
            else:
                chunks.append(mono.copy())
                now = time.time()
                if level >= cfg.vad_end_rms:
                    last_voice_time = now
                if last_voice_time is not None and (now - last_voice_time) >= cfg.end_silence_seconds:
                    duration = len(np.concatenate(chunks)) / cfg.sample_rate
                    if duration < cfg.min_record_seconds:
                        print("短すぎるので破棄", flush=True)
                        return None
                    print(f"録音終了（{duration:.2f}s）", flush=True)
                    return np.concatenate(chunks)


# =========================
# STT（faster-whisper）
# =========================

class STT:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        print(f"[INIT] WhisperModel size={cfg.whisper_model_size} device={cfg.whisper_device} compute={cfg.whisper_compute_type}", flush=True)
        self.model = WhisperModel(
            cfg.whisper_model_size,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language="ja",
            vad_filter=False,
            beam_size=2,
        )
        return "".join(seg.text for seg in segments).strip()


# =========================
# LLM（Ollama HTTP API）
# =========================

import re

class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.history = []
        self.topic = "雑談"
        self.current_game = None

    def update_topic(self, user_text: str):
        text = user_text.strip()

        game_keywords = [
            "Splatoon 3", "Splatoon3", "スプラトゥーン3", "スプラ3", "スプラ",
            "ポケモン", "ポテモン", "マリオ", "モンハン", "APEX", "原神", "ゼルダ"
        ]

        for kw in game_keywords:
            if kw.lower() in text.lower():
                self.current_game = kw
                self.topic = kw
                return

        if self.current_game:
            self.topic = self.current_game

    def sanitize_reply(self, reply: str) -> str:
        reply = reply.replace("\n", " ").replace("「", "").replace("」", "").strip()

        if "。" in reply:
            reply = reply.split("。")[0] + "。"

        reply = reply.replace("？", "。").replace("?", "。")

        banned_phrases = [
            "最新のニュース", "科学", "料理", "別の興味深い", "レベルデザイン",
            "キャラクター開発", "新しいゲームを試して"
        ]
        for phrase in banned_phrases:
            reply = reply.replace(phrase, "")

        replacements = {
            "Splatoon3": "Splatoon 3",
            "スプラトゥン3": "Splatoon 3",
            "ポテモン": "ポケモン",
        }
        for src, dst in replacements.items():
            reply = reply.replace(src, dst)

        reply = reply.strip(" 、。")
        if not reply:
            reply = "いいですね。"

        if not reply.endswith("。"):
            reply += "。"

        if self.cfg.apology_suppression_enabled:
            reply = suppress_apology_reply(reply, self.cfg)

        reply = clamp_text(reply, 40)
        if not reply.endswith("。"):
            reply += "。"

        return reply

    def chat(self, user_text: str) -> str:
        self.update_topic(user_text)

        url = f"{self.cfg.ollama_base_url}/api/chat"
        extra = "\n\n必ず短く。1文のみ。質問で返さない。40文字以内。"

        messages = [
            {"role": "system", "content": self.cfg.system_prompt + extra},
            {"role": "system", "content": build_mode_prompt(self.cfg)},
            {"role": "system", "content": f"現在の話題: {self.topic}"},
            *self.history,
            {"role": "user", "content": user_text},
        ]

        payload = {
            "model": self.cfg.ollama_model,
            "stream": False,
            "options": {
                "num_predict": 20,
                "temperature": 0.3,
            },
            "messages": messages,
        }

        print(f"[LLM] request -> {url}  model={self.cfg.ollama_model} timeout={self.cfg.llm_timeout_sec}s", flush=True)
        r = requests.post(url, json=payload, timeout=self.cfg.llm_timeout_sec)
        print(f"[LLM] response status={r.status_code}", flush=True)
        r.raise_for_status()

        data = r.json()
        content = data.get("message", {}).get("content", "").strip()
        reply = self.sanitize_reply(content)

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-6:]

        return reply


# =========================
# TTS（AivisSpeech Engine / VOICEVOX互換）
# =========================

class TTS:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.speaker = cfg.tts_speaker if cfg.tts_speaker is not None else self._pick_default_speaker()

    def _pick_default_speaker(self) -> int:
        url = f"{self.cfg.tts_base_url}/speakers"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        speakers = r.json()
        for sp in speakers:
            styles = sp.get("styles") or []
            if styles:
                sid = styles[0].get("id")
                if isinstance(sid, int):
                    print(f"TTS speaker auto-selected: {sp.get('name')} / {styles[0].get('name')} (id={sid})", flush=True)
                    return sid
        raise RuntimeError("話者が見つかりません。AivisSpeech側でモデル/話者が有効か確認してください。")

    def synthesize_wav_bytes(self, text: str) -> bytes:
        q_url = f"{self.cfg.tts_base_url}/audio_query"
        r1 = requests.post(
            q_url,
            params={"text": text, "speaker": self.speaker},
            timeout=self.cfg.tts_timeout_sec,
        )
        r1.raise_for_status()
        audio_query = r1.json()

        s_url = f"{self.cfg.tts_base_url}/synthesis"
        r2 = requests.post(
            s_url,
            params={"speaker": self.speaker},
            json=audio_query,
            timeout=self.cfg.tts_timeout_sec,
        )
        r2.raise_for_status()
        return r2.content


# =========================
# 再生（Enterで割り込み停止）
# =========================

class Player:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.output_device_index = resolve_output_device(cfg)

    def _stdin_watcher(self):
        try:
            input()
            self.stop_event.set()
        except EOFError:
            pass

    def play_wav_bytes_interruptible(self, wav_bytes: bytes):
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        elif data.ndim == 2 and data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.ndim == 2 and data.shape[1] > 2:
            data = data[:, :2]

        self.stop_event.clear()
        self.finished_event.clear()

        watcher = threading.Thread(target=self._stdin_watcher, daemon=True)
        watcher.start()

        idx = 0

        def callback(outdata, frames, time_info, status):
            nonlocal idx

            if self.stop_event.is_set():
                self.finished_event.set()
                raise sd.CallbackStop()

            end = idx + frames
            chunk = data[idx:end]

            if len(chunk) < frames:
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = 0
                self.finished_event.set()
                raise sd.CallbackStop()

            outdata[:] = chunk
            idx = end

        print("再生中…（割り込み: Enter）", flush=True)

        try:
            with sd.OutputStream(
                samplerate=sr,
                channels=2,
                dtype="float32",
                device=self.output_device_index,
                callback=callback,
            ):
                while not self.finished_event.is_set():
                    time.sleep(0.05)
        finally:
            print("再生終了", flush=True)


# =========================
# メイン
# =========================

import random

@dataclass
class RuntimeState:
    last_user_interaction_time: float
    last_assistant_speech_time: float
    last_twitch_comment_time: float = 0.0
    last_screen_event_time: float = 0.0
    last_silent_phrase: Optional[str] = None

def main():
    print("=== 月野しずく AITuber ===")
    print("終了: Ctrl+C\n")

    stt = STT(CFG)
    llm = LLM(CFG)
    tts = TTS(CFG)
    player = Player(CFG)

    comment_queue = None
    twitch_reader = None
    if CFG.twitch_comment_enabled:
        if TwitchCommentReader is None:
            print(f"[Twitch] import failed. Twitch disabled: {TWITCH_IMPORT_ERROR}", flush=True)
        else:
            comment_queue = queue.Queue()
            twitch_reader = TwitchCommentReader(CONFIG_MODULE, comment_queue)
            try:
                twitch_reader.start()
            except Exception as e:
                print(f"[Twitch] start failed. Twitch disabled: {e}", flush=True)
                twitch_reader = None
                comment_queue = None

    screen_event_queue = None
    screen_detector = None
    if CFG.screen_event_enabled:
        if ScreenEventDetector is None:
            print(f"[SCREEN] disabled reason=import_failed error={SCREEN_IMPORT_ERROR}", flush=True)
        else:
            screen_event_queue = queue.Queue()
            screen_detector = ScreenEventDetector(CFG, screen_event_queue)
            screen_detector.start()
    else:
        print("[SCREEN] disabled", flush=True)

    try:
        print("[WARMUP] LLM...", flush=True)
        _ = llm.chat("短く挨拶して。")
        print("[WARMUP] TTS...", flush=True)
        _ = tts.synthesize_wav_bytes("起動しました。")
        print("[WARMUP] done", flush=True)
    except Exception as e:
        print(f"[WARMUP] skipped: {e}", flush=True)

    state = RuntimeState(
        last_user_interaction_time=time.monotonic(),
        last_assistant_speech_time=time.monotonic()
    )

    def speak_with_llm(user_text: str, label: str) -> None:
        t2 = t()
        reply = llm.chat(user_text)
        t3 = t()
        log_time("LLM", t3 - t2)

        print(f"しずく: {reply}", flush=True)

        t4 = t()
        wav = tts.synthesize_wav_bytes(reply)
        t5 = t()
        log_time("TTS", t5 - t4)

        t6 = t()
        player.play_wav_bytes_interruptible(wav)
        t7 = t()
        log_time("PLAY", t7 - t6)

        state.last_assistant_speech_time = time.monotonic()
        if label == "twitch":
            state.last_twitch_comment_time = state.last_assistant_speech_time

    def try_handle_twitch_comment() -> bool:
        if not CFG.twitch_comment_priority:
            return False

        comment = get_next_twitch_comment(comment_queue)
        if comment is None:
            return False

        now = time.monotonic()
        user_text = format_twitch_comment_for_llm(comment)
        if not is_cooldown_ready(now, state.last_assistant_speech_time, CFG.ai_speech_cooldown_sec):
            print("[SKIP] Twitch comment: AI speech cooldown", flush=True)
            return False
        if not is_cooldown_ready(now, state.last_twitch_comment_time, CFG.twitch_comment_cooldown_sec):
            print("[SKIP] Twitch comment: comment cooldown", flush=True)
            return False

        print(f"視聴者: {user_text}", flush=True)
        state.last_user_interaction_time = now
        speak_with_llm(user_text, "twitch")
        print("-" * 40, flush=True)
        return True

    def speak_fixed_phrase(phrase: str, label: str) -> None:
        print(f"しずく ({label}): {phrase}", flush=True)

        t_tts_s = t()
        wav = tts.synthesize_wav_bytes(phrase)
        log_time(f"TTS({label})", t() - t_tts_s)

        t_play_s = t()
        player.play_wav_bytes_interruptible(wav)
        log_time(f"PLAY({label})", t() - t_play_s)

        state.last_assistant_speech_time = time.monotonic()

    def try_handle_screen_event() -> bool:
        event = get_next_screen_event(screen_event_queue)
        if event is None:
            return False

        if has_queued_item(comment_queue):
            print("[SKIP] screen_event priority lower than twitch_comment", flush=True)
            return False

        now = time.monotonic()
        if not is_cooldown_ready(now, state.last_screen_event_time, CFG.death_event_cooldown_sec):
            print("[SCREEN] skip reason=cooldown", flush=True)
            return False

        phrases = tuple(p for p in CFG.death_event_reaction_phrases if p)
        if not phrases:
            print("[SCREEN] skip reason=no_reaction_phrases", flush=True)
            return False

        event_type = getattr(event, "event_type", "unknown")
        confidence = getattr(event, "confidence", 0.0)
        print(f"[EVENT] source=screen_event type={event_type} confidence={confidence:.2f}", flush=True)
        phrase = random.choice(phrases)
        speak_fixed_phrase(phrase, "screen_event")
        state.last_screen_event_time = time.monotonic()
        state.last_user_interaction_time = state.last_screen_event_time
        print("-" * 40, flush=True)
        return True

    try:
        while True:
            try:
                if try_handle_twitch_comment():
                    continue

                if try_handle_screen_event():
                    continue

                audio = record_utterance(CFG)
                now = time.monotonic()

                if try_handle_twitch_comment():
                    continue

                if audio is None:
                    if try_handle_screen_event():
                        continue

                    if CFG.silent_reaction_enabled:
                        if (now - state.last_user_interaction_time >= CFG.silent_reaction_interval_sec and 
                            now - state.last_assistant_speech_time >= CFG.silent_reaction_interval_sec):
                            
                            phrases = list(CFG.silent_reaction_phrases)
                            if not phrases:
                                continue
                                
                            candidates = [p for p in phrases if p != state.last_silent_phrase]
                            if not candidates:
                                candidates = phrases
                            
                            phrase = random.choice(candidates)
                            print(f"しずく (無音リアクション): {phrase}", flush=True)
                            
                            t_tts_s = t()
                            wav = tts.synthesize_wav_bytes(phrase)
                            log_time("TTS(Silent)", t() - t_tts_s)
                            
                            t_play_s = t()
                            player.play_wav_bytes_interruptible(wav)
                            log_time("PLAY(Silent)", t() - t_play_s)
                            
                            state.last_silent_phrase = phrase
                            state.last_assistant_speech_time = time.monotonic()
                    continue

                # 録音として有効なユーザー発話を受け取った時点で更新する
                state.last_user_interaction_time = time.monotonic()

                t0 = t()
                user_text = stt.transcribe(audio)
                t1 = t()
                log_time("STT", t1 - t0)

                if not user_text:
                    print("認識結果: （空）", flush=True)
                    continue

                print(f"あなた: {user_text}", flush=True)

                now = time.monotonic()
                if not should_reply_to_streamer(user_text, now, state, CFG):
                    continue

                speak_with_llm(user_text, "streamer")
                print("-" * 40, flush=True)

            except KeyboardInterrupt:
                print("\n終了します。")
                break
            except Exception as e:
                print(f"\nエラー: {e}\n", flush=True)
                time.sleep(0.5)
    finally:
        if screen_detector is not None:
            screen_detector.stop()
        if twitch_reader is not None:
            twitch_reader.stop()


if __name__ == "__main__":
    main()
