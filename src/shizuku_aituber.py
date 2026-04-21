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


# =========================================
# config/config.py を読み込む
# =========================================
try:
    from config.config import (
        SAMPLE_RATE, CHANNELS, INPUT_DEVICE, OUTPUT_DEVICE,
        VAD_START_RMS, VAD_END_RMS, MAX_RECORD_SECONDS, MIN_RECORD_SECONDS, END_SILENCE_SECONDS,
        WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
        OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_TIMEOUT_SEC,
        TTS_BASE_URL, TTS_SPEAKER, TTS_TIMEOUT_SEC,
        SYSTEM_PROMPT,
        MAX_RESPONSE_CHARS, ADD_SHORTENER_PROMPT,
        OUTPUT_DEVICE_NAME
    )
except Exception as e:
    raise RuntimeError(
        "config/config.py を読み込めませんでした。"
        "プロジェクト直下で `python src/shizuku_aituber.py` を実行しているか確認してください。"
    ) from e


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

def main():
    print("=== 月野しずく AITuber ===")
    print("終了: Ctrl+C\n")

    stt = STT(CFG)
    llm = LLM(CFG)
    tts = TTS(CFG)
    player = Player(CFG)

    try:
        print("[WARMUP] LLM...", flush=True)
        _ = llm.chat("短く挨拶して。")
        print("[WARMUP] TTS...", flush=True)
        _ = tts.synthesize_wav_bytes("起動しました。")
        print("[WARMUP] done", flush=True)
    except Exception as e:
        print(f"[WARMUP] skipped: {e}", flush=True)

    while True:
        try:
            audio = record_utterance(CFG)
            if audio is None:
                continue

            t0 = t()
            user_text = stt.transcribe(audio)
            t1 = t()
            log_time("STT", t1 - t0)

            if not user_text:
                print("認識結果: （空）", flush=True)
                continue

            print(f"あなた: {user_text}", flush=True)

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

            print("-" * 40, flush=True)

        except KeyboardInterrupt:
            print("\n終了します。")
            break
        except Exception as e:
            print(f"\nエラー: {e}\n", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main()