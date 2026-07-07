import random
import unicodedata
from typing import Any, Dict, Iterable, Tuple


DEFAULT_STREAMER_UTTERANCE_KEYWORDS = {
    "short_noise": (
        "あー",
        "うーん",
        "うーむ",
        "えー",
        "おっと",
        "はい",
    ),
    "frustration": (
        "きつい",
        "無理",
        "やばい",
        "なんで",
        "つらい",
        "厳しい",
        "しんどい",
    ),
    "close_call": (
        "惜しい",
        "あと少し",
        "あとちょっと",
        "いけた",
        "当たった",
        "もう少し",
    ),
    "success": (
        "ナイス",
        "勝った",
        "よし",
        "いいね",
        "うまい",
        "取れた",
        "倒した",
    ),
    "angry": (
        "ふざけんな",
        "それはない",
        "おかしい",
        "意味わからん",
        "なんだよ",
        "ありえない",
    ),
}


DEFAULT_STREAMER_FIXED_RESPONSE_PHRASES = {
    "frustration": (
        "今のはさすがにきついわね。",
        "これは悔しいやつね。",
        "それは声出るわね。",
    ),
    "close_call": (
        "今のは惜しかったわね。",
        "そこまで行けたのはいいわね。",
        "もう少しだったわね。",
    ),
    "success": (
        "いい流れじゃない。",
        "今のは気持ちいいわね。",
        "それはナイスね。",
    ),
    "angry": (
        "はいはい、そういうことするのね。",
        "それはちょっと嫌すぎるわ。",
        "今のは無理があるわね。",
    ),
    "generic": (
        "いいですね。",
        "その感じでいきましょう。",
        "見ていますよ。",
    ),
}


def normalize_streamer_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    return "".join(normalized.split()).casefold()


def _get_mapping(cfg, name: str, default):
    value = getattr(cfg, name, None)
    return value or default


def _contains_keyword(text: str, keywords: Iterable[str]) -> Tuple[str, ...]:
    matched = []
    for keyword in keywords:
        normalized_keyword = normalize_streamer_text(keyword)
        if normalized_keyword and normalized_keyword in text:
            matched.append(str(keyword))
    return tuple(matched)


def _contains_call_keyword(normalized_text: str, cfg) -> Tuple[str, ...]:
    keywords = []
    for keyword in tuple(getattr(cfg, "shizuku_call_keywords", ()) or ()):
        if keyword not in keywords:
            keywords.append(keyword)
    for keyword in tuple(getattr(cfg, "streamer_force_reply_keywords", ()) or ()):
        if keyword not in keywords:
            keywords.append(keyword)
    return _contains_keyword(normalized_text, keywords)


def classify_streamer_utterance(text: str, cfg) -> Dict[str, Any]:
    normalized = normalize_streamer_text(text)
    if not normalized:
        return {"utterance_type": "short_noise", "matched_keywords": (), "reason": "empty_text"}

    call_matches = _contains_call_keyword(normalized, cfg)
    if call_matches:
        return {"utterance_type": "addressed", "matched_keywords": call_matches, "reason": "addressed"}

    keywords_by_type = _get_mapping(cfg, "streamer_utterance_keywords", DEFAULT_STREAMER_UTTERANCE_KEYWORDS)
    for utterance_type in ("short_noise", "angry", "frustration", "close_call", "success"):
        matched = _contains_keyword(normalized, keywords_by_type.get(utterance_type, ()))
        if matched:
            return {
                "utterance_type": utterance_type,
                "matched_keywords": matched,
                "reason": "keyword_match",
            }

    return {"utterance_type": "generic", "matched_keywords": (), "reason": "no_keyword_match"}


def select_streamer_fixed_response(utterance_type: str, cfg) -> str:
    phrases_by_type = _get_mapping(
        cfg,
        "streamer_fixed_response_phrases",
        DEFAULT_STREAMER_FIXED_RESPONSE_PHRASES,
    )
    phrases = tuple(p for p in phrases_by_type.get(utterance_type, ()) if p)
    if not phrases:
        phrases = tuple(p for p in phrases_by_type.get("generic", ()) if p)
    return random.choice(phrases) if phrases else ""


def _cooldown_remaining(now: float, last_time: float, cooldown_sec: float) -> float:
    return max(0.0, max(0.0, cooldown_sec) - (now - last_time))


def decide_streamer_response(text: str, now: float, state, cfg) -> Dict[str, Any]:
    classification = classify_streamer_utterance(text, cfg)
    utterance_type = classification["utterance_type"]
    matched_keywords = classification["matched_keywords"]

    if utterance_type == "addressed":
        return {
            "action": "llm",
            "utterance_type": utterance_type,
            "phrase": "",
            "reason": "addressed",
            "matched_keywords": matched_keywords,
            "cooldown_remaining": 0.0,
        }

    ai_remaining = _cooldown_remaining(
        now,
        getattr(state, "last_assistant_speech_time", 0.0),
        float(getattr(cfg, "ai_speech_cooldown_sec", 0.0)),
    )
    if ai_remaining > 0:
        return {
            "action": "skip",
            "utterance_type": utterance_type,
            "phrase": "",
            "reason": "ai_speech_cooldown",
            "matched_keywords": matched_keywords,
            "cooldown_remaining": ai_remaining,
        }

    if utterance_type == "short_noise" and bool(getattr(cfg, "streamer_short_noise_skip", True)):
        return {
            "action": "skip",
            "utterance_type": utterance_type,
            "phrase": "",
            "reason": "short_noise",
            "matched_keywords": matched_keywords,
            "cooldown_remaining": 0.0,
        }

    fixed_enabled = bool(getattr(cfg, "streamer_fixed_response_enabled", True))
    fixed_types = {"frustration", "close_call", "success", "angry"}
    probability = min(1.0, max(0.0, float(getattr(cfg, "streamer_response_probability", 0.25))))

    if utterance_type == "generic" and random.random() > probability:
        return {
            "action": "skip",
            "utterance_type": utterance_type,
            "phrase": "",
            "reason": "probability",
            "matched_keywords": matched_keywords,
            "cooldown_remaining": 0.0,
        }

    fixed_rate = min(1.0, max(0.0, float(getattr(cfg, "streamer_fixed_response_rate", 0.5))))
    should_try_fixed = fixed_enabled and (utterance_type in fixed_types or utterance_type == "generic")
    if should_try_fixed and random.random() <= fixed_rate:
        fixed_remaining = _cooldown_remaining(
            now,
            getattr(state, "last_streamer_fixed_response_time", 0.0),
            float(getattr(cfg, "streamer_fixed_response_cooldown_sec", 15.0)),
        )
        if fixed_remaining > 0:
            return {
                "action": "skip",
                "utterance_type": utterance_type,
                "phrase": "",
                "reason": "fixed_response_cooldown",
                "matched_keywords": matched_keywords,
                "cooldown_remaining": fixed_remaining,
            }
        phrase = select_streamer_fixed_response(utterance_type, cfg)
        if phrase:
            return {
                "action": "fixed_phrase",
                "utterance_type": utterance_type,
                "phrase": phrase,
                "reason": "emotion_match" if utterance_type in fixed_types else "generic_fixed",
                "matched_keywords": matched_keywords,
                "cooldown_remaining": 0.0,
            }

    if bool(getattr(cfg, "streamer_llm_on_address_only", True)):
        return {
            "action": "skip",
            "utterance_type": utterance_type,
            "phrase": "",
            "reason": "llm_address_only",
            "matched_keywords": matched_keywords,
            "cooldown_remaining": 0.0,
        }

    if utterance_type != "generic" and random.random() > probability:
        return {
            "action": "skip",
            "utterance_type": utterance_type,
            "phrase": "",
            "reason": "probability",
            "matched_keywords": matched_keywords,
            "cooldown_remaining": 0.0,
        }

    return {
        "action": "llm",
        "utterance_type": utterance_type,
        "phrase": "",
        "reason": "fallback_llm",
        "matched_keywords": matched_keywords,
        "cooldown_remaining": 0.0,
    }
