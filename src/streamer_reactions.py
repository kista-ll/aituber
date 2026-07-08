import random
import unicodedata
from typing import Any, Dict, Iterable, Tuple


DEFAULT_ADDRESS_STRONG_KEYWORDS = (
    "しずく",
    "雫",
    "しづく",
    "シズク",
    "しずくちゃん",
    "しーちゃん",
)

DEFAULT_ADDRESS_WEAK_KEYWORDS = (
    "しず",
    "シズ",
    "しずこ",
    "静岡",
    "しずおか",
    "しずか",
    "続く",
)

DEFAULT_ADDRESS_CONTEXT_WORDS = (
    "どう",
    "今の",
    "これ",
    "見て",
    "聞いて",
    "思う",
    "教えて",
    "お願い",
    "反応して",
    "何",
    "なんで",
    "自己紹介",
    "挨拶",
    "説明",
    "あなた誰",
    "何者",
)

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


DEFAULT_CHARACTER_BREAK_FIXED_RESPONSE_PHRASES = {
    "frustration": (
        "いや今のは普通にきついわね。",
        "それ通すの、なかなか嫌な感じね。",
        "今のはちょっと声出るやつね。",
    ),
    "close_call": (
        "惜しい、今のは取れてもよかったわね。",
        "そこまで行ったなら、もう一回いけるわ。",
        "今のあと一歩、かなり惜しいわね。",
    ),
    "success": (
        "いいじゃない、ちょっと調子出てきたわね。",
        "今のはだいぶ気持ちいいわね。",
        "はい、今のはちゃんと偉いです。",
    ),
    "angry": (
        "はい出ました、嫌なやつですね。",
        "それはちょっと無理があるわね。",
        "今のはさすがに文句出るわね。",
    ),
    "generic": (
        "見てるわよ、ちゃんとね。",
        "その感じ、悪くないわね。",
        "ちょっと荒れてきたけど、まだいけるわ。",
    ),
}


DEFAULT_STREAMER_KEYWORD_FIXED_RESPONSES = {
    "self_intro": {
        "keywords": (
            "自己紹介",
            "初見さん",
            "挨拶",
            "あなた誰",
            "何者",
            "説明して",
        ),
        "phrases": (
            "月野しずくです。近所に住んでるゲーム好きのお姉さんみたいな立ち位置で、配信を横から見ています。対戦中は少しだけ口が悪くなることがあります。",
        ),
    },
}


def normalize_streamer_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    return "".join(normalized.split()).casefold()


def _get_mapping(cfg, name: str, default):
    value = getattr(cfg, name, None)
    return value or default


def _get_sequence(cfg, name: str, default) -> Tuple[str, ...]:
    value = getattr(cfg, name, None)
    return tuple(value or default)


def _contains_keyword(text: str, keywords: Iterable[str]) -> Tuple[str, ...]:
    matched = []
    for keyword in keywords:
        normalized_keyword = normalize_streamer_text(keyword)
        if normalized_keyword and normalized_keyword in text:
            matched.append(str(keyword))
    return tuple(matched)


def _first_keyword_match(text: str, keywords: Iterable[str]) -> str:
    matched = _contains_keyword(text, keywords)
    return matched[0] if matched else ""


def _weak_context_match(normalized_text: str, context_words: Iterable[str]) -> str:
    for context_word in context_words:
        normalized_context = normalize_streamer_text(context_word)
        if not normalized_context or normalized_context not in normalized_text:
            continue
        if normalized_context in {"思う", "おもう"}:
            if "どう思" not in normalized_text and "どうおも" not in normalized_text:
                continue
        return str(context_word)
    return ""


def classify_address_match(normalized_text: str, cfg) -> Dict[str, Any]:
    strong_keywords = list(_get_sequence(cfg, "shizuku_address_strong_keywords", DEFAULT_ADDRESS_STRONG_KEYWORDS))
    for keyword in tuple(getattr(cfg, "shizuku_call_keywords", ()) or ()):  # backwards compatible
        if keyword not in strong_keywords:
            strong_keywords.append(keyword)

    strong_match = _first_keyword_match(normalized_text, strong_keywords)
    if strong_match:
        return {
            "address_match_type": "strong",
            "matched_address_keyword": strong_match,
            "matched_context_word": "",
        }

    weak_keywords = _get_sequence(cfg, "shizuku_address_weak_keywords", DEFAULT_ADDRESS_WEAK_KEYWORDS)
    weak_match = _first_keyword_match(normalized_text, weak_keywords)
    if weak_match:
        context_words = _get_sequence(cfg, "shizuku_address_context_words", DEFAULT_ADDRESS_CONTEXT_WORDS)
        context_match = _weak_context_match(normalized_text, context_words)
        if context_match:
            return {
                "address_match_type": "weak_context",
                "matched_address_keyword": weak_match,
                "matched_context_word": context_match,
            }

    return {
        "address_match_type": "none",
        "matched_address_keyword": "",
        "matched_context_word": "",
    }


def _contains_call_keyword(normalized_text: str, cfg) -> Tuple[str, ...]:
    address_match = classify_address_match(normalized_text, cfg)
    if address_match["address_match_type"] == "none":
        return ()
    return (address_match["matched_address_keyword"],)


def classify_streamer_utterance(text: str, cfg) -> Dict[str, Any]:
    normalized = normalize_streamer_text(text)
    base = {
        "normalized_text": normalized,
        "address_match_type": "none",
        "matched_address_keyword": "",
        "matched_context_word": "",
    }
    if not normalized:
        return {**base, "utterance_type": "short_noise", "matched_keywords": (), "reason": "empty_text"}

    address_match = classify_address_match(normalized, cfg)
    base.update(address_match)
    if address_match["address_match_type"] != "none":
        matched_keywords = (address_match["matched_address_keyword"],)
        if address_match["matched_context_word"]:
            matched_keywords += (address_match["matched_context_word"],)
        return {
            **base,
            "utterance_type": "addressed",
            "matched_keywords": matched_keywords,
            "reason": "addressed",
        }

    keywords_by_type = _get_mapping(cfg, "streamer_utterance_keywords", DEFAULT_STREAMER_UTTERANCE_KEYWORDS)
    for utterance_type in ("short_noise", "angry", "frustration", "close_call", "success"):
        matched = _contains_keyword(normalized, keywords_by_type.get(utterance_type, ()))
        if matched:
            return {
                **base,
                "utterance_type": utterance_type,
                "matched_keywords": matched,
                "reason": "keyword_match",
            }

    return {**base, "utterance_type": "generic", "matched_keywords": (), "reason": "no_keyword_match"}


def _is_battle_mode(cfg) -> bool:
    return str(getattr(cfg, "game_mode", "normal") or "normal").strip().lower() == "battle"


def _character_break_remaining(now: float, state) -> float:
    return max(0.0, float(getattr(state, "character_break_until", 0.0) or 0.0) - now)


def _character_break_cooldown_remaining(now: float, state, cfg) -> float:
    return _cooldown_remaining(
        now,
        float(getattr(state, "last_character_break_time", 0.0) or 0.0),
        float(getattr(cfg, "character_break_cooldown_sec", 300.0)),
    )


def maybe_update_character_break(now: float, state, cfg) -> Dict[str, Any]:
    if not bool(getattr(cfg, "character_break_enabled", False)):
        return {
            "character_break_active": False,
            "character_break_triggered": False,
            "character_break_remaining": 0.0,
            "character_break_reason": "disabled",
        }
    if not _is_battle_mode(cfg):
        return {
            "character_break_active": False,
            "character_break_triggered": False,
            "character_break_remaining": 0.0,
            "character_break_reason": "not_battle_mode",
        }

    active_remaining = _character_break_remaining(now, state)
    if active_remaining > 0:
        return {
            "character_break_active": True,
            "character_break_triggered": False,
            "character_break_remaining": active_remaining,
            "character_break_reason": "active",
        }

    cooldown_remaining = _character_break_cooldown_remaining(now, state, cfg)
    if cooldown_remaining > 0:
        return {
            "character_break_active": False,
            "character_break_triggered": False,
            "character_break_remaining": cooldown_remaining,
            "character_break_reason": "cooldown",
        }

    rate = min(1.0, max(0.0, float(getattr(cfg, "character_break_rate", 0.02))))
    if random.random() > rate:
        return {
            "character_break_active": False,
            "character_break_triggered": False,
            "character_break_remaining": 0.0,
            "character_break_reason": "rate_miss",
        }

    duration = max(0.0, float(getattr(cfg, "character_break_duration_sec", 20.0)))
    setattr(state, "last_character_break_time", now)
    setattr(state, "character_break_until", now + duration)
    return {
        "character_break_active": True,
        "character_break_triggered": True,
        "character_break_remaining": duration,
        "character_break_reason": "triggered",
    }


def select_character_break_fixed_response(utterance_type: str, cfg) -> str:
    phrases_by_type = _get_mapping(
        cfg,
        "character_break_fixed_response_phrases",
        DEFAULT_CHARACTER_BREAK_FIXED_RESPONSE_PHRASES,
    )
    phrases = tuple(p for p in phrases_by_type.get(utterance_type, ()) if p)
    if not phrases:
        phrases = tuple(p for p in phrases_by_type.get("generic", ()) if p)
    return random.choice(phrases) if phrases else ""


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


def select_keyword_fixed_response(normalized_text: str, cfg) -> Dict[str, Any]:
    if not bool(getattr(cfg, "streamer_keyword_fixed_response_enabled", True)):
        return {"keyword_response_id": "", "phrase": "", "matched_keywords": (), "reason": "disabled"}

    responses = _get_mapping(
        cfg,
        "streamer_keyword_fixed_responses",
        DEFAULT_STREAMER_KEYWORD_FIXED_RESPONSES,
    )
    for response_id, spec in responses.items():
        keywords = tuple((spec or {}).get("keywords", ()) or ())
        matched = _contains_keyword(normalized_text, keywords)
        if not matched:
            continue
        phrases = tuple(p for p in (spec or {}).get("phrases", ()) if p)
        if not phrases:
            return {
                "keyword_response_id": str(response_id),
                "phrase": "",
                "matched_keywords": matched,
                "reason": "no_phrase",
            }
        return {
            "keyword_response_id": str(response_id),
            "phrase": random.choice(phrases),
            "matched_keywords": matched,
            "reason": "keyword_match",
        }

    return {"keyword_response_id": "", "phrase": "", "matched_keywords": (), "reason": "no_keyword_match"}


def _cooldown_remaining(now: float, last_time: float, cooldown_sec: float) -> float:
    return max(0.0, max(0.0, cooldown_sec) - (now - last_time))


def _decision_base(classification: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "utterance_type": classification.get("utterance_type", "generic"),
        "matched_keywords": classification.get("matched_keywords", ()),
        "normalized_text": classification.get("normalized_text", ""),
        "address_match_type": classification.get("address_match_type", "none"),
        "matched_address_keyword": classification.get("matched_address_keyword", ""),
        "matched_context_word": classification.get("matched_context_word", ""),
        "keyword_response_id": "",
        "llm_response_mode": "normal",
        "character_break_active": False,
        "character_break_triggered": False,
        "character_break_remaining": 0.0,
        "character_break_reason": "",
    }


def decide_streamer_response(text: str, now: float, state, cfg) -> Dict[str, Any]:
    classification = classify_streamer_utterance(text, cfg)
    utterance_type = classification["utterance_type"]
    matched_keywords = classification["matched_keywords"]
    base = _decision_base(classification)

    if utterance_type == "addressed":
        keyword_response = select_keyword_fixed_response(classification.get("normalized_text", ""), cfg)
        if keyword_response["phrase"]:
            return {
                **base,
                "action": "keyword_fixed_phrase",
                "phrase": keyword_response["phrase"],
                "reason": "keyword_fixed_response",
                "matched_keywords": matched_keywords + tuple(keyword_response.get("matched_keywords", ())),
                "keyword_response_id": keyword_response["keyword_response_id"],
                "cooldown_remaining": 0.0,
            }
        return {
            **base,
            "action": "llm_address",
            "phrase": "",
            "reason": "addressed",
            "matched_keywords": matched_keywords,
            "llm_response_mode": "address_long",
            "cooldown_remaining": 0.0,
        }

    ai_remaining = _cooldown_remaining(
        now,
        getattr(state, "last_assistant_speech_time", 0.0),
        float(getattr(cfg, "ai_speech_cooldown_sec", 0.0)),
    )
    if ai_remaining > 0:
        return {
            **base,
            "action": "skip",
            "phrase": "",
            "reason": "ai_speech_cooldown",
            "matched_keywords": matched_keywords,
            "cooldown_remaining": ai_remaining,
        }

    if utterance_type == "short_noise" and bool(getattr(cfg, "streamer_short_noise_skip", True)):
        return {
            **base,
            "action": "skip",
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
            **base,
            "action": "skip",
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
                **base,
                "action": "skip",
                "phrase": "",
                "reason": "fixed_response_cooldown",
                "matched_keywords": matched_keywords,
                "cooldown_remaining": fixed_remaining,
            }
        character_break = maybe_update_character_break(now, state, cfg)
        phrase = ""
        reason = "emotion_match" if utterance_type in fixed_types else "generic_fixed"
        if character_break["character_break_active"]:
            phrase = select_character_break_fixed_response(utterance_type, cfg)
            if phrase:
                reason = "character_break_fixed"
        if not phrase:
            phrase = select_streamer_fixed_response(utterance_type, cfg)
        if phrase:
            return {
                **base,
                **character_break,
                "action": "fixed_phrase",
                "phrase": phrase,
                "reason": reason,
                "matched_keywords": matched_keywords,
                "cooldown_remaining": 0.0,
            }

    if bool(getattr(cfg, "streamer_llm_on_address_only", True)):
        return {
            **base,
            "action": "skip",
            "phrase": "",
            "reason": "llm_address_only",
            "matched_keywords": matched_keywords,
            "cooldown_remaining": 0.0,
        }

    if utterance_type != "generic" and random.random() > probability:
        return {
            **base,
            "action": "skip",
            "phrase": "",
            "reason": "probability",
            "matched_keywords": matched_keywords,
            "cooldown_remaining": 0.0,
        }

    return {
        **base,
        "action": "llm",
        "phrase": "",
        "reason": "fallback_llm",
        "matched_keywords": matched_keywords,
        "cooldown_remaining": 0.0,
    }
