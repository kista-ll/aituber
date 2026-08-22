import random
import unicodedata
from typing import Any, Dict, Iterable, Tuple


WEAPON_CATEGORY_PRIORITY = (
    "special",
    "bomb",
    "charger",
    "blaster",
    "roller",
    "spinner",
    "shelter",
    "slosher",
    "maneuver",
    "brush",
    "splatana",
    "stringer",
    "shooter",
)


DEFAULT_DEATH_EVENT_WEAPON_KEYWORDS = {
    "roller": (
        "ローラー",
        "カーボン",
        "ヴァリアブル",
        "ダイナモ",
        "ワイドローラー",
    ),
    "charger": (
        "チャージャー",
        "スコープ",
        "リッター",
        "スクイックリン",
        "ソイチューバー",
    ),
    "blaster": (
        "ブラスター",
        "ホット",
        "ロング",
        "クラッシュ",
        "ノヴァ",
        "ラピッド",
    ),
    "spinner": (
        "スピナー",
        "バレル",
        "ハイドラ",
        "クーゲル",
        "ノーチラス",
        "イグザミナー",
    ),
    "shelter": (
        "シェルター",
        "パラシェルター",
        "キャンピング",
        "スパイガジェット",
        "24式張替傘",
    ),
    "slosher": (
        "スロッシャー",
        "バケツ",
        "ヒッセン",
        "エクスプロッシャー",
        "オーバーフロッシャー",
    ),
    "shooter": (
        "シューター",
        "スプラシューター",
        "わかば",
        "もみじ",
        "ZAP",
        "シャープマーカー",
        "ボールド",
        "プライム",
        "ジェットスイーパー",
    ),
    "maneuver": (
        "マニューバー",
        "スパッタリー",
        "デュアル",
        "クアッド",
        "ケルビン",
    ),
    "brush": (
        "フデ",
        "パブロ",
        "ホクサイ",
        "フィンセント",
    ),
    "splatana": (
        "ワイパー",
        "ジムワイパー",
        "ドライブワイパー",
    ),
    "stringer": (
        "ストリンガー",
        "トライストリンガー",
        "LACT",
    ),
    "bomb": (
        "ボム",
        "スプラッシュボム",
        "キューバン",
        "クイックボム",
        "ロボットボム",
        "タンサン",
        "トーピード",
    ),
    "special": (
        "ウルトラ",
        "ナイスダマ",
        "ジェットパック",
        "カニタンク",
        "トルネード",
        "メガホン",
        "サメライド",
        "テイオウ",
    ),
}


DEFAULT_DEATH_EVENT_EMOTIONS_BY_CATEGORY = {
    "roller": ("fear", "frustration"),
    "charger": ("annoyance", "respect"),
    "blaster": ("annoyance", "frustration"),
    "spinner": ("annoyance", "frustration"),
    "shelter": ("annoyance", "frustration"),
    "slosher": ("annoyance", "frustration"),
    "shooter": ("respect", "frustration"),
    "maneuver": ("annoyance", "respect"),
    "brush": ("tease", "annoyance"),
    "splatana": ("fear", "respect"),
    "stringer": ("annoyance", "respect"),
    "bomb": ("accident", "annoyance"),
    "special": ("fear", "frustration"),
    "unknown": ("frustration", "tease", "generic"),
}


DEFAULT_DEATH_EVENT_REACTIONS_BY_CATEGORY = {
    "roller": {
        "fear": (
            "ローラー、近いと怖いわね。",
            "そこまで詰められたら無理よ。",
        ),
        "frustration": (
            "またローラー……そのゴリ押し、もう見飽きたわ。",
            "塗りながら来るの、本当に面倒。",
        ),
    },
    "charger": {
        "annoyance": (
            "遠くから抜いてくるの、本当に面倒くさいわ。",
            "近づく前に撃つの、いい加減にして。",
        ),
        "respect": (
            "今の抜き方はうまいわね。",
            "相手、よく見てたわ。",
        ),
    },
    "blaster": {
        "annoyance": (
            "爆風、それはずるいわね。",
            "今の届くの、嫌すぎるわ。",
        ),
        "frustration": (
            "また爆風……直撃じゃないのが余計むかつくわ。",
            "爆風頼り、いい加減にしてほしいわ。",
        ),
    },
    "spinner": {
        "annoyance": (
            "いつまで弾ばら撒いてんのよ、うるさいわね。",
            "回しっぱなしで撃つの、本当に面倒。",
        ),
        "frustration": (
            "またスピナー……その弾幕、いい加減にして。",
            "弾の量で押し切るの、納得いかないわ。",
        ),
    },
    "shelter": {
        "annoyance": (
            "また傘開いてる……その守り、本当に面倒。",
            "傘に隠れて粘るの、いい加減にしなさいよ。",
        ),
        "frustration": (
            "その傘、いつまで攻撃止めるのよ。",
            "守りながら詰めてくるの、ほんと嫌。",
        ),
    },
    "slosher": {
        "annoyance": (
            "曲射、それはいやらしいわね。",
            "上から来るの、嫌すぎるわ。",
        ),
        "frustration": (
            "バケツぶちまけるだけで当たるの、納得いかないわ。",
            "その曲射、変なところから当たりすぎなのよ。",
        ),
    },
    "shooter": {
        "respect": (
            "相手、撃ち合い上手いわね。",
            "今の詰め方はきれいだったわ。",
        ),
        "frustration": (
            "はぁ……連射だけで勝った気になってるの？",
            "弾ばら撒くだけなら誰でもできるんだけど",
        ),
    },
    "maneuver": {
        "annoyance": (
            "スライドで逃げ回るの、いい加減にしなさいよ。",
            "その動き、追いかけるだけで疲れるわ。",
        ),
        "respect": (
            "相手の動き、かなり細かいわ。",
            "今のかわし方はうまいわね。",
        ),
    },
    "brush": {
        "tease": (
            "筆、元気に走ってきたわね。",
            "はいはい、そういう近づき方ね。",
        ),
        "annoyance": (
            "その走り回る筆、本当に落ち着きないわね。",
            "筆で撫でながら来るの、ほんと面倒。",
        ),
    },
    "splatana": {
        "fear": (
            "またワイパー……振り回すだけで届きすぎなのよ。",
            "その一振り、判定広すぎじゃない？",
        ),
        "respect": (
            "今の間合い、相手が上手いわ。",
            "詰め方がきれいだったわね。",
        ),
    },
    "stringer": {
        "annoyance": (
            "その弓、変なところから当ててくるのやめなさいよ。",
            "チャージ長いくせに、しっかり当ててくるの腹立つわ。",
        ),
        "respect": (
            "今の狙い、よく見てたわ。",
            "相手、射線管理がうまいわ。",
        ),
    },
    "bomb": {
        "accident": (
            "今のは事故みたいなものね。",
            "ボム、それは踏むわ。",
        ),
        "annoyance": (
            "置き方がいやらしいわね。",
            "そこに置くのは性格出てるわ。",
        ),
    },
    "special": {
        "fear": (
            "スペシャルの圧、すごいわね。",
            "今の展開は逃げ場が少ないわ。",
        ),
        "frustration": (
            "スペシャル絡みはきついわね。",
            "今のタイミングは悔しいわ。",
        ),
    },
    "unknown": {
        "frustration": (
            "今のはさすがにきついわね。",
            "これは悔しいやつね。",
        ),
        "tease": (
            "相手、やってるわね。",
            "はいはい、そういうことするのね。",
        ),
        "generic": (
            "これは声出るやつね。",
            "今のはつらいわね。",
        ),
    },
}


def normalize_death_ocr_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("！", "!")
    return "".join(normalized.split())


def _get_mapping(cfg, name: str, default):
    value = getattr(cfg, name, None)
    return value or default


def _normalize_keyword(keyword: str) -> str:
    return normalize_death_ocr_text(keyword).casefold()


def _iter_keywords(mapping: Dict[str, Iterable[str]], category: str) -> Tuple[str, ...]:
    return tuple(str(keyword) for keyword in mapping.get(category, ()) if str(keyword).strip())


def classify_death_weapon_category(ocr_text: str, ocr_confidence: float, cfg) -> str:
    result = classify_death_weapon_category_detail(ocr_text, ocr_confidence, cfg)
    return result["weapon_category"]


def classify_death_weapon_category_detail(ocr_text: str, ocr_confidence: float, cfg) -> Dict[str, Any]:
    normalized = normalize_death_ocr_text(ocr_text)
    min_confidence = float(getattr(cfg, "death_event_ocr_category_min_confidence", 60.0))
    if not normalized:
        return {
            "weapon_category": "unknown",
            "normalized_ocr_text": normalized,
            "matched_keywords": (),
            "category_reason": "empty_ocr_text",
        }
    if float(ocr_confidence or 0.0) < min_confidence:
        return {
            "weapon_category": "unknown",
            "normalized_ocr_text": normalized,
            "matched_keywords": (),
            "category_reason": "low_ocr_confidence",
        }

    haystack = normalized.casefold()
    keywords_by_category = _get_mapping(cfg, "death_event_weapon_keywords", DEFAULT_DEATH_EVENT_WEAPON_KEYWORDS)
    matched_by_category = {}
    for category in WEAPON_CATEGORY_PRIORITY:
        matched = [
            keyword
            for keyword in _iter_keywords(keywords_by_category, category)
            if _normalize_keyword(keyword) and _normalize_keyword(keyword) in haystack
        ]
        if matched:
            matched_by_category[category] = tuple(matched)

    for category in WEAPON_CATEGORY_PRIORITY:
        if category in matched_by_category:
            return {
                "weapon_category": category,
                "normalized_ocr_text": normalized,
                "matched_keywords": matched_by_category[category],
                "category_reason": "keyword_match",
            }

    return {
        "weapon_category": "unknown",
        "normalized_ocr_text": normalized,
        "matched_keywords": (),
        "category_reason": "no_keyword_match",
    }


def choose_death_emotion_category(weapon_category: str) -> str:
    emotions = DEFAULT_DEATH_EVENT_EMOTIONS_BY_CATEGORY.get(weapon_category)
    if not emotions:
        emotions = DEFAULT_DEATH_EVENT_EMOTIONS_BY_CATEGORY["unknown"]
    return random.choice(tuple(emotions))


def _choose_phrase(reactions_by_category, weapon_category: str, emotion_category: str) -> Tuple[str, str]:
    category_reactions = reactions_by_category.get(weapon_category, {})
    phrases = tuple(p for p in category_reactions.get(emotion_category, ()) if p)
    if phrases:
        source = "unknown_fixed_phrase" if weapon_category == "unknown" else "category_fixed_phrase"
        return random.choice(phrases), source

    for emotion, candidates in category_reactions.items():
        phrases = tuple(p for p in candidates if p)
        if phrases:
            source = "unknown_fixed_phrase" if weapon_category == "unknown" else "category_fixed_phrase"
            return random.choice(phrases), source

    unknown_reactions = reactions_by_category.get("unknown", {})
    for candidates in unknown_reactions.values():
        phrases = tuple(p for p in candidates if p)
        if phrases:
            return random.choice(phrases), "unknown_fixed_phrase"

    return "", "fallback_fixed_phrase"


def select_death_context_reaction(cfg) -> Dict[str, Any]:
    reactions_by_category = _get_mapping(
        cfg,
        "death_event_reactions_by_category",
        DEFAULT_DEATH_EVENT_REACTIONS_BY_CATEGORY,
    )
    emotion_category = choose_death_emotion_category("unknown")
    phrase, _source = _choose_phrase(reactions_by_category, "unknown", emotion_category)
    return {
        "phrase": phrase,
        "weapon_category": "unknown",
        "emotion_category": emotion_category,
        "reaction_source": "context_fixed_phrase" if phrase else "fallback_fixed_phrase",
        "normalized_ocr_text": "",
        "matched_keywords": (),
        "category_reason": "context_reaction",
    }


def select_death_reaction(event_details: dict, cfg) -> Dict[str, Any]:
    use_context = bool(getattr(cfg, "death_event_use_context_reactions", True))
    use_weapon_category = bool(getattr(cfg, "death_event_use_weapon_category_reactions", False))
    if not use_weapon_category:
        if use_context:
            return select_death_context_reaction(cfg)
        return {
            "phrase": "",
            "weapon_category": "unknown",
            "emotion_category": "",
            "reaction_source": "fallback_fixed_phrase",
            "normalized_ocr_text": "",
            "matched_keywords": (),
            "category_reason": "category_disabled",
        }

    details = event_details or {}
    ocr_reason = str(details.get("ocr_reason", "") or "")
    ocr_text = str(details.get("ocr_text", "") or "")
    ocr_confidence = float(details.get("ocr_confidence", 0.0) or 0.0)

    if ocr_reason and ocr_reason != "ok":
        category = {
            "weapon_category": "unknown",
            "normalized_ocr_text": normalize_death_ocr_text(ocr_text),
            "matched_keywords": (),
            "category_reason": f"ocr_reason_{ocr_reason}",
        }
    else:
        category = classify_death_weapon_category_detail(ocr_text, ocr_confidence, cfg)

    weapon_category = category["weapon_category"]
    if weapon_category == "unknown" and use_context:
        reaction = select_death_context_reaction(cfg)
        reaction["normalized_ocr_text"] = category["normalized_ocr_text"]
        reaction["matched_keywords"] = category["matched_keywords"]
        reaction["category_reason"] = category["category_reason"]
        return reaction

    emotion_category = choose_death_emotion_category(weapon_category)
    reactions_by_category = _get_mapping(
        cfg,
        "death_event_reactions_by_category",
        DEFAULT_DEATH_EVENT_REACTIONS_BY_CATEGORY,
    )
    phrase, reaction_source = _choose_phrase(reactions_by_category, weapon_category, emotion_category)

    return {
        "phrase": phrase,
        "weapon_category": weapon_category,
        "emotion_category": emotion_category,
        "reaction_source": reaction_source,
        "normalized_ocr_text": category["normalized_ocr_text"],
        "matched_keywords": category["matched_keywords"],
        "category_reason": category["category_reason"],
    }
