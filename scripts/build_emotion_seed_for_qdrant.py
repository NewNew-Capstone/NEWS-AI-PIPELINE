from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from kiwipiepy import Kiwi

sys.path.insert(0, str(Path(__file__).parent.parent))

KNU_PATH = Path("analysis_bc/data/SentiWord_info.json")
KOTE_RAW_DIR = Path("analysis_bc/data/eval/raw/kote")
DEFAULT_OUTPUT = Path("analysis_bc/data/eval/processed/kote/emotion_seed_for_qdrant.jsonl")
DEFAULT_REJECTED_OUTPUT = Path("analysis_bc/data/eval/processed/kote/emotion_seed_rejected_for_review.jsonl")
DECISIONS_PATH = Path("analysis_bc/tagger/emotion_stopword_review_decisions.json")

_VALID_POS = {"NNG", "VV", "VA", "XR"}
_KOREAN_RE = re.compile(r"[가-힣]")
_KIWI: Kiwi | None = None

# KOTE 44 labels (dataset card 순서)
_KOTE_LABELS = [
    "불평/불만", "환영/호의", "감동/감탄", "지긋지긋", "고마움", "슬픔", "화남/분노", "존경", "기대감",
    "우쭐댐/무시함", "안타까움/실망", "비장함", "의심/불신", "뿌듯함", "편안/쾌적", "신기함/관심",
    "아껴주는", "부끄러움", "공포/무서움", "절망", "한심함", "역겨움/징그러움", "짜증", "어이없음",
    "없음", "패배/자기혐오", "귀찮음", "힘듦/지침", "즐거움/신남", "깨달음", "죄책감", "증오/혐오",
    "흐뭇함(귀여움/예쁨)", "당황/난처", "경악", "부담/안_내킴", "서러움", "재미없음", "불쌍함/연민",
    "놀람", "행복", "불안/걱정", "기쁨", "안심/신뢰",
]

_NEGATIVE_LABELS = {
    "불평/불만", "지긋지긋", "슬픔", "화남/분노", "우쭐댐/무시함", "안타까움/실망", "의심/불신",
    "공포/무서움", "절망", "한심함", "역겨움/징그러움", "짜증", "어이없음", "패배/자기혐오",
    "귀찮음", "힘듦/지침", "죄책감", "증오/혐오", "당황/난처", "경악", "부담/안_내킴", "서러움",
    "재미없음", "불안/걱정",
}
_POSITIVE_LABELS = {
    "환영/호의", "감동/감탄", "고마움", "존경", "기대감", "뿌듯함", "편안/쾌적", "신기함/관심",
    "아껴주는", "즐거움/신남", "깨달음", "흐뭇함(귀여움/예쁨)", "놀람", "행복", "기쁨", "안심/신뢰",
}

_BLACKLIST = {
    "가량", "주의", "시장", "인사", "경상", "해지",
    "지우다", "흐르다", "궁금하다", "정한", "정하다", "똑같다",
    "친구", "이야기", "이야기하다", "추진", "추진하다",
}

_KOTE_STOPWORDS = {
    "미국", "대통령", "뉴스", "문제", "이익", "영향", "산업", "세계", "한국", "대만", "제조업",
    "반도체", "거래", "인식", "자신", "절반", "확보", "희망", "기대", "주장",
}

_EMOTION_NNG_WHITELIST = {
    "분노", "화", "짜증", "혐오", "증오", "경악", "불안", "걱정", "공포", "두려움",
    "슬픔", "절망", "실망", "불만", "의심", "불신", "분개", "분개심", "분함", "불쾌",
    "기쁨", "행복", "환영", "감동", "감탄", "고마움", "안심", "신뢰", "즐거움",
    "연민", "죄책감", "당황", "부끄러움", "귀찮음", "지침", "피로", "우려",
}

_KNU_KEEP_POS_PREFIXES = ("VA", "VV", "XR")
_KNU_DESCRIPTOR_NOUNS = {
    "모양", "데", "정도", "사람", "것", "수", "상태", "경우", "느낌", "적",
}
_KNU_GENERIC_NOUNS = {
    "가격", "가능", "가능성", "친구", "이야기", "추진",
}
_KNU_GENERIC_PREDICATES = {
    "가누", "있", "없", "못하", "하", "되",
    # KNU 설명구에서 보조/일반 서술어만 뼈대로 남으면 Qdrant exact hit
    # seed가 되어 ACCEPT 오탐을 크게 늘린다.
    "만들", "보이", "일어나", "움직이", "이루어지", "이기", "바라",
    "나오", "끝나", "오래", "충분", "벌어지", "취하", "위하", "세우",
    "빛나", "놓치", "나아가", "나가", "지나가", "내놓", "흘러가",
    "앞서", "살펴보", "보내", "만나", "따르", "드러나", "대하",
    "넘기", "갖추", "가져가", "지내", "올리", "이끌",
    "깨", "시키",
    "지우", "흐르", "궁금", "궁금하", "정한", "정하", "똑같",
    "이야기", "추진", "추진하",
}
_KNU_EVALUATIVE_NNG_WHITELIST = _EMOTION_NNG_WHITELIST | {
    "가난", "거북",
}
_KNU_ADJECTIVE_NOUNS = {
    "가난", "거북",
}
_KNU_RAW_BLACKLIST = {
    "시장", "인사", "경상", "잘", "해지", "친구", "이야기", "추진",
}
_LEXICALIZED_NOUN_VERB_SEEDS = {
    ("짜증", "나"): "짜증나다",
    ("화", "나"): "화나다",
}
_DECISION_BLOCKLIST: frozenset[str] | None = None


def _surface_from_decision_item(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if not isinstance(item, dict):
        return None
    for key in ("surface", "keyword", "keyword_text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load_review_decision_blocklist(path: Path = DECISIONS_PATH) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()

    words: set[str] = set()
    for section in (
        "block_confirmed",
        "review_promoted_from_added_data",
        "auto_review_blocked",
    ):
        for item in raw.get(section, []):
            surface = _surface_from_decision_item(item)
            if surface:
                words.add(surface)
    return frozenset(words)


def _review_decision_blocklist() -> frozenset[str]:
    global _DECISION_BLOCKLIST
    if _DECISION_BLOCKLIST is None:
        _DECISION_BLOCKLIST = _load_review_decision_blocklist()
    return _DECISION_BLOCKLIST


def _get_kiwi() -> Kiwi:
    global _KIWI
    if _KIWI is None:
        _KIWI = Kiwi()
    return _KIWI


def _normalize_polarity(value: object) -> int | None:
    try:
        polarity = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return polarity if polarity != 0 else None


def _is_keep_pos(tag: str) -> bool:
    return tag == "XR" or tag.startswith(_KNU_KEEP_POS_PREFIXES)


def _is_generic_form(form: str) -> bool:
    base = form[:-1] if form.endswith("다") else form
    predicate = _predicate_seed(base)
    root = _root_seed(base)
    decision_blocklist = _review_decision_blocklist()
    return (
        form in decision_blocklist
        or base in decision_blocklist
        or predicate in decision_blocklist
        or root in decision_blocklist
        or base in _BLACKLIST
        or base in _KNU_DESCRIPTOR_NOUNS
        or base in _KNU_GENERIC_NOUNS
        or base in _KNU_GENERIC_PREDICATES
    )


def _is_allowed_knu_nng(form: str) -> bool:
    if form in _KNU_DESCRIPTOR_NOUNS or form in _KNU_GENERIC_NOUNS:
        return False
    return form in _KNU_EVALUATIVE_NNG_WHITELIST


def _predicate_seed(form: str) -> str:
    return form if form.endswith("다") else f"{form}다"


def _root_seed(form: str) -> str:
    return form if form.endswith("하다") else f"{form}하다"


def _category_for_clean_seed(clean_seed: str, polarity: int) -> str:
    if clean_seed in _EMOTION_NNG_WHITELIST:
        return "DIRECT_EMOTION"
    if polarity < 0:
        return "NEGATIVE_STATE"
    return "EVALUATIVE_WORD"


def _reject_row(row: dict, reason: str, polarity: int | None = None) -> dict:
    return {
        "original_word": str(row.get("word", "")).strip(),
        "original_word_root": str(row.get("word_root", "")).strip(),
        "clean_seed": None,
        "polarity": polarity,
        "is_valid_emotion": False,
        "reject_reason": reason,
        "source": "knu",
    }


def _content_tokens(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for token in _get_kiwi().tokenize(text):
        form = token.form.strip()
        if not form or not _is_korean(form):
            continue
        if token.tag.startswith("J") or token.tag.startswith("E") or token.tag.startswith("XS"):
            continue
        if form in _KNU_DESCRIPTOR_NOUNS:
            continue
        tokens.append((form, token.tag))
    return tokens


def _clean_seed_from_text(text: str) -> tuple[str | None, str | None]:
    content_tokens = _content_tokens(text)
    for idx, (form, tag) in enumerate(content_tokens[:-1]):
        next_form, next_tag = content_tokens[idx + 1]
        seed = _LEXICALIZED_NOUN_VERB_SEEDS.get((form, next_form))
        if seed and tag == "NNG" and next_tag.startswith("VV"):
            return seed, None

    direct_nouns: list[str] = []
    predicates: list[str] = []
    roots: list[str] = []
    adjective_nouns: list[str] = []

    for form, tag in content_tokens:
        if _is_generic_form(form):
            continue
        if tag == "NNG":
            if form in _EMOTION_NNG_WHITELIST:
                direct_nouns.append(form)
            elif form in _KNU_ADJECTIVE_NOUNS:
                adjective_nouns.append(_root_seed(form))
            continue
        if tag.startswith(("VA", "VV")):
            predicates.append(_predicate_seed(form))
            continue
        if tag == "XR":
            roots.append(_root_seed(form))

    if direct_nouns:
        return direct_nouns[0], None
    if predicates:
        return predicates[0], None
    if roots:
        return roots[0], None
    if adjective_nouns:
        return adjective_nouns[0], None
    return None, "no_emotion_candidate"


def clean_knu_entry(row: dict) -> dict:
    word = str(row.get("word", "")).strip()
    root = str(row.get("word_root", "")).strip()
    polarity = _normalize_polarity(row.get("polarity"))
    if polarity is None:
        return _reject_row(row, "invalid_polarity")

    source_text = root or word
    if not source_text:
        return _reject_row(row, "empty_text", polarity)
    if not _is_korean(source_text):
        return _reject_row(row, "non_korean", polarity)

    clean_seed, reject_reason = _clean_seed_from_text(source_text)
    if clean_seed is None:
        return _reject_row(row, reject_reason or "no_emotion_candidate", polarity)

    return {
        "original_word": word,
        "original_word_root": root,
        "clean_seed": clean_seed,
        "polarity": polarity,
        "is_valid_emotion": True,
        "reject_reason": None,
        "source": "knu",
        "category": _category_for_clean_seed(clean_seed, polarity),
    }


def _seed_row_from_clean(cleaned: dict, source: str, **extra: object) -> dict:
    clean_seed = str(cleaned["clean_seed"])
    return {
        "embed_text": clean_seed,
        "word": clean_seed,
        "word_root": clean_seed,
        "clean_seed": clean_seed,
        "polarity": str(cleaned["polarity"]),
        "source": source,
        "category": cleaned.get("category"),
        "confidence_tier": "HIGH",
        "original_word": cleaned.get("original_word"),
        "original_word_root": cleaned.get("original_word_root"),
        **extra,
    }


def _is_blocked_seed_text(text: str) -> bool:
    normalized = text.strip()
    if not normalized or not _is_korean(normalized):
        return True
    decision_blocklist = _review_decision_blocklist()
    if normalized in decision_blocklist or normalized in _BLACKLIST or normalized in _KNU_RAW_BLACKLIST:
        return True
    compact = normalized.replace(" ", "")
    if any(blocked and blocked.replace(" ", "") in compact for blocked in decision_blocklist):
        return True
    for form, _tag in _content_tokens(normalized):
        if _is_generic_form(form):
            return True
    return False


def _seed_row_from_knu_surface(
    row: dict,
    *,
    embed_text: str,
    polarity: int,
    cleaned: dict | None,
) -> dict:
    canonical_seed = None
    category = "BROAD_KNU"
    if cleaned and cleaned.get("is_valid_emotion"):
        canonical_seed = str(cleaned.get("clean_seed") or "")
        category = str(cleaned.get("category") or category)

    return {
        "embed_text": embed_text,
        "word": str(row.get("word", "")).strip(),
        "word_root": str(row.get("word_root", "")).strip(),
        "clean_seed": embed_text,
        "canonical_seed": canonical_seed,
        "polarity": str(polarity),
        "source": "knu_surface",
        "category": category,
        "confidence_tier": "BROAD",
        "original_word": str(row.get("word", "")).strip(),
        "original_word_root": str(row.get("word_root", "")).strip(),
    }


def _surface_seed_rows_from_knu_entry(row: dict, cleaned: dict | None = None) -> list[dict]:
    polarity = _normalize_polarity(row.get("polarity"))
    if polarity is None:
        return []

    rows: list[dict] = []
    seen: set[str] = set()
    for key in ("word", "word_root"):
        text = str(row.get(key, "")).strip()
        if text in seen:
            continue
        seen.add(text)
        if _is_blocked_seed_text(text):
            continue
        rows.append(
            _seed_row_from_knu_surface(
                row,
                embed_text=text,
                polarity=polarity,
                cleaned=cleaned,
            )
        )
    return rows


def skeletonize_knu_entry(row: dict) -> dict | None:
    cleaned = clean_knu_entry(row)
    if not cleaned["is_valid_emotion"]:
        return None
    return _seed_row_from_clean(cleaned, "knu", score_hint=1.0)


def _is_korean(text: str) -> bool:
    return bool(text and _KOREAN_RE.search(text))


def _read_knu_entries_with_rejected() -> tuple[list[dict], list[dict]]:
    with KNU_PATH.open(encoding="utf-8") as f:
        rows = json.load(f)

    out: list[dict] = []
    rejected: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cleaned = clean_knu_entry(row)
        if cleaned["is_valid_emotion"]:
            seed_row = _seed_row_from_clean(cleaned, "knu", score_hint=1.0)
            key = (seed_row["embed_text"], seed_row["polarity"])
            if key not in seen:
                seen.add(key)
                out.append(seed_row)
        else:
            rejected.append(cleaned)

        for surface_row in _surface_seed_rows_from_knu_entry(
            row,
            cleaned if cleaned["is_valid_emotion"] else None,
        ):
            key = (surface_row["embed_text"], surface_row["polarity"])
            if key in seen:
                continue
            seen.add(key)
            out.append(surface_row)
    return out, rejected


def _read_knu_entries() -> list[dict]:
    return _read_knu_entries_with_rejected()[0]


def _row_polarity(label_ids: list[int]) -> str | None:
    names = [
        _KOTE_LABELS[i] for i in label_ids
        if isinstance(i, int) and 0 <= i < len(_KOTE_LABELS)
    ]
    if not names:
        return None
    if "없음" in names:
        return None

    neg = sum(1 for n in names if n in _NEGATIVE_LABELS)
    pos = sum(1 for n in names if n in _POSITIVE_LABELS)
    if neg == pos:
        return None
    return "-1" if neg > pos else "1"


def _clean_kote_token(surface: str, form: str, pos: str) -> str | None:
    if surface in _BLACKLIST or surface in _KOTE_STOPWORDS:
        return None
    if _is_generic_form(form) or _is_generic_form(surface):
        return None
    if pos == "NNG":
        if form in _EMOTION_NNG_WHITELIST:
            return form
        if form in _KNU_ADJECTIVE_NOUNS:
            return _root_seed(form)
        return None
    if pos.startswith(("VA", "VV")):
        return _predicate_seed(form)
    if pos == "XR":
        return _root_seed(form)
    return None


def _extract_kote_entries(min_freq: int, min_polarity_ratio: float) -> list[dict]:
    kiwi = Kiwi()

    total_count: defaultdict[str, int] = defaultdict(int)
    neg_count: defaultdict[str, int] = defaultdict(int)
    pos_count: defaultdict[str, int] = defaultdict(int)

    for fp in sorted(KOTE_RAW_DIR.glob("*.jsonl")):
        with fp.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = str(row.get("text", "")).strip()
                labels = row.get("labels", [])
                if not text or not isinstance(labels, list):
                    continue

                pol = _row_polarity(labels)
                if pol is None:
                    continue

                seen_in_sentence: set[str] = set()
                for token in kiwi.tokenize(text):
                    if token.tag not in _VALID_POS:
                        continue
                    surface = text[token.start:token.start + token.len].strip()
                    if len(surface) < 2:
                        continue
                    if not _is_korean(surface):
                        continue
                    clean_seed = _clean_kote_token(surface, token.form.strip(), token.tag)
                    if clean_seed is None:
                        continue
                    if clean_seed in seen_in_sentence:
                        continue
                    seen_in_sentence.add(clean_seed)

                for w in seen_in_sentence:
                    total_count[w] += 1
                    if pol == "-1":
                        neg_count[w] += 1
                    else:
                        pos_count[w] += 1

    out: list[dict] = []
    for clean_seed, total in total_count.items():
        if total < min_freq:
            continue
        neg = neg_count[clean_seed]
        pos = pos_count[clean_seed]
        dominant = max(neg, pos)
        ratio = dominant / total if total else 0.0
        if ratio < min_polarity_ratio:
            continue

        polarity = "-1" if neg >= pos else "1"
        category = _category_for_clean_seed(clean_seed, int(polarity))
        out.append({
            "embed_text": clean_seed,
            "word": clean_seed,
            "word_root": clean_seed,
            "clean_seed": clean_seed,
            "polarity": polarity,
            "source": "kote",
            "category": category,
            "confidence_tier": "HIGH",
            "score_hint": round(ratio, 4),
            "count_hint": total,
            "neg_count": neg,
            "pos_count": pos,
        })

    return out


def _merge_key(row: dict) -> tuple[str, str]:
    return (str(row["embed_text"]), str(row["polarity"]))


def _merge_entries(knu: list[dict], kote: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}

    for row in knu:
        key = _merge_key(row)
        merged[key] = row

    for row in kote:
        key = _merge_key(row)
        if key in merged:
            existing = merged[key]
            src = existing.get("source", "")
            existing["source"] = "knu+kote" if "kote" not in src else src
            existing["kote_score_hint"] = row.get("score_hint")
            existing["kote_count_hint"] = row.get("count_hint")
            existing["kote_neg_count"] = row.get("neg_count")
            existing["kote_pos_count"] = row.get("pos_count")
            continue
        merged[key] = row

    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build emotion seed JSONL from KOTE + KNU.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected-output", type=Path, default=DEFAULT_REJECTED_OUTPUT)
    parser.add_argument("--min-freq", type=int, default=20)
    parser.add_argument("--min-polarity-ratio", type=float, default=0.7)
    args = parser.parse_args()

    knu, rejected = _read_knu_entries_with_rejected()
    kote = _extract_kote_entries(
        min_freq=args.min_freq,
        min_polarity_ratio=args.min_polarity_ratio,
    )
    merged = _merge_entries(knu, kote)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.rejected_output.parent.mkdir(parents=True, exist_ok=True)
    with args.rejected_output.open("w", encoding="utf-8") as f:
        for row in rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"KNU entries   : {len(knu)}")
    print(f"KNU rejected  : {len(rejected)}")
    print(f"KOTE entries  : {len(kote)}")
    print(f"Merged entries: {len(merged)}")
    print(f"Output        : {args.output}")
    print(f"Rejected      : {args.rejected_output}")


if __name__ == "__main__":
    main()
