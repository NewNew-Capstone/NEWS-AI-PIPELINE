from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from kiwipiepy import Kiwi

sys.path.insert(0, str(Path(__file__).parent.parent))

KNU_PATH = Path("analysis_bc/data/SentiWord_info.json")
KOTE_RAW_DIR = Path("analysis_bc/data/eval/raw/kote")
DEFAULT_OUTPUT = Path("analysis_bc/data/eval/processed/kote/emotion_seed_for_qdrant.jsonl")

_VALID_POS = {"NNG", "NNP", "VV", "VA", "MAG", "XR"}
_KOREAN_RE = re.compile(r"[가-힣]")

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


def _is_korean(text: str) -> bool:
    return bool(text and _KOREAN_RE.search(text))


def _read_knu_entries() -> list[dict]:
    with KNU_PATH.open(encoding="utf-8") as f:
        rows = json.load(f)

    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        word = str(row.get("word", "")).strip()
        root = str(row.get("word_root", "")).strip()
        polarity = str(row.get("polarity", "0")).strip()
        if not word or not root:
            continue
        try:
            pol_int = int(polarity)
        except ValueError:
            continue
        if pol_int > -1:
            continue
        if root in _BLACKLIST:
            continue
        if not _is_korean(word) and not _is_korean(root):
            continue
        key = (word, root, polarity)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "embed_text": word,
            "word": word,
            "word_root": root,
            "polarity": str(pol_int),
            "source": "knu",
            "score_hint": 1.0,
        })
    return out


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


def _is_allowed_kote_token(surface: str, pos: str) -> bool:
    if surface in _BLACKLIST or surface in _KOTE_STOPWORDS:
        return False
    if pos == "NNG":
        return surface in _EMOTION_NNG_WHITELIST
    return True


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
                    if not _is_allowed_kote_token(surface, token.tag):
                        continue
                    if surface in seen_in_sentence:
                        continue
                    seen_in_sentence.add(surface)

                for w in seen_in_sentence:
                    total_count[w] += 1
                    if pol == "-1":
                        neg_count[w] += 1
                    else:
                        pos_count[w] += 1

    out: list[dict] = []
    for word, total in total_count.items():
        if total < min_freq:
            continue
        neg = neg_count[word]
        pos = pos_count[word]
        dominant = max(neg, pos)
        ratio = dominant / total if total else 0.0
        if ratio < min_polarity_ratio:
            continue

        polarity = "-1" if neg >= pos else "1"
        out.append({
            "embed_text": word,
            "word": word,
            "word_root": word,
            "polarity": polarity,
            "source": "kote",
            "score_hint": round(ratio, 4),
            "count_hint": total,
            "neg_count": neg,
            "pos_count": pos,
        })

    return out


def _merge_entries(knu: list[dict], kote: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}

    for row in knu:
        key = row["embed_text"]
        merged[key] = row

    for row in kote:
        key = row["embed_text"]
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
    parser.add_argument("--min-freq", type=int, default=20)
    parser.add_argument("--min-polarity-ratio", type=float, default=0.7)
    args = parser.parse_args()

    knu = _read_knu_entries()
    kote = _extract_kote_entries(
        min_freq=args.min_freq,
        min_polarity_ratio=args.min_polarity_ratio,
    )
    merged = _merge_entries(knu, kote)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"KNU entries   : {len(knu)}")
    print(f"KOTE entries  : {len(kote)}")
    print(f"Merged entries: {len(merged)}")
    print(f"Output        : {args.output}")


if __name__ == "__main__":
    main()
