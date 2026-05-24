from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEGACY_EMOTION_STOPWORDS: frozenset[str] = frozenset(
    {
        "중국",
        "미국",
        "결국",
        "과연",
        "오히려",
        "어차피",
        "부리",
    }
)

_DECISIONS_PATH = Path(__file__).with_name("emotion_stopword_review_decisions.json")


def _load_review_decisions(path: Path = _DECISIONS_PATH) -> set[str]:
    if not path.exists():
        return set()

    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    words: set[str] = set()
    for item in raw.get("block_confirmed", []):
        if isinstance(item, dict):
            surface = item.get("surface")
            if isinstance(surface, str) and surface:
                words.add(surface)
        elif isinstance(item, str) and item:
            words.add(item)

    for surface in raw.get("review_promoted_from_added_data", []):
        if isinstance(surface, str) and surface:
            words.add(surface)

    return words


EMOTION_STOPWORDS: frozenset[str] = frozenset(
    LEGACY_EMOTION_STOPWORDS | _load_review_decisions()
)


def _strip_leading_digits(value: str) -> str:
    return value.lstrip("0123456789")


def is_blocked_emotion_stopword(surface: str, form: str | None = None) -> bool:
    candidates = {surface}
    if form:
        candidates.add(form)

    candidates.update(_strip_leading_digits(value) for value in list(candidates))
    candidates.discard("")
    return any(value in EMOTION_STOPWORDS for value in candidates)
