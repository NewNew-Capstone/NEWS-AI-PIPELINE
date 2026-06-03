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


def _load_review_decisions(path: Path = _DECISIONS_PATH) -> set[str]:
    if not path.exists():
        return set()

    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

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
        if not form.endswith("다"):
            candidates.add(f"{form}다")
            candidates.add(f"{form}하다")

    candidates.update(_strip_leading_digits(value) for value in list(candidates))
    candidates.discard("")
    return any(value in EMOTION_STOPWORDS for value in candidates)
