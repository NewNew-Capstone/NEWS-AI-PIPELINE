from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import build_emotion_seed_for_qdrant as seed_builder


def _skeleton(word: str, root: str, polarity: str) -> dict | None:
    return seed_builder.skeletonize_knu_entry(
        {"word": word, "word_root": root, "polarity": polarity}
    )


def test_skeletonize_knu_entry_removes_descriptor_nouns() -> None:
    row = _skeleton("움츠러드는 모양", "움츠러들 모양", "-1")

    assert row is not None
    assert row["embed_text"] == "움츠러들다"
    assert row["word"] == "움츠러들다"
    assert row["word_root"] == "움츠러들다"
    assert row["clean_seed"] == "움츠러들다"
    assert row["polarity"] == "-1"
    assert row["source"] == "knu"
    assert row["category"] == "NEGATIVE_STATE"
    assert row["confidence_tier"] == "HIGH"
    assert row["original_word"] == "움츠러드는 모양"
    assert row["original_word_root"] == "움츠러들 모양"


def test_skeletonize_knu_entry_keeps_core_evaluative_roots() -> None:
    assert _skeleton("괴로운 모양", "괴롭 모양", "-2")["embed_text"] == "괴롭다"  # type: ignore[index]
    assert _skeleton("거북한 모양", "거북 모양", "-2")["embed_text"] == "거북하다"  # type: ignore[index]
    assert _skeleton("가난한 사람을", "가난 사람", "-2")["embed_text"] == "가난하다"  # type: ignore[index]


def test_clean_knu_entry_outputs_requested_review_shape() -> None:
    row = seed_builder.clean_knu_entry(
        {"word": "아름다운 모양", "word_root": "아름다운 모양", "polarity": "1"}
    )

    assert row == {
        "original_word": "아름다운 모양",
        "original_word_root": "아름다운 모양",
        "clean_seed": "아름답다",
        "polarity": 1,
        "is_valid_emotion": True,
        "reject_reason": None,
        "source": "knu",
        "category": "EVALUATIVE_WORD",
    }


def test_clean_knu_entry_handles_lexicalized_emotion_predicate() -> None:
    row = seed_builder.clean_knu_entry(
        {"word": "짜증나는 사람", "word_root": "짜증나 사람", "polarity": "-1"}
    )

    assert row["clean_seed"] == "짜증나다"
    assert row["is_valid_emotion"] is True


def test_skeletonize_knu_entry_removes_noisy_nouns_from_keyword() -> None:
    rows = [
        _skeleton("움츠러드는 모양", "움츠러들 모양", "-1"),
        _skeleton("가난한 사람을", "가난 사람", "-2"),
        _skeleton("가격이 싸다", "가격 싸", "1"),
    ]

    for row in rows:
        assert row is not None
        for noisy in ("모양", "사람", "정도", "데", "것", "수"):
            assert noisy not in row["embed_text"]


def test_skeletonize_knu_entry_keeps_core_positive_evaluative_word() -> None:
    row = _skeleton("가격이 싸다", "가격 싸", "1")

    assert row is not None
    assert row["embed_text"] == "싸다"
    assert row["polarity"] == "1"
    assert row["category"] == "EVALUATIVE_WORD"


def test_skeletonize_knu_entry_drops_context_dependent_general_phrase() -> None:
    assert _skeleton("가능성이 있다고", "가능성 있", "2") is None
    assert _skeleton("가눌 수 없을 정도로", "가누 수 없", "-2") is None
    rejected = seed_builder.clean_knu_entry(
        {"word": "가능성이 있다고", "word_root": "가능성 있", "polarity": "2"}
    )
    assert rejected["is_valid_emotion"] is False
    assert rejected["clean_seed"] is None
    assert rejected["reject_reason"] == "no_emotion_candidate"


def test_skeletonize_knu_entry_drops_generic_predicate_only_skeletons() -> None:
    cases = [
        ("적합하도록 만들어지다", "적합 만들", "1"),
        ("위엄을 보이거나", "위엄 보이", "2"),
        ("수선스럽게 움직이다", "수선 움직이", "-1"),
        ("실제로 이루어지다", "실제로 이루어지", "1"),
        ("고름이 나오는", "고름 나오", "-1"),
        ("충분하게", "충분", "1"),
    ]

    for word, root, polarity in cases:
        assert _skeleton(word, root, polarity) is None


def test_skeletonize_knu_entry_drops_invalid_or_neutral_polarity() -> None:
    assert _skeleton("분노", "분노", "0") is None
    assert _skeleton("분노", "분노", "bad") is None


def test_skeletonize_knu_entry_keeps_direct_emotion_nouns() -> None:
    for word in ("분노", "불안", "걱정", "슬픔"):
        row = _skeleton(word, word, "-2")
        assert row is not None
        assert row["embed_text"] == word
        assert row["category"] == "DIRECT_EMOTION"


def test_kote_clean_token_drops_generic_verb_false_positives() -> None:
    assert seed_builder._clean_kote_token("깨진", "깨", "VV") is None
    assert seed_builder._clean_kote_token("나온", "나오", "VV") is None
    assert seed_builder._clean_kote_token("시키", "시키", "VV") is None


def test_kote_clean_token_uses_clean_seed_payload_text() -> None:
    assert seed_builder._clean_kote_token("심각한", "심각", "XR") == "심각하다"
    assert seed_builder._clean_kote_token("불안", "불안", "NNG") == "불안"


def test_read_knu_entries_uses_skeletonized_rows(tmp_path: Path, monkeypatch) -> None:
    knu_path = tmp_path / "SentiWord_info.json"
    knu_path.write_text(
        json.dumps(
            [
                {"word": "움츠러드는 모양", "word_root": "움츠러들 모양", "polarity": "-1"},
                {"word": "가능성이 있다고", "word_root": "가능성 있", "polarity": "2"},
                {"word": "분노", "word_root": "분노", "polarity": "-2"},
                {"word": "중립", "word_root": "중립", "polarity": "0"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(seed_builder, "KNU_PATH", knu_path)

    rows = seed_builder._read_knu_entries()

    assert [row["embed_text"] for row in rows] == ["움츠러들다", "분노"]
    assert all(row["word"] != "움츠러드는 모양" for row in rows)


def test_read_knu_entries_with_rejected_preserves_invalid_rows(tmp_path: Path, monkeypatch) -> None:
    knu_path = tmp_path / "SentiWord_info.json"
    knu_path.write_text(
        json.dumps(
            [
                {"word": "분노할 정도", "word_root": "분노할 정도", "polarity": "-1"},
                {"word": "적합하도록 만들어지다", "word_root": "적합 만들", "polarity": "1"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(seed_builder, "KNU_PATH", knu_path)

    rows, rejected = seed_builder._read_knu_entries_with_rejected()

    assert [row["clean_seed"] for row in rows] == ["분노"]
    assert len(rejected) == 1
    assert rejected[0]["original_word"] == "적합하도록 만들어지다"
    assert rejected[0]["is_valid_emotion"] is False


def test_merge_entries_preserves_same_seed_with_different_polarity() -> None:
    negative = {
        "embed_text": "아름답다",
        "clean_seed": "아름답다",
        "word": "아름답다",
        "word_root": "아름답다",
        "polarity": "-1",
        "source": "knu",
    }
    positive = {
        "embed_text": "아름답다",
        "clean_seed": "아름답다",
        "word": "아름답다",
        "word_root": "아름답다",
        "polarity": "1",
        "source": "kote",
        "score_hint": 0.9,
        "count_hint": 20,
    }

    rows = seed_builder._merge_entries([negative], [positive])

    assert {(row["clean_seed"], row["polarity"]) for row in rows} == {
        ("아름답다", "-1"),
        ("아름답다", "1"),
    }
