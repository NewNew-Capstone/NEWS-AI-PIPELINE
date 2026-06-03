from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import expand_emotion_review_set_from_news_videos as expander


class FakeAnalysisService:
    def __init__(self, words_by_video: dict[str, list[str]]) -> None:
        self.words_by_video = words_by_video
        self.requests = []

    def analyze(self, request):
        self.requests.append(request)
        words = self.words_by_video.get(str(request.target_id), [])
        if not words:
            words = self.words_by_video.get(request.title, [])
        return SimpleNamespace(
            emotion_keywords=[
                SimpleNamespace(keyword_text=word, keyword_type="EMOTION", score=1.0)
                for word in words
            ],
            keywords=[],
            sentence_labels=[],
        )


def _write_decisions(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "block_confirmed": [{"surface": "그렇다", "reason": "existing"}],
                "review_promoted_from_added_data": [],
                "auto_review_blocked": [],
                "keep_confirmed": [{"surface": "분노", "reason": "emotion"}],
                "review_pending": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_auto_decision_defaults() -> None:
    assert expander.auto_decide_candidate(expander.ReviewCandidateStats("그렇다")).decision == "BLOCK"  # type: ignore[union-attr]
    assert expander.auto_decide_candidate(expander.ReviewCandidateStats("부르다")).decision == "BLOCK"  # type: ignore[union-attr]
    assert expander.auto_decide_candidate(expander.ReviewCandidateStats("통하다")).decision == "BLOCK"  # type: ignore[union-attr]
    assert expander.auto_decide_candidate(expander.ReviewCandidateStats("알리다")).decision == "BLOCK"  # type: ignore[union-attr]
    assert expander.auto_decide_candidate(expander.ReviewCandidateStats("어렵다")).decision == "REVIEW"  # type: ignore[union-attr]
    assert expander.auto_decide_candidate(expander.ReviewCandidateStats("죄송하다")).decision == "REVIEW"  # type: ignore[union-attr]

    for word in ("분노", "불안", "슬프다", "무섭다", "행복", "실망", "고맙다"):
        assert expander.auto_decide_candidate(expander.ReviewCandidateStats(word)) is None


def test_run_batch_applies_block_and_review_decisions(tmp_path: Path) -> None:
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path)
    video = expander.NewsVideoCandidate(video_id="v1", title="video-one")

    def search_fn(_keyword: str, _limit: int):
        return [video]

    service = FakeAnalysisService(
        {
            "video-one": [
                "그렇다",
                "알리다",
                "어렵다",
                "죄송하다",
                "분노",
            ]
        }
    )

    report = expander.run_review_expansion_batch(
        keywords=["대통령 논란 뉴스"],
        decisions_path=decisions_path,
        report_dir=tmp_path / "reports",
        processed_log_path=tmp_path / "processed.jsonl",
        apply=True,
        search_fn=search_fn,
        transcript_loader=lambda _video_id, _country, _priority: "뉴스 문장입니다.",
        analysis_service=service,
        now_fn=lambda: "2026-06-03T00:00:00+09:00",
    )

    assert report["decision_count"] == 3
    assert {row["keyword"] for row in report["decisions"]} == {"알리다", "어렵다", "죄송하다"}

    raw = json.loads(decisions_path.read_text(encoding="utf-8"))
    block = {row["surface"] for row in raw["block_confirmed"]}
    auto_review = {row["surface"] for row in raw["auto_review_blocked"]}

    assert "알리다" in block
    assert {"어렵다", "죄송하다"} <= auto_review
    assert "분노" not in auto_review


def test_run_batch_limits_new_decisions_to_fifty(tmp_path: Path, monkeypatch) -> None:
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path)
    words = [f"테스트{i}하다" for i in range(60)]
    video = expander.NewsVideoCandidate(video_id="v1", title="video-one")

    monkeypatch.setattr(expander, "normalize_emotion_keyword", lambda text: text)

    report = expander.run_review_expansion_batch(
        keywords=["경제 갈등 뉴스"],
        decisions_path=decisions_path,
        report_dir=tmp_path / "reports",
        processed_log_path=tmp_path / "processed.jsonl",
        max_new_candidates=50,
        apply=True,
        search_fn=lambda _keyword, _limit: [video],
        transcript_loader=lambda _video_id, _country, _priority: "뉴스 문장입니다.",
        analysis_service=FakeAnalysisService({"video-one": words}),
        now_fn=lambda: "2026-06-03T00:00:00+09:00",
    )

    assert report["candidate_count"] == 60
    assert report["decision_count"] == 50
    raw = json.loads(decisions_path.read_text(encoding="utf-8"))
    added = [row for row in raw["auto_review_blocked"] if row["surface"].startswith("테스트")]
    assert len(added) == 50


def test_processed_videos_are_skipped(tmp_path: Path) -> None:
    decisions_path = tmp_path / "decisions.json"
    _write_decisions(decisions_path)
    video = expander.NewsVideoCandidate(video_id="already", title="video-one")

    report = expander.run_review_expansion_batch(
        keywords=["사회 수사 뉴스"],
        decisions_path=decisions_path,
        report_dir=tmp_path / "reports",
        processed_log_path=tmp_path / "processed.jsonl",
        processed_video_ids={"already"},
        search_fn=lambda _keyword, _limit: [video],
        transcript_loader=lambda _video_id, _country, _priority: "뉴스 문장입니다.",
        analysis_service=FakeAnalysisService({"video-one": ["알리다"]}),
        now_fn=lambda: "2026-06-03T00:00:00+09:00",
    )

    assert report["video_count"] == 0
    assert report["decision_count"] == 0
