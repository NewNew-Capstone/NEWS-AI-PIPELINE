from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.enums import TargetType
from analysis_bc.keyword_extractor import KeywordExtractor
from analysis_bc.preprocessor import split_into_sentences
from analysis_bc.schemas import AnalyzeRequestDto, SentenceInputDto


SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
DEFAULT_DECISIONS_PATH = Path("analysis_bc/tagger/emotion_stopword_review_decisions.json")
DEFAULT_REPORT_DIR = Path("analysis_bc/data/eval/reports/emotion_review_batches")
DEFAULT_PROCESSED_VIDEO_LOG = DEFAULT_REPORT_DIR / "processed_news_videos.jsonl"

DEFAULT_NEWS_TOPICS: tuple[str, ...] = (
    "대통령",
    "국회",
    "검찰",
    "대법원",
    "트럼프",
    "대만",
    "북한",
    "반도체",
    "부동산",
    "의대",
    "AI",
    "환율",
    "물가",
    "재난",
)
DEFAULT_NEWS_MODIFIERS: tuple[str, ...] = (
    "논란",
    "갈등",
    "발언",
    "대응",
    "위기",
    "전망",
    "규제",
    "수사",
)

CORE_EMOTION_KEEP: frozenset[str] = frozenset(
    {
        "분노",
        "불안",
        "걱정",
        "슬픔",
        "슬프다",
        "무섭다",
        "공포",
        "두려움",
        "두렵다",
        "행복",
        "행복하다",
        "실망",
        "실망하다",
        "고맙다",
        "고마움",
        "안타깝다",
        "부끄럽다",
        "반갑다",
        "놀라다",
        "재밌다",
        "재미있다",
        "즐겁다",
        "기쁘다",
        "짜증",
        "짜증나다",
        "화나다",
        "불쾌",
        "불쾌하다",
        "불신",
        "불만",
        "충격",
        "당황",
        "당황하다",
        "서럽다",
        "귀찮다",
        "역겹다",
        "싫다",
        "안심",
        "환영",
        "감동",
        "감탄",
    }
)

GENERIC_BLOCK_PREDICATES: frozenset[str] = frozenset(
    {
        "그렇다",
        "그러다",
        "알리다",
        "밝히다",
        "전하다",
        "부르다",
        "통하다",
        "내리다",
        "꺼내다",
        "띄우다",
        "남기다",
        "던지다",
        "끼치다",
        "의하다",
        "팔리다",
        "드리다",
        "드리겠습니다",
        "하다",
        "되다",
        "있다",
        "없다",
        "보다",
        "말하다",
    }
)
GENERIC_BLOCK_SURFACES: frozenset[str] = frozenset({"그래", "다른"})
CONTEXT_REVIEW_PREDICATES: frozenset[str] = frozenset(
    {
        "어렵다",
        "죄송하다",
        "심하다",
        "괜찮다",
        "상관없다",
        "만만하다",
        "날카롭다",
        "지나치다",
        "어쩌다",
        "건드리다",
        "견디다",
        "뜨겁다",
        "따뜻하다",
        "위대하다",
        "흔들리다",
    }
)

_KEYWORD_EXTRACTOR: KeywordExtractor | None = None


@dataclass(frozen=True)
class NewsVideoCandidate:
    video_id: str
    title: str
    description: str = ""
    country_code: str = "KR"
    language: str = "ko"
    channel_id: str | None = None
    channel_name: str | None = None
    published_at: str | None = None
    thumbnail_url: str | None = None
    view_count: float = 0.0


@dataclass
class ReviewCandidateStats:
    keyword: str
    count: int = 0
    score_sum: float = 0.0
    max_score: float = 0.0
    surfaces: Counter[str] = field(default_factory=Counter)
    video_ids: set[str] = field(default_factory=set)
    search_keywords: set[str] = field(default_factory=set)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        surface: str,
        score: float,
        video: NewsVideoCandidate,
        search_keyword: str,
        sentence_text: str | None = None,
    ) -> None:
        self.count += 1
        self.score_sum += score
        self.max_score = max(self.max_score, score)
        self.video_ids.add(video.video_id)
        self.search_keywords.add(search_keyword)
        if surface:
            self.surfaces[surface] += 1
        if len(self.examples) < 5:
            self.examples.append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "search_keyword": search_keyword,
                    "surface": surface,
                    "score": round(score, 4),
                    "sentence_text": sentence_text,
                }
            )

    @property
    def video_count(self) -> int:
        return len(self.video_ids)

    @property
    def avg_score(self) -> float:
        return self.score_sum / self.count if self.count else 0.0

    def to_report(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "count": self.count,
            "video_count": self.video_count,
            "max_score": round(self.max_score, 4),
            "avg_score": round(self.avg_score, 4),
            "surfaces": dict(self.surfaces.most_common(5)),
            "search_keywords": sorted(self.search_keywords),
            "examples": self.examples,
        }


@dataclass(frozen=True)
class AutoDecision:
    keyword: str
    decision: str
    reason: str
    stats: ReviewCandidateStats

    def to_json_entry(self, batch_id: str, now: str) -> dict[str, Any]:
        return {
            "surface": self.keyword,
            "reason": self.reason,
            "batch_id": batch_id,
            "count": self.stats.count,
            "video_count": self.stats.video_count,
            "max_score": round(self.stats.max_score, 4),
            "avg_score": round(self.stats.avg_score, 4),
            "first_seen_at": now,
            "examples": self.stats.examples,
        }

    def to_report(self) -> dict[str, Any]:
        row = self.stats.to_report()
        row["decision"] = self.decision
        row["reason"] = self.reason
        return row


@dataclass(frozen=True)
class DecisionSets:
    block: frozenset[str]
    review: frozenset[str]
    keep: frozenset[str]

    @property
    def known(self) -> frozenset[str]:
        return self.block | self.review | self.keep


def _get_keyword_extractor() -> KeywordExtractor:
    global _KEYWORD_EXTRACTOR
    if _KEYWORD_EXTRACTOR is None:
        _KEYWORD_EXTRACTOR = KeywordExtractor()
    return _KEYWORD_EXTRACTOR


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _batch_id(now: str) -> str:
    safe = now.replace(":", "").replace("-", "").replace("+", "Z").split(".")[0]
    return f"emotion-review-{safe}"


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


def _load_decision_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "source_report": "",
            "source_summary": "",
            "description": "Manual and automatic review decisions for emotion ACCEPT tokens.",
            "block_confirmed": [],
            "review_promoted_from_added_data": [],
            "auto_review_blocked": [],
            "keep_confirmed": [],
            "review_pending": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def load_decision_sets(path: Path = DEFAULT_DECISIONS_PATH) -> DecisionSets:
    raw = _load_decision_json(path)

    def collect(section: str) -> frozenset[str]:
        words = {
            surface
            for item in raw.get(section, [])
            for surface in [_surface_from_decision_item(item)]
            if surface
        }
        return frozenset(words)

    block = collect("block_confirmed")
    review = collect("review_promoted_from_added_data") | collect("auto_review_blocked")
    keep = collect("keep_confirmed")
    return DecisionSets(block=block, review=review, keep=keep)


def generate_search_keywords(
    existing_issue_keywords: Iterable[str] | None = None,
    *,
    limit: int = 5,
) -> list[str]:
    topics: list[str] = []
    for keyword in existing_issue_keywords or []:
        text = keyword.strip()
        if text and text not in topics:
            topics.append(text)
    for topic in DEFAULT_NEWS_TOPICS:
        if topic not in topics:
            topics.append(topic)

    out: list[str] = []
    for topic in topics:
        for modifier in DEFAULT_NEWS_MODIFIERS:
            query = f"{topic} {modifier} 뉴스"
            if query not in out:
                out.append(query)
            if len(out) >= limit:
                return out
    return out


def youtube_search_videos(
    keyword: str,
    *,
    api_key: str,
    max_results: int = 50,
    http_client: Any = httpx,
) -> list[NewsVideoCandidate]:
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is required for YouTube search.")

    response = http_client.get(
        SEARCH_URL,
        params={
            "key": api_key,
            "q": keyword,
            "part": "snippet",
            "type": "video",
            "order": "date",
            "maxResults": max_results,
            "relevanceLanguage": "ko",
            "regionCode": "KR",
        },
        timeout=10,
    )
    response.raise_for_status()
    items = response.json().get("items", [])

    videos: list[NewsVideoCandidate] = []
    for item in items:
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if not video_id:
            continue
        videos.append(
            NewsVideoCandidate(
                video_id=str(video_id),
                title=html.unescape(str(snippet.get("title") or "")),
                description=html.unescape(str(snippet.get("description") or "")),
                country_code="KR",
                language="ko",
                channel_id=snippet.get("channelId"),
                channel_name=snippet.get("channelTitle"),
                published_at=snippet.get("publishedAt"),
                thumbnail_url=(
                    snippet.get("thumbnails", {}).get("high", {}).get("url")
                    or snippet.get("thumbnails", {}).get("default", {}).get("url")
                ),
            )
        )
    return videos


def _coerce_video(value: NewsVideoCandidate | dict[str, Any]) -> NewsVideoCandidate:
    if isinstance(value, NewsVideoCandidate):
        return value
    return NewsVideoCandidate(
        video_id=str(value["video_id"]),
        title=str(value.get("title") or ""),
        description=str(value.get("description") or ""),
        country_code=str(value.get("country_code") or "KR"),
        language=str(value.get("language") or "ko"),
        channel_id=value.get("channel_id"),
        channel_name=value.get("channel_name") or value.get("channel"),
        published_at=value.get("published_at"),
        thumbnail_url=value.get("thumbnail_url"),
        view_count=float(value.get("view_count") or 0.0),
    )


def _default_transcript_loader(video_id: str, country_code: str, priority: bool) -> str:
    from content_bc.modules.transcript_loader import load_transcript

    return load_transcript(video_id, country_code, priority)


def _default_analysis_service() -> Any:
    from analysis_bc.service import AnalysisService

    return AnalysisService()


def _synthetic_target_id(video_id: str) -> int:
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:15]
    return int(digest, 16)


def analyze_video(
    video: NewsVideoCandidate,
    transcript: str,
    *,
    analysis_service: Any,
) -> tuple[Any, list[SentenceInputDto]]:
    sentences = split_into_sentences(transcript, "ko")
    if not sentences:
        raise ValueError(f"No analyzable sentences for video_id={video.video_id}")

    request = AnalyzeRequestDto(
        target_id=_synthetic_target_id(video.video_id),
        target_type=TargetType.YOUTUBE_VIDEO,
        transcript_id=None,
        title=video.title,
        country=video.country_code,
        language="ko",
        sentences=sentences,
        priority=False,
    )
    return analysis_service.analyze(request), sentences


def normalize_emotion_keyword(text: str) -> str:
    value = text.strip()
    if not value:
        return ""
    return _get_keyword_extractor()._emotion_keyword_key(value)


def _string_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _is_emotion_keyword(item: Any) -> bool:
    return _string_value(getattr(item, "keyword_type", "")) == "EMOTION"


def _is_emotion_span(item: Any) -> bool:
    return _string_value(getattr(item, "label_type", "")) == "EMOTIONALLY_LOADED"


def _safe_score(item: Any) -> float:
    try:
        return float(getattr(item, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def collect_emotion_candidates(
    *,
    result: Any,
    video: NewsVideoCandidate,
    search_keyword: str,
    sentences: list[SentenceInputDto],
    stats_by_keyword: dict[str, ReviewCandidateStats],
) -> None:
    sentence_by_id = {sentence.content_sentence_id: sentence for sentence in sentences}
    seen_keyword_texts: set[str] = set()

    keyword_items = list(getattr(result, "emotion_keywords", []) or [])
    keyword_items.extend(
        item
        for item in list(getattr(result, "keywords", []) or [])
        if _is_emotion_keyword(item)
    )

    for item in keyword_items:
        surface = str(getattr(item, "keyword_text", "") or "").strip()
        if not surface or surface in seen_keyword_texts:
            continue
        seen_keyword_texts.add(surface)
        keyword = normalize_emotion_keyword(surface)
        if not keyword:
            continue
        stats = stats_by_keyword.setdefault(keyword, ReviewCandidateStats(keyword=keyword))
        stats.add(
            surface=surface,
            score=_safe_score(item),
            video=video,
            search_keyword=search_keyword,
        )

    for span in list(getattr(result, "sentence_labels", []) or []):
        if not _is_emotion_span(span):
            continue
        surface = str(getattr(span, "matched_word", "") or "").strip()
        if not surface:
            continue
        keyword = normalize_emotion_keyword(surface)
        if not keyword:
            continue
        sentence = sentence_by_id.get(getattr(span, "content_sentence_id", None))
        stats = stats_by_keyword.setdefault(keyword, ReviewCandidateStats(keyword=keyword))
        stats.add(
            surface=surface,
            score=_safe_score(span),
            video=video,
            search_keyword=search_keyword,
            sentence_text=sentence.sentence_text if sentence else None,
        )


def auto_decide_candidate(stats: ReviewCandidateStats) -> AutoDecision | None:
    keyword = stats.keyword.strip()
    if not keyword or keyword in CORE_EMOTION_KEEP:
        return None
    if keyword in GENERIC_BLOCK_SURFACES or keyword in GENERIC_BLOCK_PREDICATES:
        return AutoDecision(
            keyword=keyword,
            decision="BLOCK",
            reason="자동 BLOCK. 뉴스/보도 맥락의 일반어 또는 기능적 서술어로 감정 표현이 아님.",
            stats=stats,
        )
    if keyword in CONTEXT_REVIEW_PREDICATES:
        return AutoDecision(
            keyword=keyword,
            decision="REVIEW",
            reason="자동 REVIEW. 맥락 의존 평가 표현으로 감정어일 수 있으나 뉴스 분석에서는 오탐 위험이 큼.",
            stats=stats,
        )
    if keyword.endswith("다"):
        return AutoDecision(
            keyword=keyword,
            decision="REVIEW",
            reason="자동 REVIEW. 신규 서술어 감정 후보이며 정밀도 우선 정책에 따라 임시 차단.",
            stats=stats,
        )
    return AutoDecision(
        keyword=keyword,
        decision="REVIEW",
        reason="자동 REVIEW. 신규 명사 감정 후보이며 정밀도 우선 정책에 따라 임시 차단.",
        stats=stats,
    )


def choose_auto_decisions(
    stats_by_keyword: dict[str, ReviewCandidateStats],
    *,
    decision_sets: DecisionSets,
    max_new_candidates: int = 50,
) -> tuple[list[AutoDecision], list[dict[str, Any]]]:
    decisions: list[AutoDecision] = []
    skipped_protected: list[dict[str, Any]] = []
    ranked = sorted(
        stats_by_keyword.values(),
        key=lambda stats: (
            -stats.count,
            -stats.video_count,
            -stats.max_score,
            stats.keyword,
        ),
    )

    for stats in ranked:
        if stats.keyword in decision_sets.known:
            continue
        decision = auto_decide_candidate(stats)
        if decision is None:
            skipped_protected.append(stats.to_report())
            continue
        decisions.append(decision)
        if len(decisions) >= max_new_candidates:
            break
    return decisions, skipped_protected


def _known_surfaces(raw: dict[str, Any]) -> set[str]:
    sections = (
        "block_confirmed",
        "review_promoted_from_added_data",
        "auto_review_blocked",
        "keep_confirmed",
    )
    return {
        surface
        for section in sections
        for item in raw.get(section, [])
        for surface in [_surface_from_decision_item(item)]
        if surface
    }


def merge_auto_decisions(
    *,
    decisions_path: Path,
    decisions: list[AutoDecision],
    batch_id: str,
    now: str,
) -> dict[str, int]:
    raw = _load_decision_json(decisions_path)
    raw.setdefault("block_confirmed", [])
    raw.setdefault("review_promoted_from_added_data", [])
    raw.setdefault("auto_review_blocked", [])
    raw.setdefault("keep_confirmed", [])
    raw.setdefault("review_pending", [])

    known = _known_surfaces(raw)
    added_block = 0
    added_review = 0
    for decision in decisions:
        if decision.keyword in known:
            continue
        entry = decision.to_json_entry(batch_id, now)
        if decision.decision == "BLOCK":
            raw["block_confirmed"].append(entry)
            added_block += 1
        else:
            raw["auto_review_blocked"].append(entry)
            added_review += 1
        known.add(decision.keyword)

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"added_block": added_block, "added_review": added_review}


def load_processed_video_ids(path: Path = DEFAULT_PROCESSED_VIDEO_LOG) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            video_id = str(row.get("video_id") or "").strip()
            if video_id:
                out.add(video_id)
    return out


def append_processed_video_log(
    path: Path,
    *,
    batch_id: str,
    rows: list[dict[str, str]],
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"batch_id": batch_id, **row}, ensure_ascii=False) + "\n")


def _persist_runtime_graph(
    *,
    search_keyword: str,
    video: NewsVideoCandidate,
    transcript: str,
    analysis_result: Any,
) -> None:
    from knowledge_graph_bc.realtime_ingest_service import RealtimeIngestService
    from knowledge_graph_bc.schemas import RealtimeIngestCandidate

    candidate = RealtimeIngestCandidate(
        video_id=video.video_id,
        title=video.title,
        description=video.description,
        country_code=video.country_code,
        language=video.language,
        channel_id=video.channel_id,
        channel_name=video.channel_name,
        published_at=video.published_at,
        thumbnail_url=video.thumbnail_url,
        view_count=video.view_count,
    )
    RealtimeIngestService()._upsert_candidate_graph(  # noqa: SLF001
        search_keyword,
        candidate,
        transcript,
        analysis_result,
    )


def run_review_expansion_batch(
    *,
    keywords: list[str] | None = None,
    keyword_limit: int = 5,
    videos_per_keyword: int = 50,
    max_new_candidates: int = 50,
    decisions_path: Path = DEFAULT_DECISIONS_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    processed_log_path: Path = DEFAULT_PROCESSED_VIDEO_LOG,
    apply: bool = False,
    write_report: bool = True,
    search_fn: Callable[[str, int], list[NewsVideoCandidate | dict[str, Any]]] | None = None,
    transcript_loader: Callable[[str, str, bool], str] | None = None,
    analysis_service: Any | None = None,
    now_fn: Callable[[], str] = _now_iso,
    processed_video_ids: set[str] | None = None,
    api_key: str | None = None,
    persist_runtime_graph: bool = False,
) -> dict[str, Any]:
    now = now_fn()
    batch_id = _batch_id(now)
    selected_keywords = keywords or generate_search_keywords(limit=keyword_limit)
    processed = set(processed_video_ids or load_processed_video_ids(processed_log_path))
    seen_this_batch: set[str] = set()
    stats_by_keyword: dict[str, ReviewCandidateStats] = {}
    errors: list[dict[str, str]] = []
    processed_rows: list[dict[str, str]] = []
    video_count = 0
    transcript_success_count = 0
    analysis_success_count = 0

    if search_fn is None:
        resolved_api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")

        def search_fn(keyword: str, limit: int) -> list[NewsVideoCandidate]:
            return youtube_search_videos(
                keyword,
                api_key=resolved_api_key,
                max_results=limit,
            )

    loader = transcript_loader or _default_transcript_loader
    service = analysis_service or _default_analysis_service()

    for search_keyword in selected_keywords:
        try:
            videos = [_coerce_video(item) for item in search_fn(search_keyword, videos_per_keyword)]
        except Exception as exc:
            errors.append({"stage": "search", "keyword": search_keyword, "error": str(exc)})
            continue

        for video in videos:
            if video.video_id in processed or video.video_id in seen_this_batch:
                continue
            seen_this_batch.add(video.video_id)
            video_count += 1
            try:
                transcript = loader(video.video_id, video.country_code, False)
            except Exception as exc:
                errors.append({"stage": "transcript", "video_id": video.video_id, "error": str(exc)})
                continue
            if not transcript.strip():
                errors.append({"stage": "transcript", "video_id": video.video_id, "error": "empty transcript"})
                continue
            transcript_success_count += 1

            try:
                analysis_result, sentences = analyze_video(
                    video,
                    transcript,
                    analysis_service=service,
                )
            except Exception as exc:
                errors.append({"stage": "analysis", "video_id": video.video_id, "error": str(exc)})
                continue
            analysis_success_count += 1
            processed_rows.append({"video_id": video.video_id, "keyword": search_keyword})

            collect_emotion_candidates(
                result=analysis_result,
                video=video,
                search_keyword=search_keyword,
                sentences=sentences,
                stats_by_keyword=stats_by_keyword,
            )

            if persist_runtime_graph:
                try:
                    _persist_runtime_graph(
                        search_keyword=search_keyword,
                        video=video,
                        transcript=transcript,
                        analysis_result=analysis_result,
                    )
                except Exception as exc:
                    errors.append({"stage": "persist", "video_id": video.video_id, "error": str(exc)})

    decision_sets = load_decision_sets(decisions_path)
    decisions, skipped_protected = choose_auto_decisions(
        stats_by_keyword,
        decision_sets=decision_sets,
        max_new_candidates=max_new_candidates,
    )

    merge_counts = {"added_block": 0, "added_review": 0}
    if apply:
        merge_counts = merge_auto_decisions(
            decisions_path=decisions_path,
            decisions=decisions,
            batch_id=batch_id,
            now=now,
        )
        append_processed_video_log(processed_log_path, batch_id=batch_id, rows=processed_rows)

    report = {
        "batch_id": batch_id,
        "created_at": now,
        "applied": apply,
        "keywords": selected_keywords,
        "videos_per_keyword": videos_per_keyword,
        "video_count": video_count,
        "transcript_success_count": transcript_success_count,
        "analysis_success_count": analysis_success_count,
        "candidate_count": len(stats_by_keyword),
        "decision_count": len(decisions),
        "merge_counts": merge_counts,
        "decisions": [decision.to_report() for decision in decisions],
        "skipped_protected": skipped_protected,
        "errors": errors,
    }

    if write_report:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{batch_id}.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["report_path"] = str(report_path)

    return report


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    parser = argparse.ArgumentParser(
        description="Collect recent news videos and expand emotion REVIEW/BLOCK decisions."
    )
    parser.add_argument("--keyword", action="append", default=[], help="Search keyword. Can be repeated.")
    parser.add_argument("--keyword-limit", type=int, default=5)
    parser.add_argument("--videos-per-keyword", type=int, default=50)
    parser.add_argument("--max-new-candidates", type=int, default=50)
    parser.add_argument("--decisions-path", type=Path, default=DEFAULT_DECISIONS_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--processed-log-path", type=Path, default=DEFAULT_PROCESSED_VIDEO_LOG)
    parser.add_argument("--api-key", default=os.getenv("YOUTUBE_API_KEY", ""))
    parser.add_argument("--apply", action="store_true", help="Write decisions and processed video log.")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument(
        "--persist-runtime-graph",
        action="store_true",
        help="Also upsert analyzed videos into the runtime Neo4j graph.",
    )
    args = parser.parse_args()

    report = run_review_expansion_batch(
        keywords=args.keyword or None,
        keyword_limit=args.keyword_limit,
        videos_per_keyword=args.videos_per_keyword,
        max_new_candidates=args.max_new_candidates,
        decisions_path=args.decisions_path,
        report_dir=args.report_dir,
        processed_log_path=args.processed_log_path,
        apply=args.apply,
        write_report=not args.no_report,
        api_key=args.api_key,
        persist_runtime_graph=args.persist_runtime_graph,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
