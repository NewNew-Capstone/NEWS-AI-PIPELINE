from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from analysis_bc.enums import TargetType
from analysis_bc.preprocessor import split_into_sentences
from analysis_bc.schemas import AnalyzeRequestDto
from analysis_bc.service import AnalysisService
from content_bc.modules.transcript_loader import load_transcript
from knowledge_graph_bc.neo4j_client import Neo4jClient, get_neo4j_client
from knowledge_graph_bc.schemas import (
    RealtimeClickedVideoIngestRequest,
    RealtimeClickedVideoIngestResponse,
    RealtimeIngestCandidate,
    RealtimeIngestJobResponse,
    RealtimeIngestRequest,
    RealtimeIngestResponse,
    SearchVideosResponse,
    VideoSummary,
)
from knowledge_graph_bc.service import KnowledgeGraphComparisonService, VideoNotFoundError


JOB_STATUS_QUEUED = "QUEUED"
JOB_STATUS_RUNNING = "RUNNING"
JOB_STATUS_COMPLETED = "COMPLETED"
JOB_STATUS_COMPLETED_WITH_ERRORS = "COMPLETED_WITH_ERRORS"
JOB_STATUS_FAILED = "FAILED"
RUNTIME_CLUSTER_TYPE = "SEARCH_RUNTIME"


class RealtimeIngestJobNotFoundError(LookupError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_analysis_service() -> AnalysisService:
    # Local singleton without importing analysis_bc.router and its FastAPI globals.
    if not hasattr(_get_analysis_service, "_instance"):
        _get_analysis_service._instance = AnalysisService()  # type: ignore[attr-defined]
    return _get_analysis_service._instance  # type: ignore[attr-defined]


class RealtimeIngestService:
    def __init__(
        self,
        client: Neo4jClient | None = None,
        *,
        comparison_service: KnowledgeGraphComparisonService | None = None,
        transcript_loader: Callable[[str, str, bool], str] | None = None,
        analysis_service_factory: Callable[[], Any] | None = None,
        request_id_factory: Callable[[], str] | None = None,
        now_fn: Callable[[], str] = _utc_now,
    ) -> None:
        self.client = client or get_neo4j_client()
        self.comparison_service = comparison_service or KnowledgeGraphComparisonService(client=self.client)
        self.transcript_loader = transcript_loader or load_transcript
        self.analysis_service_factory = analysis_service_factory or _get_analysis_service
        self.request_id_factory = request_id_factory or (lambda: f"rt-{uuid.uuid4().hex}")
        self.now_fn = now_fn

    def prepare_search_candidate_job(
        self,
        request: RealtimeIngestRequest,
    ) -> tuple[RealtimeIngestResponse, list[RealtimeIngestCandidate]]:
        selected = self._select_candidates(request.candidates, request.max_per_country)
        queued: list[RealtimeIngestCandidate] = []
        skipped_existing_count = 0

        for candidate in selected:
            if self._has_success_analysis(candidate.video_id):
                skipped_existing_count += 1
                continue
            queued.append(candidate)

        request_id = self.request_id_factory()
        status = JOB_STATUS_QUEUED if queued else JOB_STATUS_COMPLETED
        self._create_job(
            request_id=request_id,
            keyword=request.keyword,
            queued_count=len(queued),
            skipped_count=skipped_existing_count,
            status=status,
        )
        current_results = self.comparison_service.search_videos(
            keyword=request.keyword,
            limit=request.max_per_country,
            scope="all",
        )
        return (
            RealtimeIngestResponse(
                request_id=request_id,
                queued_count=len(queued),
                skipped_existing_count=skipped_existing_count,
                current_results=current_results,
            ),
            queued,
        )

    def prepare_clicked_video_job(
        self,
        request: RealtimeClickedVideoIngestRequest,
    ) -> tuple[RealtimeClickedVideoIngestResponse, list[RealtimeIngestCandidate]]:
        selected = request.selected_video
        related = [
            candidate
            for candidate in request.related_candidates
            if candidate.video_id != selected.video_id
        ]
        selected_candidates = [selected] + self._select_candidates(related, request.max_per_country)
        queued: list[RealtimeIngestCandidate] = []
        skipped_existing_count = 0

        for candidate in selected_candidates:
            if self._has_success_analysis(candidate.video_id):
                skipped_existing_count += 1
                continue
            queued.append(candidate)

        request_id = self.request_id_factory()
        status = JOB_STATUS_QUEUED if queued else JOB_STATUS_COMPLETED
        self._create_job(
            request_id=request_id,
            keyword=request.keyword,
            queued_count=len(queued),
            skipped_count=skipped_existing_count,
            status=status,
        )

        current_graph = None
        try:
            current_graph = self.comparison_service.get_comparison_graph(
                video_id=selected.video_id,
                limit_per_country=request.max_per_country,
                scope="all",
            )
        except VideoNotFoundError:
            current_graph = self.comparison_service.get_comparison_graph_for_source_video(
                self._candidate_to_video_summary(selected),
                keyword=request.keyword,
                limit_per_country=request.max_per_country,
                scope="all",
            )

        return (
            RealtimeClickedVideoIngestResponse(
                request_id=request_id,
                selected_video_id=selected.video_id,
                queued_count=len(queued),
                skipped_existing_count=skipped_existing_count,
                current_graph=current_graph,
            ),
            queued,
        )

    def _candidate_to_video_summary(self, candidate: RealtimeIngestCandidate) -> VideoSummary:
        return VideoSummary(
            video_id=candidate.video_id,
            target_id=self._synthetic_target_id(candidate.video_id),
            title=candidate.title,
            description=candidate.description,
            thumbnail_url=candidate.thumbnail_url,
            channel_name=candidate.channel_name,
            published_at=candidate.published_at,
            view_count=float(candidate.view_count or 0.0),
            country_code=candidate.country_code,
            language=candidate.language,
            analysis_status=None,
            cluster_types=[],
        )

    def process_search_candidates(
        self,
        *,
        request_id: str,
        keyword: str,
        candidates: list[RealtimeIngestCandidate],
    ) -> None:
        if not candidates:
            self._finish_job(request_id, status=JOB_STATUS_COMPLETED)
            return

        self._set_job_status(request_id, JOB_STATUS_RUNNING)
        local_failed_count = 0
        try:
            for candidate in candidates:
                if self._has_success_analysis(candidate.video_id):
                    self._increment_job(request_id, skipped_delta=1)
                    continue

                try:
                    transcript = self.transcript_loader(candidate.video_id, candidate.country_code, False)
                except Exception:
                    transcript = ""

                if not transcript.strip():
                    local_failed_count += 1
                    self._increment_job(request_id, failed_delta=1)
                    continue

                try:
                    analysis_result = self._analyze_candidate(candidate, transcript)
                    self._upsert_candidate_graph(keyword, candidate, transcript, analysis_result)
                    self._increment_job(request_id, success_delta=1)
                except Exception:
                    local_failed_count += 1
                    self._increment_job(request_id, failed_delta=1)

            final_status = JOB_STATUS_COMPLETED_WITH_ERRORS if local_failed_count else JOB_STATUS_COMPLETED
            self._finish_job(request_id, status=final_status)
        except Exception:
            self._finish_job(request_id, status=JOB_STATUS_FAILED)
            raise

    def get_job(self, request_id: str) -> RealtimeIngestJobResponse:
        rows = self.client.execute_read(
            """
            MATCH (j:RealtimeIngestJob {request_id: $request_id})
            RETURN properties(j) AS job
            LIMIT 1
            """,
            {"request_id": request_id},
        )
        if not rows:
            raise RealtimeIngestJobNotFoundError(f"Realtime ingest job not found: {request_id}")
        return self._row_to_job_response(rows[0].get("job") or {})

    def _select_candidates(
        self,
        candidates: list[RealtimeIngestCandidate],
        max_per_country: int,
    ) -> list[RealtimeIngestCandidate]:
        per_country: dict[str, int] = defaultdict(int)
        seen_video_ids: set[str] = set()
        selected: list[RealtimeIngestCandidate] = []

        for candidate in candidates:
            if candidate.video_id in seen_video_ids:
                continue
            if per_country[candidate.country_code] >= max_per_country:
                continue
            seen_video_ids.add(candidate.video_id)
            per_country[candidate.country_code] += 1
            selected.append(candidate)
        return selected

    def _has_success_analysis(self, video_id: str) -> bool:
        rows = self.client.execute_read(
            """
            MATCH (v:Video {video_id: $video_id})-[:HAS_ANALYSIS]->(a:AnalysisResult)
            WHERE toUpper(toString(coalesce(a.status, ""))) = "SUCCESS"
            RETURN count(a) > 0 AS exists
            LIMIT 1
            """,
            {"video_id": video_id},
        )
        return bool(rows and rows[0].get("exists"))

    def _analyze_candidate(self, candidate: RealtimeIngestCandidate, transcript: str) -> Any:
        sentences = split_into_sentences(transcript, "ko")
        if not sentences:
            raise ValueError(f"No analyzable sentences for video_id={candidate.video_id}")
        request = AnalyzeRequestDto(
            target_id=self._synthetic_target_id(candidate.video_id),
            target_type=TargetType.YOUTUBE_VIDEO,
            transcript_id=None,
            title=candidate.title,
            country=candidate.country_code,
            language="ko",
            sentences=sentences,
            priority=False,
        )
        return self.analysis_service_factory().analyze(request)

    def _upsert_candidate_graph(
        self,
        keyword: str,
        candidate: RealtimeIngestCandidate,
        transcript: str,
        analysis_result: Any,
    ) -> None:
        row = self._candidate_graph_row(keyword, candidate, transcript, analysis_result)
        self.client.execute_write(
            """
            MERGE (v:Video {video_id: $row.video_id})
            SET v.target_id = $row.target_id,
                v.target_id_source = "PYTHON_SYNTHETIC",
                v.title = $row.title,
                v.description = $row.description,
                v.country_code = $row.country_code,
                v.language = $row.language,
                v.thumbnail_url = $row.thumbnail_url,
                v.published_at = $row.published_at,
                v.view_count = $row.view_count,
                v.video_keywords = $row.analysis_keywords,
                v.transcript_status = "success",
                v.realtime_ingested_at = $row.ingested_at
            MERGE (c:Channel {channel_id: $row.channel_id})
            SET c.channel_name = $row.channel_name
            MERGE (i:Issue {issue_id: $row.issue_id})
            SET i.name = $row.keyword,
                i.keyword = $row.keyword,
                i.cluster_type = $row.cluster_type,
                i.updated_at = $row.ingested_at
            MERGE (a:AnalysisResult {analysis_id: $row.analysis_id})
            SET a.status = "SUCCESS",
                a.target_id = $row.target_id,
                a.overall_bias_score = $row.overall_bias_score,
                a.opinion_score = $row.opinion_score,
                a.emotion_score = $row.emotion_score,
                a.summary_text = $row.summary_text,
                a.score_evidence = $row.score_evidence,
                a.analysis_keywords = $row.analysis_keywords,
                a.updated_at = $row.ingested_at
            MERGE (v)-[:PUBLISHED_BY]->(c)
            MERGE (v)-[:PART_OF]->(i)
            MERGE (v)-[:HAS_ANALYSIS]->(a)
            RETURN v.video_id AS video_id
            """,
            {"row": row},
        )

    def _candidate_graph_row(
        self,
        keyword: str,
        candidate: RealtimeIngestCandidate,
        transcript: str,
        analysis_result: Any,
    ) -> dict[str, Any]:
        target_id = self._synthetic_target_id(candidate.video_id)
        analysis_keywords = self._analysis_keywords(analysis_result)
        return {
            "video_id": candidate.video_id,
            "target_id": target_id,
            "title": candidate.title,
            "description": candidate.description or transcript[:500],
            "country_code": candidate.country_code,
            "language": candidate.language,
            "thumbnail_url": candidate.thumbnail_url,
            "published_at": candidate.published_at,
            "view_count": float(candidate.view_count or 0.0),
            "channel_id": candidate.channel_id or f"runtime:{candidate.video_id}",
            "channel_name": candidate.channel_name or "UNKNOWN_CHANNEL",
            "issue_id": self._issue_id_for_keyword(keyword),
            "keyword": keyword,
            "cluster_type": RUNTIME_CLUSTER_TYPE,
            "analysis_id": f"runtime:{candidate.video_id}:{target_id}",
            "overall_bias_score": self._result_float(analysis_result, "overall_bias_score"),
            "opinion_score": self._result_float(analysis_result, "opinion_score"),
            "emotion_score": self._result_float(analysis_result, "emotion_score"),
            "summary_text": str(getattr(analysis_result, "summary_text", "") or ""),
            "score_evidence": str(getattr(analysis_result, "score_evidence", "") or ""),
            "analysis_keywords": analysis_keywords,
            "ingested_at": self.now_fn(),
        }

    def _create_job(
        self,
        *,
        request_id: str,
        keyword: str,
        queued_count: int,
        skipped_count: int,
        status: str,
    ) -> None:
        now = self.now_fn()
        self.client.execute_write(
            """
            MERGE (j:RealtimeIngestJob {request_id: $request_id})
            SET j.keyword = $keyword,
                j.status = $status,
                j.queued_count = $queued_count,
                j.success_count = 0,
                j.failed_count = 0,
                j.skipped_count = $skipped_count,
                j.created_at = coalesce(j.created_at, $now),
                j.updated_at = $now
            RETURN j.request_id AS request_id
            """,
            {
                "request_id": request_id,
                "keyword": keyword,
                "status": status,
                "queued_count": queued_count,
                "skipped_count": skipped_count,
                "now": now,
            },
        )

    def _set_job_status(self, request_id: str, status: str) -> None:
        self.client.execute_write(
            """
            MATCH (j:RealtimeIngestJob {request_id: $request_id})
            SET j.status = $status,
                j.updated_at = $now
            RETURN j.request_id AS request_id
            """,
            {"request_id": request_id, "status": status, "now": self.now_fn()},
        )

    def _increment_job(
        self,
        request_id: str,
        *,
        success_delta: int = 0,
        failed_delta: int = 0,
        skipped_delta: int = 0,
    ) -> None:
        self.client.execute_write(
            """
            MATCH (j:RealtimeIngestJob {request_id: $request_id})
            SET j.success_count = coalesce(j.success_count, 0) + $success_delta,
                j.failed_count = coalesce(j.failed_count, 0) + $failed_delta,
                j.skipped_count = coalesce(j.skipped_count, 0) + $skipped_delta,
                j.updated_at = $now
            RETURN j.request_id AS request_id
            """,
            {
                "request_id": request_id,
                "success_delta": success_delta,
                "failed_delta": failed_delta,
                "skipped_delta": skipped_delta,
                "now": self.now_fn(),
            },
        )

    def _finish_job(self, request_id: str, status: str) -> None:
        self._set_job_status(request_id, status)

    def _row_to_job_response(self, job: dict[str, Any]) -> RealtimeIngestJobResponse:
        return RealtimeIngestJobResponse(
            request_id=str(job.get("request_id") or ""),
            keyword=str(job.get("keyword") or ""),
            status=str(job.get("status") or ""),
            queued_count=int(job.get("queued_count") or 0),
            success_count=int(job.get("success_count") or 0),
            failed_count=int(job.get("failed_count") or 0),
            skipped_count=int(job.get("skipped_count") or 0),
            created_at=job.get("created_at"),
            updated_at=job.get("updated_at"),
        )

    def _analysis_keywords(self, result: Any) -> list[str]:
        keywords: list[str] = []
        for item in list(getattr(result, "keywords", []) or []) + list(getattr(result, "focus_keywords", []) or []):
            text = getattr(item, "keyword_text", None)
            if text:
                keywords.append(str(text))
        return self._dedupe(keywords)

    def _result_float(self, result: Any, attr: str) -> float:
        try:
            return float(getattr(result, attr, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _synthetic_target_id(self, video_id: str) -> int:
        digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:15]
        return int(digest, 16)

    def _issue_id_for_keyword(self, keyword: str) -> str:
        normalized = re.sub(r"[^0-9A-Za-z가-힣一-龥]+", "-", keyword.strip().lower()).strip("-")
        return f"search:{normalized[:80] or 'unknown'}"

    def _dedupe(self, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = value.strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out
