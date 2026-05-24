from __future__ import annotations

from types import SimpleNamespace

import pytest

from knowledge_graph_bc.realtime_ingest_service import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_COMPLETED_WITH_ERRORS,
    RealtimeIngestJobNotFoundError,
    RealtimeIngestService,
    RUNTIME_CLUSTER_TYPE,
)
from knowledge_graph_bc.schemas import (
    ComparisonGraphResponse,
    RealtimeClickedVideoIngestRequest,
    RealtimeIngestCandidate,
    RealtimeIngestRequest,
    SearchVideosResponse,
    VideoSummary,
)
from knowledge_graph_bc.service import VideoNotFoundError


class FakeNeo4jClient:
    def __init__(self, success_existing: set[str] | None = None) -> None:
        self.success_existing = set(success_existing or set())
        self.jobs: dict[str, dict] = {}
        self.upserted_rows: list[dict] = []
        self.write_queries: list[str] = []

    def execute_read(self, query: str, parameters: dict | None = None) -> list[dict]:
        params = parameters or {}
        if "HAS_ANALYSIS" in query:
            return [{"exists": params["video_id"] in self.success_existing}]
        if "RealtimeIngestJob" in query:
            job = self.jobs.get(params["request_id"])
            return [{"job": job}] if job else []
        return []

    def execute_write(self, query: str, parameters: dict | None = None) -> list[dict]:
        self.write_queries.append(query)
        params = parameters or {}
        request_id = params.get("request_id")
        if "queued_count" in params:
            self.jobs[request_id] = {
                "request_id": request_id,
                "keyword": params["keyword"],
                "status": params["status"],
                "queued_count": params["queued_count"],
                "success_count": 0,
                "failed_count": 0,
                "skipped_count": params["skipped_count"],
                "created_at": params["now"],
                "updated_at": params["now"],
            }
        elif "success_delta" in params:
            job = self.jobs[request_id]
            job["success_count"] += params["success_delta"]
            job["failed_count"] += params["failed_delta"]
            job["skipped_count"] += params["skipped_delta"]
            job["updated_at"] = params["now"]
        elif "status" in params and request_id in self.jobs:
            self.jobs[request_id]["status"] = params["status"]
            self.jobs[request_id]["updated_at"] = params["now"]
        elif "row" in params:
            self.upserted_rows.append(params["row"])
            self.success_existing.add(params["row"]["video_id"])
        return []


class FakeComparisonService:
    def __init__(self, *, raise_not_found: bool = False) -> None:
        self.calls: list[dict] = []
        self.graph_calls: list[dict] = []
        self.fallback_graph_calls: list[dict] = []
        self.raise_not_found = raise_not_found

    def search_videos(self, *, keyword: str, limit: int, scope: str) -> SearchVideosResponse:
        self.calls.append({"keyword": keyword, "limit": limit, "scope": scope})
        return SearchVideosResponse(applied_cluster_type="ALL_AVAILABLE", keyword=keyword, sections=[])

    def get_comparison_graph(
        self,
        *,
        video_id: str,
        limit_per_country: int,
        scope: str,
    ) -> ComparisonGraphResponse:
        self.graph_calls.append(
            {"video_id": video_id, "limit_per_country": limit_per_country, "scope": scope}
        )
        if self.raise_not_found:
            raise VideoNotFoundError(video_id)
        return ComparisonGraphResponse(
            applied_cluster_type="ALL_AVAILABLE",
            source_video=VideoSummary(video_id=video_id),
            nodes=[],
            edges=[],
            country_perspectives=[],
        )

    def get_comparison_graph_for_source_video(
        self,
        source_video: VideoSummary,
        *,
        keyword: str,
        limit_per_country: int,
        scope: str,
    ) -> ComparisonGraphResponse:
        self.fallback_graph_calls.append(
            {
                "video_id": source_video.video_id,
                "keyword": keyword,
                "limit_per_country": limit_per_country,
                "scope": scope,
            }
        )
        return ComparisonGraphResponse(
            applied_cluster_type="ALL_AVAILABLE",
            source_video=source_video,
            nodes=[],
            edges=[],
            country_perspectives=[],
        )


class FakeAnalysisService:
    def __init__(self) -> None:
        self.requests = []

    def analyze(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            overall_bias_score=0.42,
            opinion_score=0.31,
            emotion_score=0.12,
            summary_text="",
            score_evidence="evidence",
            keywords=[SimpleNamespace(keyword_text="대만"), SimpleNamespace(keyword_text="트럼프")],
            focus_keywords=[SimpleNamespace(keyword_text="미중")],
        )


def _candidate(video_id: str, country: str = "KR", title: str = "트럼프 대만 뉴스") -> RealtimeIngestCandidate:
    return RealtimeIngestCandidate(
        video_id=video_id,
        title=title,
        description="대만 관련 설명",
        country_code=country,
        language="ko" if country == "KR" else "en",
        channel_id=f"ch-{video_id}",
        channel_name=f"{country} channel",
        published_at="2026-05-24T00:00:00Z",
        thumbnail_url=f"https://example.com/{video_id}.jpg",
        view_count=1000,
    )


def _service(
    client: FakeNeo4jClient | None = None,
    *,
    transcript_text: str = "트럼프가 대만 문제를 언급했다. 미중 갈등이 커지고 있다.",
    analysis_service: FakeAnalysisService | None = None,
    comparison_service: FakeComparisonService | None = None,
) -> tuple[RealtimeIngestService, FakeNeo4jClient, FakeComparisonService, FakeAnalysisService]:
    fake_client = client or FakeNeo4jClient()
    fake_comparison = comparison_service or FakeComparisonService()
    fake_analysis = analysis_service or FakeAnalysisService()
    service = RealtimeIngestService(
        client=fake_client,
        comparison_service=fake_comparison,  # type: ignore[arg-type]
        transcript_loader=lambda _video_id, _country, _priority: transcript_text,
        analysis_service_factory=lambda: fake_analysis,
        request_id_factory=lambda: "rt-test",
        now_fn=lambda: "2026-05-24T00:00:00+00:00",
    )
    return service, fake_client, fake_comparison, fake_analysis


def test_payload_normalizes_country_and_language() -> None:
    candidate = RealtimeIngestCandidate(
        video_id="v1",
        title="Title",
        country_code="kr",
        language="KO",
    )

    assert candidate.country_code == "KR"
    assert candidate.language == "ko"


def test_prepare_job_limits_per_country_and_skips_existing_success() -> None:
    service, client, comparison, _analysis = _service(FakeNeo4jClient(success_existing={"us1"}))
    request = RealtimeIngestRequest(
        keyword="트럼프 대만",
        max_per_country=1,
        candidates=[
            _candidate("kr1", "KR"),
            _candidate("kr2", "KR"),
            _candidate("us1", "US"),
        ],
    )

    response, queued = service.prepare_search_candidate_job(request)

    assert response.request_id == "rt-test"
    assert response.queued_count == 1
    assert response.skipped_existing_count == 1
    assert [candidate.video_id for candidate in queued] == ["kr1"]
    assert client.jobs["rt-test"]["queued_count"] == 1
    assert client.jobs["rt-test"]["skipped_count"] == 1
    assert comparison.calls == [{"keyword": "트럼프 대만", "limit": 1, "scope": "all"}]


def test_prepare_clicked_video_job_queues_selected_and_related_candidates() -> None:
    service, client, comparison, _analysis = _service(FakeNeo4jClient(success_existing={"us1"}))
    request = RealtimeClickedVideoIngestRequest(
        keyword="트럼프 대만",
        max_per_country=1,
        selected_video=_candidate("kr-selected", "KR"),
        related_candidates=[
            _candidate("kr-selected", "KR"),
            _candidate("us1", "US"),
            _candidate("us2", "US"),
            _candidate("cn1", "CN"),
            _candidate("cn2", "CN"),
        ],
    )

    response, queued = service.prepare_clicked_video_job(request)

    assert response.request_id == "rt-test"
    assert response.selected_video_id == "kr-selected"
    assert response.queued_count == 2
    assert response.skipped_existing_count == 1
    assert response.current_graph is not None
    assert [candidate.video_id for candidate in queued] == ["kr-selected", "cn1"]
    assert client.jobs["rt-test"]["queued_count"] == 2
    assert comparison.graph_calls == [
        {"video_id": "kr-selected", "limit_per_country": 1, "scope": "all"}
    ]


def test_prepare_clicked_video_job_falls_back_to_selected_metadata_when_source_is_missing() -> None:
    comparison = FakeComparisonService(raise_not_found=True)
    service, _client, comparison, _analysis = _service(comparison_service=comparison)
    request = RealtimeClickedVideoIngestRequest(
        keyword="트럼프 대만",
        max_per_country=3,
        selected_video=_candidate("kr-selected", "KR"),
        related_candidates=[],
    )

    response, queued = service.prepare_clicked_video_job(request)

    assert response.current_graph is not None
    assert response.current_graph.source_video.video_id == "kr-selected"
    assert [candidate.video_id for candidate in queued] == ["kr-selected"]
    assert comparison.fallback_graph_calls == [
        {
            "video_id": "kr-selected",
            "keyword": "트럼프 대만",
            "limit_per_country": 3,
            "scope": "all",
        }
    ]


def test_process_successful_candidate_upserts_graph_and_updates_job() -> None:
    service, client, _comparison, analysis = _service()
    request = RealtimeIngestRequest(keyword="트럼프 대만", candidates=[_candidate("kr1", "KR")])
    response, queued = service.prepare_search_candidate_job(request)

    service.process_search_candidates(
        request_id=response.request_id,
        keyword=request.keyword,
        candidates=queued,
    )

    assert len(client.upserted_rows) == 1
    row = client.upserted_rows[0]
    assert row["video_id"] == "kr1"
    assert row["cluster_type"] == RUNTIME_CLUSTER_TYPE
    assert row["issue_id"] == "search:트럼프-대만"
    assert row["analysis_keywords"] == ["대만", "트럼프", "미중"]
    assert client.jobs["rt-test"]["status"] == JOB_STATUS_COMPLETED
    assert client.jobs["rt-test"]["success_count"] == 1
    assert analysis.requests[0].language == "ko"
    assert any("MERGE (v)-[:HAS_ANALYSIS]->(a)" in query for query in client.write_queries)


def test_process_empty_transcript_records_failure_without_video_upsert() -> None:
    service, client, _comparison, _analysis = _service(transcript_text="")
    request = RealtimeIngestRequest(keyword="트럼프 대만", candidates=[_candidate("kr1", "KR")])
    response, queued = service.prepare_search_candidate_job(request)

    service.process_search_candidates(
        request_id=response.request_id,
        keyword=request.keyword,
        candidates=queued,
    )

    assert client.upserted_rows == []
    assert client.jobs["rt-test"]["failed_count"] == 1
    assert client.jobs["rt-test"]["status"] == JOB_STATUS_COMPLETED_WITH_ERRORS


def test_get_job_returns_persisted_job_status() -> None:
    service, _client, _comparison, _analysis = _service()
    request = RealtimeIngestRequest(keyword="트럼프 대만", candidates=[_candidate("kr1", "KR")])
    response, _queued = service.prepare_search_candidate_job(request)

    job = service.get_job(response.request_id)

    assert job.request_id == "rt-test"
    assert job.keyword == "트럼프 대만"
    assert job.queued_count == 1


def test_get_job_raises_for_missing_request_id() -> None:
    service, _client, _comparison, _analysis = _service()

    with pytest.raises(RealtimeIngestJobNotFoundError):
        service.get_job("missing")
