from __future__ import annotations

from knowledge_graph_bc.schemas import VideoSummary
from knowledge_graph_bc.service import KnowledgeGraphComparisonService


def _row(
    *,
    video_id: str,
    title: str,
    country_code: str,
    language: str,
    keywords: list[str],
    issue_id: str = "issue-semiconductor",
    issue_name: str = "반도체 공급망",
    entities: list[str] | None = None,
    include_analysis: bool = True,
    opinion_score: float | None = 0.5,
    cluster_type: str | None = "CURATION_MANUAL",
) -> dict:
    row = {
        "video": {
            "video_id": video_id,
            "target_id": int("".join(ch for ch in video_id if ch.isdigit()) or 1),
            "title": title,
            "description": title,
            "thumbnail_url": f"https://example.com/{video_id}.jpg",
            "channel_name": f"{country_code} channel",
            "published_at": "2026-05-13T00:00:00Z",
            "view_count": 1000,
            "country_code": country_code,
            "language": language,
            "video_keywords": keywords,
        },
        "channel": {"channel_name": f"{country_code} channel"},
        "issue_props": [{"issue_id": issue_id, "name": issue_name, "cluster_type": cluster_type}],
        "entity_props": [{"entity_key": entity} for entity in entities or []],
    }
    if include_analysis:
        row["analysis"] = {
            "status": "SUCCESS",
            "analysis_keywords": keywords,
            "opinion_score": opinion_score,
        }
    else:
        row["analysis"] = {}
    return row


class FakeNeo4jClient:
    def __init__(self, rows: list[dict], *, enable_direct_issue_lookup: bool = True) -> None:
        self.rows = rows
        self.enable_direct_issue_lookup = enable_direct_issue_lookup
        self.queries: list[str] = []

    def execute_read(self, query: str, parameters: dict | None = None) -> list[dict]:
        self.queries.append(query)
        params = parameters or {}
        cluster_type_filter = params.get("cluster_type")

        def _match_cluster_type(row: dict) -> bool:
            if not cluster_type_filter:
                return True
            for issue in row.get("issue_props", []):
                if not isinstance(issue, dict):
                    continue
                if str(issue.get("cluster_type") or "").upper() == str(cluster_type_filter).upper():
                    return True
            return False

        if "source_video_id" in params:
            if not self.enable_direct_issue_lookup:
                return []
            source_rows = [
                row for row in self.rows if row["video"]["video_id"] == params["source_video_id"]
            ]
            if not source_rows:
                return []
            source_issue_ids = {
                issue.get("issue_id")
                for issue in source_rows[0].get("issue_props", [])
                if issue.get("issue_id")
            }
            rows = [
                row
                for row in self.rows
                if row["video"]["video_id"] != params["source_video_id"]
                and row["video"]["country_code"] == params["country"]
                and row["video"]["language"] == params["language"]
                and _match_cluster_type(row)
                and bool(source_issue_ids & {
                    issue.get("issue_id")
                    for issue in row.get("issue_props", [])
                    if issue.get("issue_id")
                })
            ]
            return rows[: params.get("limit", len(rows))]
        if "video_id" in params:
            return [row for row in self.rows if row["video"]["video_id"] == params["video_id"] and _match_cluster_type(row)]
        if "country" in params and "language" in params:
            rows = [
                row
                for row in self.rows
                if row["video"]["country_code"] == params["country"]
                and row["video"]["language"] == params["language"]
                and _match_cluster_type(row)
            ]
            return rows[: params.get("limit", len(rows))]
        rows = [row for row in self.rows if _match_cluster_type(row)]
        return rows[: params.get("limit", len(rows))]


def _service() -> KnowledgeGraphComparisonService:
    return KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr001",
                    title="삼성 반도체 수출 규제",
                    country_code="KR",
                    language="ko",
                    keywords=["반도체", "수출 규제", "삼성전자"],
                    entities=["삼성전자"],
                ),
                _row(
                    video_id="us001",
                    title="Chip export controls and national security",
                    country_code="US",
                    language="en",
                    keywords=["반도체", "수출 규제", "national security"],
                    entities=["삼성전자"],
                ),
                _row(
                    video_id="cn001",
                    title="半导体 制裁 与 技术 自立",
                    country_code="CN",
                    language="zh",
                    keywords=["반도체", "제재", "기술 자립"],
                ),
                _row(
                    video_id="jp001",
                    title="Japan unrelated video",
                    country_code="JP",
                    language="ja",
                    keywords=["반도체"],
                ),
            ]
        )
    )


def test_comparison_home_returns_three_country_sections() -> None:
    response = _service().get_comparison_home(limit=5)

    assert [section.country_code for section in response.sections] == ["KR", "US", "CN"]
    assert [section.language for section in response.sections] == ["ko", "en", "zh"]
    assert response.applied_cluster_type == "CURATION_MANUAL"
    assert response.issue_keywords


def test_search_videos_applies_country_and_language_targets() -> None:
    response = _service().search_videos(keyword="반도체", limit=5, scope="curation")

    assert {section.country_code for section in response.sections} == {"KR", "US", "CN"}
    assert response.applied_cluster_type == "CURATION_MANUAL"
    for section in response.sections:
        assert all(video.country_code == section.country_code for video in section.videos)
        assert all(video.language == section.language for video in section.videos)


def test_search_videos_all_scope_includes_non_curated_rows() -> None:
    service = KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr-auto",
                    title="트럼프 대만 정책",
                    country_code="KR",
                    language="ko",
                    keywords=["트럼프", "대만"],
                    cluster_type=None,
                ),
                _row(
                    video_id="us-auto",
                    title="Trump Taiwan policy",
                    country_code="US",
                    language="en",
                    keywords=["trump", "taiwan"],
                    cluster_type=None,
                ),
            ]
        )
    )
    service._expand_match_keywords = lambda keywords: keywords + ["trump", "taiwan"]  # type: ignore[method-assign]

    all_response = service.search_videos(keyword="트럼프 대만", limit=5)
    curation_response = service.search_videos(keyword="트럼프 대만", limit=5, scope="curation")

    assert all_response.applied_cluster_type == "ALL_AVAILABLE"
    assert any(video.video_id == "us-auto" for section in all_response.sections for video in section.videos)
    assert all(
        video.video_id != "us-auto"
        for section in curation_response.sections
        for video in section.videos
    )


def test_comparison_graph_contains_source_and_only_other_target_countries() -> None:
    response = _service().get_comparison_graph(video_id="kr001", limit_per_country=5, scope="curation")

    source_nodes = [node for node in response.nodes if node.node_type == "source"]
    related_nodes = [node for node in response.nodes if node.node_type == "related"]

    assert len(source_nodes) == 1
    assert source_nodes[0].video_id == "kr001"
    assert response.applied_cluster_type == "CURATION_MANUAL"
    assert {node.country_code for node in related_nodes} <= {"US", "CN"}
    assert {node.country_code for node in related_nodes} == {"US", "CN"}


def test_comparison_graph_edges_include_reasons_or_keywords() -> None:
    response = _service().get_comparison_graph(video_id="kr001", limit_per_country=5, scope="curation")

    assert response.edges
    assert all(edge.reasons or edge.keywords for edge in response.edges)


def test_queries_accept_spring_issue_labels() -> None:
    client = FakeNeo4jClient([])
    service = KnowledgeGraphComparisonService(client=client)

    service.get_comparison_home(limit=5)
    service.search_videos(keyword="반도체", limit=5, scope="curation")

    joined_queries = "\n".join(client.queries)
    assert "[:PART_OF]->(i:Issue)" in joined_queries
    assert "[:PUBLISHED_BY]->(c:Channel)" in joined_queries
    assert "(ic:IssueCluster)" in joined_queries
    assert "(i:Issue)" in joined_queries
    assert "(inode:IssueNode)" in joined_queries
    assert "cluster_type" in joined_queries


def test_comparison_graph_uses_direct_same_issue_traversal_first() -> None:
    client = FakeNeo4jClient(
        [
            _row(
                video_id="kr001",
                title="한국 반도체",
                country_code="KR",
                language="ko",
                keywords=["반도체"],
                issue_id="issue-1",
            ),
            _row(
                video_id="us001",
                title="US chip issue",
                country_code="US",
                language="en",
                keywords=["chip"],
                issue_id="issue-1",
            ),
        ]
    )
    service = KnowledgeGraphComparisonService(client=client)

    response = service.get_comparison_graph(video_id="kr001", limit_per_country=5, scope="curation")

    assert any(node.video_id == "us001" for node in response.nodes)
    assert any(edge.relation_type == "SAME_ISSUE" for edge in response.edges)
    assert any("Issue)<-[:PART_OF]-(related:Video)" in query for query in client.queries)


def test_comparison_graph_does_not_require_analysis_result() -> None:
    service = KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr001",
                    title="한국 반도체",
                    country_code="KR",
                    language="ko",
                    keywords=["반도체"],
                    issue_id="issue-1",
                    include_analysis=False,
                ),
                _row(
                    video_id="us001",
                    title="US chip issue",
                    country_code="US",
                    language="en",
                    keywords=["chip"],
                    issue_id="issue-1",
                    include_analysis=False,
                ),
            ]
        )
    )

    response = service.get_comparison_graph(video_id="kr001", limit_per_country=5, scope="curation")

    assert response.source_video.video_id == "kr001"
    assert any(node.video_id == "us001" for node in response.nodes)
    assert all(node.analysis_status is None for node in response.nodes)


def test_comparison_graph_falls_back_to_keyword_matching_without_same_issue() -> None:
    service = KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr001",
                    title="한국 반도체",
                    country_code="KR",
                    language="ko",
                    keywords=["반도체", "공급망"],
                    issue_id="issue-1",
                ),
                _row(
                    video_id="us001",
                    title="US semiconductor supply chain",
                    country_code="US",
                    language="en",
                    keywords=["반도체", "supply chain"],
                    issue_id="issue-2",
                ),
            ],
            enable_direct_issue_lookup=False,
        )
    )

    response = service.get_comparison_graph(video_id="kr001", limit_per_country=5, scope="curation")

    assert any(node.video_id == "us001" for node in response.nodes)
    assert any(edge.relation_type == "SHARED_KEYWORD" for edge in response.edges)


def test_comparison_graph_all_scope_links_non_curated_other_country_candidates() -> None:
    service = KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr-auto",
                    title="트럼프 대만 긴장 고조",
                    country_code="KR",
                    language="ko",
                    keywords=["트럼프", "대만"],
                    issue_id="issue-kr",
                    cluster_type=None,
                ),
                _row(
                    video_id="us-auto",
                    title="Trump Taiwan tensions rise",
                    country_code="US",
                    language="en",
                    keywords=["trump", "taiwan"],
                    issue_id="issue-us",
                    cluster_type=None,
                ),
                _row(
                    video_id="cn-auto",
                    title="特朗普 台湾 局势",
                    country_code="CN",
                    language="zh",
                    keywords=["特朗普", "台湾"],
                    issue_id="issue-cn",
                    cluster_type=None,
                ),
            ],
            enable_direct_issue_lookup=False,
        ),
        semantic_similarity_fn=lambda _source, _candidate: 0.7,
    )
    service._expand_match_keywords = lambda keywords: keywords + ["trump", "taiwan", "特朗普", "台湾"]  # type: ignore[method-assign]

    response = service.get_comparison_graph(video_id="kr-auto", limit_per_country=5)
    related_nodes = [node for node in response.nodes if node.node_type == "related"]

    assert response.applied_cluster_type == "ALL_AVAILABLE"
    assert {node.country_code for node in related_nodes} == {"US", "CN"}
    assert all(node.country_code != "KR" for node in related_nodes)
    assert all(edge.reasons for edge in response.edges)


def test_comparison_graph_for_source_video_uses_selected_metadata_without_source_node() -> None:
    service = KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr-other",
                    title="다른 한국 트럼프 대만 영상",
                    country_code="KR",
                    language="ko",
                    keywords=["트럼프", "대만"],
                    issue_id="issue-kr",
                    cluster_type=None,
                ),
                _row(
                    video_id="us-auto",
                    title="Trump Taiwan tensions rise",
                    country_code="US",
                    language="en",
                    keywords=["trump", "taiwan"],
                    issue_id="issue-us",
                    cluster_type=None,
                ),
                _row(
                    video_id="cn-auto",
                    title="特朗普 台湾 局势",
                    country_code="CN",
                    language="zh",
                    keywords=["特朗普", "台湾"],
                    issue_id="issue-cn",
                    cluster_type=None,
                ),
            ]
        ),
        semantic_similarity_fn=lambda _source, _candidate: 0.7,
    )
    service._expand_match_keywords = lambda keywords: keywords + ["trump", "taiwan", "特朗普", "台湾"]  # type: ignore[method-assign]

    response = service.get_comparison_graph_for_source_video(
        VideoSummary(
            video_id="selected-kr",
            title="트럼프 대만 중국 관련 한국 뉴스",
            description="트럼프 대만 중국 관련 뉴스",
            country_code="KR",
            language="ko",
        ),
        keyword="트럼프 대만",
        limit_per_country=3,
    )

    related_nodes = [node for node in response.nodes if node.node_type == "related"]
    assert response.source_video.video_id == "selected-kr"
    assert {node.country_code for node in related_nodes} == {"US", "CN"}
    assert all(node.country_code != "KR" for node in related_nodes)
    assert all(edge.reasons for edge in response.edges)


def test_comparison_graph_all_scope_uses_semantic_similarity_for_ordering() -> None:
    def semantic_similarity(_source: str, candidate: str) -> float:
        return 0.9 if "high" in candidate else 0.4

    service = KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr001",
                    title="한국 원문",
                    country_code="KR",
                    language="ko",
                    keywords=["동아시아"],
                    issue_id="issue-1",
                    cluster_type=None,
                ),
                _row(
                    video_id="us-low",
                    title="low semantic candidate",
                    country_code="US",
                    language="en",
                    keywords=["policy"],
                    issue_id="issue-2",
                    cluster_type=None,
                ),
                _row(
                    video_id="us-high",
                    title="high semantic candidate",
                    country_code="US",
                    language="en",
                    keywords=["policy"],
                    issue_id="issue-3",
                    cluster_type=None,
                ),
            ],
            enable_direct_issue_lookup=False,
        ),
        semantic_similarity_fn=semantic_similarity,
    )
    service._expand_match_keywords = lambda keywords: keywords  # type: ignore[method-assign]

    response = service.get_comparison_graph(video_id="kr001", limit_per_country=2)
    us_edges = [edge for edge in response.edges if edge.target in {"video:us-low", "video:us-high"}]

    assert [edge.target for edge in us_edges] == ["video:us-high", "video:us-low"]
    assert all(edge.relation_type == "SEMANTIC_SIMILARITY" for edge in us_edges)


def test_issue_and_entity_extractors_ignore_empty_optional_matches() -> None:
    service = KnowledgeGraphComparisonService(client=FakeNeo4jClient([]))
    row = {
        "video": {"video_id": "kr001", "title": "반도체 공급망"},
        "issue_props": [None, {}, {"issue_id": "issue-1", "name": "반도체"}],
        "entity_props": [None, {}, {"entity_key": "삼성전자"}],
    }

    assert service._extract_issue_ids(row) == ["issue-1"]
    assert service._extract_issue_names(row) == ["반도체"]
    assert service._extract_entity_keys(row) == ["삼성전자"]


def test_comparison_graph_prioritizes_smaller_opinion_distance() -> None:
    service = KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr001",
                    title="한국 반도체",
                    country_code="KR",
                    language="ko",
                    keywords=["반도체", "공급망"],
                    issue_id="issue-1",
                    opinion_score=0.6,
                ),
                _row(
                    video_id="us-near",
                    title="US near opinion",
                    country_code="US",
                    language="en",
                    keywords=["반도체", "supply chain"],
                    issue_id="issue-1",
                    opinion_score=0.58,
                ),
                _row(
                    video_id="us-far",
                    title="US far opinion",
                    country_code="US",
                    language="en",
                    keywords=["반도체", "supply chain"],
                    issue_id="issue-1",
                    opinion_score=0.2,
                ),
            ]
        )
    )

    response = service.get_comparison_graph(video_id="kr001", limit_per_country=2, scope="curation")

    us_edges = [edge for edge in response.edges if edge.target in {"video:us-near", "video:us-far"}]
    assert len(us_edges) == 2
    assert us_edges[0].target == "video:us-near"
    assert us_edges[0].opinion_distance is not None
    assert us_edges[0].similarity_score is not None
    assert us_edges[0].opinion_distance <= us_edges[1].opinion_distance


def test_comparison_graph_fallback_when_opinion_score_missing() -> None:
    service = KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr001",
                    title="한국 반도체",
                    country_code="KR",
                    language="ko",
                    keywords=["반도체"],
                    issue_id="issue-1",
                    opinion_score=None,
                ),
                _row(
                    video_id="us001",
                    title="US chip issue",
                    country_code="US",
                    language="en",
                    keywords=["반도체"],
                    issue_id="issue-1",
                    opinion_score=0.4,
                ),
            ]
        )
    )

    response = service.get_comparison_graph(video_id="kr001", limit_per_country=5, scope="curation")

    edge = next(edge for edge in response.edges if edge.target == "video:us001")
    assert edge.opinion_distance is None
    assert edge.similarity_score is None
    assert any("opinion_score 없음으로 내용 유사도 기반 fallback" in reason for reason in edge.reasons)


def test_comparison_graph_skips_candidates_missing_country_or_language() -> None:
    service = KnowledgeGraphComparisonService(
        client=FakeNeo4jClient(
            [
                _row(
                    video_id="kr001",
                    title="한국 반도체",
                    country_code="KR",
                    language="ko",
                    keywords=["반도체"],
                    issue_id="issue-1",
                    opinion_score=0.5,
                ),
                _row(
                    video_id="us-null-country",
                    title="US missing country",
                    country_code="",
                    language="en",
                    keywords=["반도체"],
                    issue_id="issue-1",
                    opinion_score=0.45,
                ),
                _row(
                    video_id="us-null-lang",
                    title="US missing language",
                    country_code="US",
                    language="",
                    keywords=["반도체"],
                    issue_id="issue-1",
                    opinion_score=0.45,
                ),
            ]
        )
    )

    response = service.get_comparison_graph(video_id="kr001", limit_per_country=5, scope="curation")

    assert all(node.video_id != "us-null-country" for node in response.nodes)
    assert all(node.video_id != "us-null-lang" for node in response.nodes)
