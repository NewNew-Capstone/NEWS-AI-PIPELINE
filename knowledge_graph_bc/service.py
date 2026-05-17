from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from knowledge_graph_bc.neo4j_client import Neo4jClient, get_neo4j_client
from knowledge_graph_bc.schemas import (
    ComparisonGraphResponse,
    ComparisonHomeResponse,
    CountryPerspective,
    GraphEdge,
    GraphNode,
    SearchVideosResponse,
    VideoSection,
    VideoSummary,
)


COUNTRY_LANGUAGE_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("KR", "ko", "한국"),
    ("US", "en", "미국"),
    ("CN", "zh", "중국"),
)

FALLBACK_ISSUE_KEYWORDS = ["반도체", "AI", "전기차", "대만", "기후위기"]
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "영상",
    "뉴스",
    "관련",
    "오늘",
    "이번",
    "대한",
    "있는",
    "없는",
}

logger = logging.getLogger(__name__)

CLUSTER_TYPE_SEARCH_AUTO = "SEARCH_AUTO"
CLUSTER_TYPE_CURATION_MANUAL = "CURATION_MANUAL"


class VideoNotFoundError(LookupError):
    pass


class KnowledgeGraphComparisonService:
    def __init__(self, client: Neo4jClient | None = None) -> None:
        self.client = client or get_neo4j_client()

    def get_comparison_home(self, limit: int = 5) -> ComparisonHomeResponse:
        safe_limit = self._safe_limit(limit)
        rows = self._fetch_recent_video_context(limit=500)
        issue_keywords = self._rank_issue_keywords(rows, max_count=5)
        sections = [
            self._build_section(country, language, label, safe_limit)
            for country, language, label in COUNTRY_LANGUAGE_TARGETS
        ]
        return ComparisonHomeResponse(
            applied_cluster_type=CLUSTER_TYPE_CURATION_MANUAL,
            issue_keywords=issue_keywords,
            sections=sections,
        )

    def search_videos(self, keyword: str, limit: int = 5) -> SearchVideosResponse:
        keyword = keyword.strip()
        safe_limit = self._safe_limit(limit)
        sections: list[VideoSection] = []

        for country, language, label in COUNTRY_LANGUAGE_TARGETS:
            rows = self._fetch_videos(country=country, language=language, limit=max(safe_limit * 20, 50))
            ranked = self._rank_search_rows(rows, keyword)[:safe_limit]
            sections.append(
                VideoSection(
                    country_code=country,
                    language=language,
                    label=label,
                    videos=[self._row_to_summary(row) for row, _score in ranked],
                )
            )

        return SearchVideosResponse(
            applied_cluster_type=CLUSTER_TYPE_CURATION_MANUAL,
            keyword=keyword,
            sections=sections,
        )

    def get_comparison_graph(
        self,
        video_id: str,
        limit_per_country: int = 5,
    ) -> ComparisonGraphResponse:
        safe_limit = self._safe_limit(limit_per_country)
        source_row = self._fetch_video_by_id(video_id)
        if source_row is None:
            raise VideoNotFoundError(f"Video not found: {video_id}")

        source_summary = self._row_to_summary(source_row)
        source_keywords = self._extract_keywords(source_row, max_count=8)
        source_issue_ids = set(self._extract_issue_ids(source_row))
        source_entities = set(self._extract_entity_keys(source_row))
        source_country = source_summary.country_code
        source_language = source_summary.language
        source_opinion_score = self._extract_opinion_score(source_row)

        nodes = [self._summary_to_node(source_summary, node_type="source")]
        edges: list[GraphEdge] = []
        perspectives: list[CountryPerspective] = []
        degrade_reasons: Counter[str] = Counter()

        for country, language, _label in COUNTRY_LANGUAGE_TARGETS:
            if country == source_country and language == source_language:
                continue

            same_issue_rows = self._fetch_related_videos_by_issue(
                source_video_id=source_summary.video_id,
                country=country,
                language=language,
                limit=safe_limit,
            )
            ranked = self._rank_related_rows(
                same_issue_rows,
                source_video_id=source_summary.video_id,
                source_keywords=source_keywords,
                source_issue_ids=source_issue_ids,
                source_entities=source_entities,
                source_opinion_score=source_opinion_score,
                degrade_reasons=degrade_reasons,
            )

            seen_video_ids = {
                self._row_to_summary(row).video_id
                for row, _score, _shared_keywords, _reasons, _similarity_score, _opinion_distance in ranked
            }
            seen_video_ids.add(source_summary.video_id)

            if len(ranked) < safe_limit:
                candidates = self._fetch_videos(country=country, language=language, limit=300)
                fallback_ranked = self._rank_related_rows(
                    candidates,
                    source_video_id=source_summary.video_id,
                    source_keywords=source_keywords,
                    source_issue_ids=source_issue_ids,
                    source_entities=source_entities,
                    source_opinion_score=source_opinion_score,
                    degrade_reasons=degrade_reasons,
                )
                for item in fallback_ranked:
                    fallback_summary = self._row_to_summary(item[0])
                    if fallback_summary.video_id in seen_video_ids:
                        continue
                    ranked.append(item)
                    seen_video_ids.add(fallback_summary.video_id)
                    if len(ranked) >= safe_limit:
                        break

            ranked = ranked[:safe_limit]

            related_rows = [
                row
                for row, _score, _shared_keywords, _reasons, _similarity_score, _opinion_distance in ranked
            ]
            country_keywords = self._top_keywords_for_rows(related_rows, max_count=5)
            perspectives.append(
                CountryPerspective(
                    country_code=country,
                    language=language,
                    summary=self._build_perspective_summary(country, country_keywords),
                    top_keywords=country_keywords,
                )
            )

            for row, score, shared_keywords, reasons, similarity_score, opinion_distance in ranked:
                summary = self._row_to_summary(row)
                nodes.append(self._summary_to_node(summary, node_type="related"))
                if not shared_keywords and not reasons:
                    shared_keywords = source_keywords[:1]
                    reasons = ["선택 영상의 핵심 키워드와 관련된 후보 영상입니다."]
                edges.append(
                    GraphEdge(
                        source=self._node_id(source_summary.video_id),
                        target=self._node_id(summary.video_id),
                        relation_type=self._relation_type(reasons),
                        weight=round(score, 4),
                        similarity_score=self._rounded_or_none(similarity_score),
                        opinion_distance=self._rounded_or_none(opinion_distance),
                        keywords=shared_keywords[:5],
                        reasons=reasons,
                    )
                )

        if source_country is None:
            degrade_reasons["source_missing_country_code"] += 1
        if source_language is None:
            degrade_reasons["source_missing_language"] += 1
        if source_opinion_score is None:
            degrade_reasons["source_missing_opinion_score"] += 1
        if not edges:
            degrade_reasons["candidate_shortage"] += 1
            self._log_graph_degrade(
                source_video_id=source_summary.video_id,
                source_country=source_country,
                source_language=source_language,
                source_has_analysis=source_opinion_score is not None,
                reasons=degrade_reasons,
            )

        return ComparisonGraphResponse(
            applied_cluster_type=CLUSTER_TYPE_CURATION_MANUAL,
            source_video=source_summary,
            core_keywords=source_keywords,
            nodes=nodes,
            edges=edges,
            country_perspectives=perspectives,
        )

    def _build_section(
        self,
        country: str,
        language: str,
        label: str,
        limit: int,
    ) -> VideoSection:
        rows = self._fetch_videos(country=country, language=language, limit=limit)
        return VideoSection(
            country_code=country,
            language=language,
            label=label,
            videos=[self._row_to_summary(row) for row in rows[:limit]],
        )

    def _fetch_recent_video_context(self, limit: int) -> list[dict[str, Any]]:
        return self.client.execute_read(
            """
            MATCH (v:Video)
            OPTIONAL MATCH (v)-[:PART_OF]->(i:Issue)
            OPTIONAL MATCH (v)-[]-(legacy_i:Issue)
            OPTIONAL MATCH (v)-[]-(ic:IssueCluster)
            OPTIONAL MATCH (v)-[]-(inode:IssueNode)
            OPTIONAL MATCH (v)-[]-(a:AnalysisResult)
            RETURN properties(v) AS video,
                   [x IN (
                     collect(DISTINCT CASE
                       WHEN i IS NOT NULL AND toUpper(toString(coalesce(i.cluster_type, i.clusterType, ""))) = $cluster_type
                       THEN properties(i) + {cluster_type: toUpper(toString(coalesce(i.cluster_type, i.clusterType)))}
                     END) +
                     collect(DISTINCT CASE
                       WHEN legacy_i IS NOT NULL AND toUpper(toString(coalesce(legacy_i.cluster_type, legacy_i.clusterType, ""))) = $cluster_type
                       THEN properties(legacy_i) + {cluster_type: toUpper(toString(coalesce(legacy_i.cluster_type, legacy_i.clusterType)))}
                     END) +
                     collect(DISTINCT CASE
                       WHEN ic IS NOT NULL AND toUpper(toString(coalesce(ic.cluster_type, ic.clusterType, ""))) = $cluster_type
                       THEN properties(ic) + {cluster_type: toUpper(toString(coalesce(ic.cluster_type, ic.clusterType)))}
                     END) +
                     collect(DISTINCT CASE
                       WHEN inode IS NOT NULL AND toUpper(toString(coalesce(inode.cluster_type, inode.clusterType, ""))) = $cluster_type
                       THEN properties(inode) + {cluster_type: toUpper(toString(coalesce(inode.cluster_type, inode.clusterType)))}
                     END)
                   ) WHERE x IS NOT NULL] AS issue_props,
                   collect(DISTINCT properties(a)) AS analysis_props
            ORDER BY coalesce(v.published_at, v.publishedAt, v.created_at, v.createdAt, "") DESC
            LIMIT $limit
            """,
            {"limit": limit, "cluster_type": CLUSTER_TYPE_CURATION_MANUAL},
        )

    def _fetch_videos(self, country: str, language: str, limit: int) -> list[dict[str, Any]]:
        return self.client.execute_read(
            """
            MATCH (v:Video)
            OPTIONAL MATCH (v)-[:PART_OF]->(i:Issue)
            OPTIONAL MATCH (v)-[]-(legacy_i:Issue)
            OPTIONAL MATCH (v)-[]-(ic:IssueCluster)
            OPTIONAL MATCH (v)-[]-(inode:IssueNode)
            OPTIONAL MATCH (v)-[:PUBLISHED_BY]->(c:Channel)
            OPTIONAL MATCH (v)-[]-(legacy_c:Channel)
            OPTIONAL MATCH (v)-[]-(a:AnalysisResult)
            OPTIONAL MATCH (v)-[]-(e:Entity)
            WITH v,
                 [x IN (
                   collect(DISTINCT CASE
                     WHEN i IS NOT NULL AND toUpper(toString(coalesce(i.cluster_type, i.clusterType, ""))) = $cluster_type
                     THEN properties(i) + {cluster_type: toUpper(toString(coalesce(i.cluster_type, i.clusterType)))}
                   END) +
                   collect(DISTINCT CASE
                     WHEN legacy_i IS NOT NULL AND toUpper(toString(coalesce(legacy_i.cluster_type, legacy_i.clusterType, ""))) = $cluster_type
                     THEN properties(legacy_i) + {cluster_type: toUpper(toString(coalesce(legacy_i.cluster_type, legacy_i.clusterType)))}
                   END) +
                   collect(DISTINCT CASE
                     WHEN ic IS NOT NULL AND toUpper(toString(coalesce(ic.cluster_type, ic.clusterType, ""))) = $cluster_type
                     THEN properties(ic) + {cluster_type: toUpper(toString(coalesce(ic.cluster_type, ic.clusterType)))}
                   END) +
                   collect(DISTINCT CASE
                     WHEN inode IS NOT NULL AND toUpper(toString(coalesce(inode.cluster_type, inode.clusterType, ""))) = $cluster_type
                     THEN properties(inode) + {cluster_type: toUpper(toString(coalesce(inode.cluster_type, inode.clusterType)))}
                   END)
                 ) WHERE x IS NOT NULL] AS issue_props,
                 collect(DISTINCT properties(c)) +
                 collect(DISTINCT properties(legacy_c)) AS channel_props,
                 collect(DISTINCT properties(a)) AS analysis_props,
                 collect(DISTINCT properties(e)) AS entity_props
            WHERE toUpper(toString(coalesce(v.country_code, v.country, v.countryCode, v.region, ""))) = $country
              AND toLower(toString(coalesce(v.language, v.lang, v.language_code, v.languageCode, ""))) = $language
            RETURN properties(v) AS video,
                   issue_props,
                   CASE WHEN size(channel_props) > 0 THEN channel_props[0] ELSE {} END AS channel,
                   CASE WHEN size(analysis_props) > 0 THEN analysis_props[0] ELSE {} END AS analysis,
                   entity_props
            ORDER BY
              CASE WHEN toUpper(toString(coalesce(CASE WHEN size(analysis_props) > 0 THEN analysis_props[0].status ELSE "SUCCESS" END, "SUCCESS"))) = "SUCCESS" THEN 1 ELSE 0 END DESC,
              coalesce(v.published_at, v.publishedAt, v.created_at, v.createdAt, "") DESC,
              coalesce(v.view_count, v.viewCount, 0) DESC
            LIMIT $limit
            """,
            {
                "country": country,
                "language": language.lower(),
                "limit": limit,
                "cluster_type": CLUSTER_TYPE_CURATION_MANUAL,
            },
        )

    def _fetch_video_by_id(self, video_id: str) -> dict[str, Any] | None:
        rows = self.client.execute_read(
            """
            MATCH (v:Video)
            WHERE toString(coalesce(v.video_id, v.id, "")) = $video_id
            OPTIONAL MATCH (v)-[:PART_OF]->(i:Issue)
            OPTIONAL MATCH (v)-[]-(legacy_i:Issue)
            OPTIONAL MATCH (v)-[]-(ic:IssueCluster)
            OPTIONAL MATCH (v)-[]-(inode:IssueNode)
            OPTIONAL MATCH (v)-[:PUBLISHED_BY]->(c:Channel)
            OPTIONAL MATCH (v)-[]-(legacy_c:Channel)
            OPTIONAL MATCH (v)-[]-(a:AnalysisResult)
            OPTIONAL MATCH (v)-[]-(e:Entity)
            RETURN properties(v) AS video,
                   [x IN (
                     collect(DISTINCT CASE
                       WHEN i IS NOT NULL AND toUpper(toString(coalesce(i.cluster_type, i.clusterType, ""))) = $cluster_type
                       THEN properties(i) + {cluster_type: toUpper(toString(coalesce(i.cluster_type, i.clusterType)))}
                     END) +
                     collect(DISTINCT CASE
                       WHEN legacy_i IS NOT NULL AND toUpper(toString(coalesce(legacy_i.cluster_type, legacy_i.clusterType, ""))) = $cluster_type
                       THEN properties(legacy_i) + {cluster_type: toUpper(toString(coalesce(legacy_i.cluster_type, legacy_i.clusterType)))}
                     END) +
                     collect(DISTINCT CASE
                       WHEN ic IS NOT NULL AND toUpper(toString(coalesce(ic.cluster_type, ic.clusterType, ""))) = $cluster_type
                       THEN properties(ic) + {cluster_type: toUpper(toString(coalesce(ic.cluster_type, ic.clusterType)))}
                     END) +
                     collect(DISTINCT CASE
                       WHEN inode IS NOT NULL AND toUpper(toString(coalesce(inode.cluster_type, inode.clusterType, ""))) = $cluster_type
                       THEN properties(inode) + {cluster_type: toUpper(toString(coalesce(inode.cluster_type, inode.clusterType)))}
                     END)
                   ) WHERE x IS NOT NULL] AS issue_props,
                   CASE
                     WHEN size(collect(DISTINCT properties(c)) + collect(DISTINCT properties(legacy_c))) > 0
                     THEN (collect(DISTINCT properties(c)) + collect(DISTINCT properties(legacy_c)))[0]
                     ELSE {}
                   END AS channel,
                   CASE WHEN count(a) > 0 THEN collect(DISTINCT properties(a))[0] ELSE {} END AS analysis,
                   collect(DISTINCT properties(e)) AS entity_props
            LIMIT 1
            """,
            {"video_id": video_id, "cluster_type": CLUSTER_TYPE_CURATION_MANUAL},
        )
        return rows[0] if rows else None

    def _fetch_related_videos_by_issue(
        self,
        *,
        source_video_id: str,
        country: str,
        language: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.client.execute_read(
            """
            MATCH (source:Video)-[:PART_OF]->(i:Issue)<-[:PART_OF]-(related:Video)
            WHERE toString(coalesce(source.video_id, source.id, "")) = $source_video_id
              AND toString(coalesce(related.video_id, related.id, "")) <> $source_video_id
              AND toUpper(toString(coalesce(i.cluster_type, i.clusterType, ""))) = $cluster_type
              AND toUpper(toString(coalesce(related.country_code, related.country, related.countryCode, related.region, ""))) = $country
              AND toLower(toString(coalesce(related.language, related.lang, related.language_code, related.languageCode, ""))) = $language
            OPTIONAL MATCH (related)-[]-(legacy_i:Issue)
            OPTIONAL MATCH (related)-[]-(ic:IssueCluster)
            OPTIONAL MATCH (related)-[]-(inode:IssueNode)
            OPTIONAL MATCH (related)-[:PUBLISHED_BY]->(c:Channel)
            OPTIONAL MATCH (related)-[]-(legacy_c:Channel)
            OPTIONAL MATCH (related)-[]-(a:AnalysisResult)
            OPTIONAL MATCH (related)-[]-(e:Entity)
            WITH related,
                 [x IN (
                   collect(DISTINCT CASE
                     WHEN i IS NOT NULL AND toUpper(toString(coalesce(i.cluster_type, i.clusterType, ""))) = $cluster_type
                     THEN properties(i) + {cluster_type: toUpper(toString(coalesce(i.cluster_type, i.clusterType)))}
                   END) +
                   collect(DISTINCT CASE
                     WHEN legacy_i IS NOT NULL AND toUpper(toString(coalesce(legacy_i.cluster_type, legacy_i.clusterType, ""))) = $cluster_type
                     THEN properties(legacy_i) + {cluster_type: toUpper(toString(coalesce(legacy_i.cluster_type, legacy_i.clusterType)))}
                   END) +
                   collect(DISTINCT CASE
                     WHEN ic IS NOT NULL AND toUpper(toString(coalesce(ic.cluster_type, ic.clusterType, ""))) = $cluster_type
                     THEN properties(ic) + {cluster_type: toUpper(toString(coalesce(ic.cluster_type, ic.clusterType)))}
                   END) +
                   collect(DISTINCT CASE
                     WHEN inode IS NOT NULL AND toUpper(toString(coalesce(inode.cluster_type, inode.clusterType, ""))) = $cluster_type
                     THEN properties(inode) + {cluster_type: toUpper(toString(coalesce(inode.cluster_type, inode.clusterType)))}
                   END)
                 ) WHERE x IS NOT NULL] AS issue_props,
                 collect(DISTINCT properties(c)) +
                 collect(DISTINCT properties(legacy_c)) AS channel_props,
                 collect(DISTINCT properties(a)) AS analysis_props,
                 collect(DISTINCT properties(e)) AS entity_props
            RETURN properties(related) AS video,
                   issue_props,
                   CASE WHEN size(channel_props) > 0 THEN channel_props[0] ELSE {} END AS channel,
                   CASE WHEN size(analysis_props) > 0 THEN analysis_props[0] ELSE {} END AS analysis,
                   entity_props
            ORDER BY
              CASE WHEN toUpper(toString(coalesce(CASE WHEN size(analysis_props) > 0 THEN analysis_props[0].status ELSE "SUCCESS" END, "SUCCESS"))) = "SUCCESS" THEN 1 ELSE 0 END DESC,
              coalesce(related.view_count, related.viewCount, 0) DESC,
              coalesce(related.published_at, related.publishedAt, related.created_at, related.createdAt, "") DESC
            LIMIT $limit
            """,
            {
                "source_video_id": source_video_id,
                "country": country,
                "language": language.lower(),
                "limit": limit,
                "cluster_type": CLUSTER_TYPE_CURATION_MANUAL,
            },
        )

    def _rank_issue_keywords(self, rows: list[dict[str, Any]], max_count: int) -> list[str]:
        counter: Counter[str] = Counter()
        country_sets: dict[str, set[str]] = {}

        for row in rows:
            country = self._norm_country(row.get("video") or {})
            for keyword in self._extract_keywords(row, max_count=20):
                counter[keyword] += 1
                country_sets.setdefault(keyword, set())
                if country:
                    country_sets[keyword].add(country)

        ranked = sorted(
            counter,
            key=lambda kw: (len(country_sets.get(kw, set())), counter[kw], kw),
            reverse=True,
        )
        keywords = [kw for kw in ranked if len(country_sets.get(kw, set())) >= 2][:max_count]
        if len(keywords) < max_count:
            keywords.extend([kw for kw in ranked if kw not in keywords][: max_count - len(keywords)])
        if not keywords:
            keywords = FALLBACK_ISSUE_KEYWORDS[:max_count]
        return keywords[:max_count]

    def _rank_search_rows(
        self,
        rows: list[dict[str, Any]],
        keyword: str,
    ) -> list[tuple[dict[str, Any], float]]:
        if not keyword:
            return [(row, self._base_video_score(row)) for row in rows]

        needle = keyword.lower()
        ranked: list[tuple[dict[str, Any], float]] = []
        for row in rows:
            haystack_parts = [
                str((row.get("video") or {}).get("title") or ""),
                str((row.get("video") or {}).get("description") or ""),
                " ".join(self._extract_keywords(row, max_count=50)),
            ]
            haystack = " ".join(haystack_parts).lower()
            if needle not in haystack:
                continue
            score = self._base_video_score(row)
            score += haystack.count(needle) * 2.0
            ranked.append((row, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _rank_related_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        source_video_id: str,
        source_keywords: list[str],
        source_issue_ids: set[str],
        source_entities: set[str],
        source_opinion_score: float | None,
        degrade_reasons: Counter[str] | None = None,
    ) -> list[tuple[dict[str, Any], float, list[str], list[str], float | None, float | None]]:
        ranked: list[tuple[dict[str, Any], float, list[str], list[str], float | None, float | None]] = []
        source_keyword_set = {kw.lower(): kw for kw in source_keywords}

        for row in rows:
            summary = self._row_to_summary(row)
            if summary.video_id == source_video_id:
                continue
            if not summary.video_id:
                if degrade_reasons is not None:
                    degrade_reasons["candidate_missing_video_id"] += 1
                continue
            if summary.country_code is None or summary.language is None:
                if degrade_reasons is not None:
                    degrade_reasons["candidate_missing_country_or_language"] += 1
                continue

            row_keywords = self._extract_keywords(row, max_count=20)
            row_keyword_map = {kw.lower(): kw for kw in row_keywords}
            shared_keys = sorted(set(source_keyword_set) & set(row_keyword_map))
            shared_keywords = [source_keyword_set[key] for key in shared_keys]

            issue_overlap = source_issue_ids & set(self._extract_issue_ids(row))
            entity_overlap = source_entities & set(self._extract_entity_keys(row))

            reasons: list[str] = []
            if issue_overlap:
                issue_names = self._extract_issue_names(row)
                reasons.append(f"같은 이슈 클러스터: {', '.join(issue_names[:3]) or ', '.join(sorted(issue_overlap)[:3])}")
            if shared_keywords:
                reasons.append(f"공유 키워드: {', '.join(shared_keywords[:5])}")
            if entity_overlap:
                reasons.append(f"공유 엔티티: {', '.join(sorted(entity_overlap)[:5])}")

            score = self._base_video_score(row)
            score += len(issue_overlap) * 5.0
            score += len(shared_keywords) * 2.0
            score += len(entity_overlap) * 3.0

            candidate_opinion = self._extract_opinion_score(row)
            opinion_distance: float | None = None
            similarity_score: float | None = None
            if source_opinion_score is not None and candidate_opinion is not None:
                opinion_distance = abs(source_opinion_score - candidate_opinion)
                similarity_score = max(0.0, min(1.0, 1.0 - opinion_distance))
            else:
                reasons.append("opinion_score 없음으로 내용 유사도 기반 fallback을 적용했습니다.")
                if degrade_reasons is not None:
                    degrade_reasons["missing_analysis_for_similarity"] += 1

            if reasons:
                ranked.append((row, score, shared_keywords, reasons, similarity_score, opinion_distance))
            elif degrade_reasons is not None:
                degrade_reasons["candidate_no_relation_signal"] += 1

        def _sort_key(item: tuple[dict[str, Any], float, list[str], list[str], float | None, float | None]) -> tuple[float, float]:
            _row, relevance, _shared, _reasons, _similarity, distance = item
            if distance is None:
                # fallback rows: keep original relevance ordering
                return (float("inf"), -relevance)
            return (distance, -relevance)

        return sorted(ranked, key=_sort_key)

    def _row_to_summary(self, row: dict[str, Any]) -> VideoSummary:
        video = row.get("video") or {}
        channel = row.get("channel") or {}
        analysis = row.get("analysis") or {}
        return VideoSummary(
            video_id=str(video.get("video_id") or video.get("id") or ""),
            target_id=self._to_int(video.get("target_id") or video.get("targetId")),
            title=str(video.get("title") or ""),
            description=self._optional_str(video.get("description")),
            thumbnail_url=self._optional_str(
                video.get("thumbnail_url") or video.get("thumbnailUrl") or video.get("thumbnail")
            ),
            channel_name=self._optional_str(
                channel.get("channel_name")
                or channel.get("name")
                or video.get("channel_name")
                or video.get("channelName")
            ),
            published_at=self._optional_str(
                video.get("published_at")
                or video.get("publishedAt")
                or video.get("created_at")
                or video.get("createdAt")
            ),
            view_count=self._to_float(video.get("view_count") or video.get("viewCount")),
            country_code=self._norm_country(video),
            language=self._norm_language(video),
            analysis_status=self._optional_str(analysis.get("status") or video.get("analysis_status")),
            cluster_types=self._extract_cluster_types(row),
        )

    def _summary_to_node(self, summary: VideoSummary, node_type: str) -> GraphNode:
        return GraphNode(
            id=self._node_id(summary.video_id),
            video_id=summary.video_id,
            target_id=summary.target_id,
            title=summary.title,
            thumbnail_url=summary.thumbnail_url,
            channel_name=summary.channel_name,
            country_code=summary.country_code,
            language=summary.language,
            node_type=node_type,
            analysis_status=summary.analysis_status,
            cluster_types=summary.cluster_types,
        )

    def _extract_keywords(self, row: dict[str, Any], max_count: int) -> list[str]:
        video = row.get("video") or {}
        analysis = row.get("analysis") or {}
        raw_values: list[Any] = []
        raw_values.extend(
            [
                video.get("video_keywords"),
                video.get("videoKeywords"),
                video.get("keywords"),
                analysis.get("analysis_keywords"),
                analysis.get("analysisKeywords"),
                analysis.get("keywords"),
            ]
        )
        for issue in self._dict_items(row.get("issue_props")):
            raw_values.extend([issue.get("name"), issue.get("title"), issue.get("keyword")])

        keywords = self._flatten_keywords(raw_values)
        if not keywords:
            keywords = self._tokenize_text(
                " ".join([str(video.get("title") or ""), str(video.get("description") or "")])
            )

        seen: set[str] = set()
        cleaned: list[str] = []
        for keyword in keywords:
            normalized = keyword.strip()
            if not normalized or normalized.lower() in STOPWORDS:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
            if len(cleaned) >= max_count:
                break
        return cleaned

    def _flatten_keywords(self, values: list[Any]) -> list[str]:
        keywords: list[str] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                if "," in value:
                    keywords.extend(part.strip() for part in value.split(","))
                else:
                    keywords.append(value.strip())
                continue
            if isinstance(value, dict):
                for key in ("keyword_text", "keyword", "name", "text", "value"):
                    if value.get(key):
                        keywords.append(str(value[key]).strip())
                        break
                continue
            if isinstance(value, list):
                keywords.extend(self._flatten_keywords(value))
        return keywords

    def _tokenize_text(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", text) if token.lower() not in STOPWORDS]

    def _extract_issue_ids(self, row: dict[str, Any]) -> list[str]:
        values = []
        for issue in self._dict_items(row.get("issue_props")):
            raw = issue.get("issue_id") or issue.get("id") or issue.get("name")
            if raw:
                values.append(str(raw))
        return values

    def _extract_issue_names(self, row: dict[str, Any]) -> list[str]:
        names = []
        for issue in self._dict_items(row.get("issue_props")):
            raw = issue.get("name") or issue.get("title") or issue.get("keyword") or issue.get("issue_id")
            if raw:
                names.append(str(raw))
        return names

    def _extract_cluster_types(self, row: dict[str, Any]) -> list[str]:
        cluster_types: list[str] = []
        for issue in self._dict_items(row.get("issue_props")):
            raw = issue.get("cluster_type") or issue.get("clusterType")
            if not raw:
                continue
            normalized = str(raw).strip().upper()
            if normalized and normalized not in cluster_types:
                cluster_types.append(normalized)
        return cluster_types

    def _extract_entity_keys(self, row: dict[str, Any]) -> list[str]:
        values = []
        for entity in self._dict_items(row.get("entity_props")):
            raw = entity.get("entity_key") or entity.get("name") or entity.get("text") or entity.get("id")
            if raw:
                values.append(str(raw))
        return values

    def _top_keywords_for_rows(self, rows: list[dict[str, Any]], max_count: int) -> list[str]:
        counter: Counter[str] = Counter()
        original: dict[str, str] = {}
        for row in rows:
            for keyword in self._extract_keywords(row, max_count=30):
                key = keyword.lower()
                original.setdefault(key, keyword)
                counter[key] += 1
        return [original[key] for key, _count in counter.most_common(max_count)]

    def _build_perspective_summary(self, country: str, keywords: list[str]) -> str:
        country_name = {"KR": "한국", "US": "미국", "CN": "중국"}.get(country, country)
        if not keywords:
            return f"{country_name} 관련 영상이 부족해 관점 요약을 만들 수 없습니다."
        return f"{country_name}은/는 {', '.join(keywords[:3])} 키워드를 중심으로 이 이슈를 다룹니다."

    def _base_video_score(self, row: dict[str, Any]) -> float:
        summary = self._row_to_summary(row)
        score = 0.0
        if (summary.analysis_status or "SUCCESS").upper() == "SUCCESS":
            score += 2.0
        score += min(summary.view_count, 1_000_000.0) / 1_000_000.0
        if summary.published_at:
            score += 0.5
        return score

    def _relation_type(self, reasons: list[str]) -> str:
        if any(reason.startswith("같은 이슈") for reason in reasons):
            return "SAME_ISSUE"
        if any(reason.startswith("공유 엔티티") for reason in reasons):
            return "SHARED_ENTITY"
        return "SHARED_KEYWORD"

    def _safe_limit(self, limit: int) -> int:
        return max(1, min(limit, 20))

    def _node_id(self, video_id: str) -> str:
        return f"video:{video_id}"

    def _norm_country(self, video: dict[str, Any]) -> str | None:
        for key in ("country_code", "country", "countryCode", "region", "region_code"):
            raw = video.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip().upper()
        return None

    def _norm_language(self, video: dict[str, Any]) -> str | None:
        for key in ("language", "lang", "language_code", "languageCode"):
            raw = video.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip().lower()
        return None

    def _to_float(self, value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _extract_opinion_score(self, row: dict[str, Any] | None) -> float | None:
        if not row:
            return None
        analysis = row.get("analysis") or {}
        video = row.get("video") or {}
        for raw in (
            analysis.get("opinion_score"),
            analysis.get("opinionScore"),
            video.get("opinion_score"),
            video.get("opinionScore"),
        ):
            if raw is None:
                continue
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue
            return max(0.0, min(1.0, score))
        return None

    def _rounded_or_none(self, value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 4)

    def _log_graph_degrade(
        self,
        *,
        source_video_id: str,
        source_country: str | None,
        source_language: str | None,
        source_has_analysis: bool,
        reasons: Counter[str],
    ) -> None:
        logger.info(
            "comparison_graph_degrade feature_mode=CURATION source_video_id=%s source_country=%s source_language=%s source_has_analysis=%s cluster_type=%s reasons=%s",
            source_video_id,
            source_country,
            source_language,
            source_has_analysis,
            CLUSTER_TYPE_CURATION_MANUAL,
            dict(reasons),
        )

    def _to_int(self, value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        return text if text else None

    def _dict_items(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict) and item]
