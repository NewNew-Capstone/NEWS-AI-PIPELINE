# Spring Neo4j Ingestion Contract

## Purpose

Spring owns Neo4j writes. Python owns Neo4j schema initialization and read APIs under `/kg/*`.

The integration is successful when Spring-created graph data can be read by:

- `GET /kg/comparison-home`
- `GET /kg/search-videos?keyword=...`
- `GET /kg/videos/{video_id}/comparison-graph`

Spring proxy endpoints should forward these Python responses without reshaping the JSON.

## Required Nodes

### `Video`

Required properties:

- `video_id`: YouTube video ID. Unique key.
- `target_id`: Internal analysis target PK used by `/api/v1/analysis/{targetId}` routing.
- `country_code`: Uppercase country code such as `KR`, `US`, `CN`.
- `language`: Lowercase language code such as `ko`, `en`, `zh`.
- `title`
- `thumbnail_url`
- `published_at`
- `view_count`

Accepted fallback names already supported by Python:

- `country`, `countryCode`, `region`, `region_code`
- `lang`, `language_code`, `languageCode`
- `thumbnailUrl`, `thumbnail`
- `publishedAt`, `created_at`, `createdAt`
- `viewCount`
- `targetId`

### `Channel`

Required properties:

- `channel_id`: Unique key.
- `channel_name`

Accepted fallback names:

- `name`

### Issue Node

Preferred Spring label:

- `Issue`

Also supported by Python:

- `IssueCluster`
- `IssueNode`

Required properties:

- `issue_id`: Unique key.
- At least one display/search field: `name`, `title`, or `keyword`.
- `cluster_type`: `SEARCH_AUTO` or `CURATION_MANUAL` (required for `IssueCluster`, recommended for all compatible issue labels)

Cluster type policy in Python:

- Feature1 (`/issue/*`) uses `SEARCH_AUTO` only.
- Feature2 (`/kg/comparison-*`) uses `CURATION_MANUAL` only.
- Legacy missing `cluster_type` is treated as `SEARCH_AUTO` only in feature1; feature2 excludes missing values.

### `AnalysisResult`

Recommended properties:

- `analysis_id`: Unique key.
- `status`: `SUCCESS` should be used for successfully analyzed videos.
- `overall_bias_score`
- `tone_label`
- `analysis_keywords` or `keywords`

## Required Relationships

Python prioritizes the Spring relationship types below for `/kg/*` reads. Legacy direction-agnostic matches are still kept as fallback for older data:

- `Video` connected to `Channel`
- `Video` connected to `Issue` / `IssueCluster` / `IssueNode`
- `Video` connected to `AnalysisResult` when analysis exists

Recommended names for readability:

- `(Video)-[:PUBLISHED_BY]->(Channel)`
- `(Video)-[:PART_OF]->(Issue)`
- `(Video)-[:HAS_ANALYSIS]->(AnalysisResult)`

## Manual Verification Cypher

Confirm video and channel ingestion:

```cypher
MATCH (v:Video)-[:PUBLISHED_BY]->(c:Channel)
RETURN v.video_id,
       v.target_id,
       v.title,
       v.country_code,
       v.language,
       c.channel_id,
       c.channel_name
LIMIT 20;
```

Confirm issue links:

```cypher
MATCH (v:Video)-[:PART_OF]->(i:Issue)
RETURN v.video_id,
       v.country_code,
       i.issue_id,
       i.name,
       i.keyword
LIMIT 30;
```

Confirm graph candidates for a selected source video:

```cypher
MATCH (source:Video)-[:PART_OF]->(i:Issue)<-[:PART_OF]-(related:Video)
WHERE source.video_id = "PUT_YOUTUBE_VIDEO_ID_HERE"
RETURN source.video_id,
       source.country_code,
       i.issue_id,
       i.name,
       i.keyword,
       related.video_id,
       related.country_code,
       related.title
LIMIT 30;
```

## Spring Proxy Contract

React should call Spring only:

- `GET /api/v1/comparison/home` -> Python `GET /kg/comparison-home`
- `GET /api/v1/comparison/search?keyword=...` -> Python `GET /kg/search-videos?keyword=...`
- `GET /api/v1/comparison/videos/{id}/graph` -> Python `GET /kg/videos/{id}/comparison-graph`
- `GET /api/v1/comparison/videos/{id}/target` -> Spring DB lookup from YouTube `video_id` to internal `targetId`

The first three endpoints should preserve the Python response body shape. The `/target` endpoint should return the internal target ID used to navigate to `/api/v1/analysis/{targetId}`.

## Verification Checklist

1. Run Python `/kg/health` and confirm `status=ok`.
2. Run Python `/kg/comparison-home?limit=5` and confirm sections for `KR/ko`, `US/en`, `CN/zh`.
3. Run Python `/kg/search-videos?keyword=반도체&limit=5` and confirm at least one returned `video_id` after Spring has ingested matching data.
4. Run Python `/kg/videos/{video_id}/comparison-graph?limit_per_country=5` and confirm one source node plus related nodes or an intentionally empty graph.
5. Run Spring `/api/v1/comparison/*` proxy requests and confirm the JSON shape matches Python responses.
6. Click a graph node in React and confirm `video_id -> /target -> targetId -> /api/v1/analysis/{targetId}` navigation works.
