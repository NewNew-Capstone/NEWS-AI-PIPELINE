# Country Video Comparison Postman Test Guide

## Files

- Collection: `docs/postman/country_video_comparison.postman_collection.json`
- Environment: `docs/postman/country_video_comparison.postman_environment.json`

## Preconditions

Python FastAPI must be running at:

```bash
http://localhost:8000
```

Spring is assumed to run at:

```bash
http://localhost:8080
```

Spring comparison proxy paths are expected under:

```bash
/api/v1/comparison
```

Python must have Neo4j environment variables configured:

```env
NEO4J_URI=neo4j+s://...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

## Import

1. Open Postman.
2. Import `country_video_comparison.postman_collection.json`.
3. Import `country_video_comparison.postman_environment.json`.
4. Select the `Country Video Comparison Local` environment.

## Run Order

Run the Python requests first:

1. `Python - Health`
2. `Python - Neo4j Health`
3. `Python - Comparison Home`
4. `Python - Search Videos`
5. `Python - Comparison Graph`

`Python - Search Videos` automatically stores the first returned `video_id` in the environment. If no videos are returned, set `video_id` manually from Neo4j-backed data before running the graph request.

Then run the Spring proxy requests if Spring proxy endpoints exist:

1. `Spring - Comparison Home`
2. `Spring - Search Videos`
3. `Spring - Comparison Graph`
4. `Spring - Video Target`

Spring proxy requests treat `404` and `501` as "proxy not implemented yet" and skip structure checks.

## Expected Checks

- Python `/health` returns `status=ok`.
- Python `/kg/health` returns `status=ok` and `ok=1`.
- `/kg/comparison-home` returns three sections: `KR/ko`, `US/en`, `CN/zh`.
- `/kg/search-videos` returns the same three section keys.
- `/kg/videos/{video_id}/comparison-graph` returns one `source` node.
- Every returned graph node includes `video_id` for React routing.
- Every returned graph edge has at least one `keyword` or `reason`.
- `opinion_distance` is non-negative when present.
- `similarity_score` is within `0..1` when present.
- Spring `/api/v1/comparison/videos/{video_id}/target` returns a numeric target ID for analysis routing.

## Common Failures

- `503` from `/kg/health`: check Neo4j env vars and network access.
- Empty `videos`: Neo4j has no matching `country_code + language` data yet.
- Graph pre-request error: run search first or set `video_id` manually.
- Spring `404`: Spring proxy endpoint has not been implemented; test Python directly.
