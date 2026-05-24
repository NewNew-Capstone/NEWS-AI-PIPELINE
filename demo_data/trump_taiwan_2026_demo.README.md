# Trump-Taiwan Demo Curation Dataset

This folder is the source of truth for the manual demo set used by feature 2.

## Fixed Demo Scope

- Issue ID: `trump_taiwan_2026_demo`
- Issue name: `트럼프-대만/미중 갈등`
- Cluster type in Spring/Neo4j: `CURATION_MANUAL`
- Countries: `KR`, `US`, `CN`
- Target size: 10-15 videos per country, 30-45 videos total

## Required CSV Columns

- `issue_id`
- `issue_name`
- `country_code`
- `language`
- `youtube_video_id`
- `title`
- `channel_name`
- `channel_id`
- `published_at`
- `curation_reason`

Optional columns:

- `stance_hint`
- `frame_hint`
- `backup_rank`

## Curation Rules

Include videos that directly connect Trump, Taiwan, and US-China conflict framing.

Recommended search terms:

- `Trump Taiwan`
- `Trump China Taiwan`
- `Taiwan Strait`
- `트럼프 대만`
- `트럼프 중국 대만`
- `特朗普 台湾`
- `特朗普 台海`

Exclude videos when:

- The video is only general US politics.
- China is mentioned but Taiwan is not part of the issue.
- Transcript or analysis cannot be produced.
- Country or language cannot be assigned.

## Workflow

1. Fill `trump_taiwan_2026_demo.csv`.
2. Validate the dataset:

   ```bash
   .venv/bin/python scripts/validate_demo_curation.py \
     --csv demo_data/trump_taiwan_2026_demo.csv \
     --output-json logs/demo_curation/trump_taiwan_validation.json \
     --output-md logs/demo_curation/trump_taiwan_validation.md
   ```

3. Apply it to Spring after the backend is running:

   ```bash
   .venv/bin/python scripts/apply_demo_curation.py \
     --csv demo_data/trump_taiwan_2026_demo.csv \
     --backend-url http://127.0.0.1:8080 \
     --trigger-analysis \
     --lock
   ```

4. Verify:

   - `GET /api/v1/issues/curation/sets/{issueClusterId}/status`
   - `GET /api/v1/issues/comparison/report?issueClusterId={issueClusterId}`
   - `GET /api/v1/comparison/home`
   - `GET /api/v1/comparison/search?keyword=트럼프%20대만`
   - `GET /api/v1/comparison/videos/{youtubeVideoId}/graph`
