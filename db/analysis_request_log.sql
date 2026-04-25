-- 운영 분석 요청 로그(문장 단위) 저장 테이블
CREATE TABLE IF NOT EXISTS public.analysis_request_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_endpoint TEXT NOT NULL,
    target_id BIGINT NULL,
    transcript_id BIGINT NULL,
    language VARCHAR(8) NULL,
    target_type TEXT NULL,
    country VARCHAR(8) NULL,
    sentence_count INT NOT NULL,
    sentences_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analysis_request_log_created_at
    ON public.analysis_request_log (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analysis_request_log_endpoint_created_at
    ON public.analysis_request_log (source_endpoint, created_at DESC);
