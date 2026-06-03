# analysis_bc/config.py
import os

from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

QDRANT_EMOTION_COLLECTION = os.getenv("QDRANT_EMOTION_COLLECTION", "emotion_words_v2")
EMOTION_VECTOR_SIZE = 300  # FastText cc.ko.300
EMOTION_SIMILARITY_THRESHOLD = float(os.getenv("EMOTION_SIMILARITY_THRESHOLD", "0.88"))
EMOTION_TOP_K = int(os.getenv("EMOTION_TOP_K", "3"))
EMOTION_SCORE_MARGIN = float(os.getenv("EMOTION_SCORE_MARGIN", "0.06"))

EMOTION_GATE_ENABLED = os.getenv("EMOTION_GATE_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
EMOTION_GATE_MODEL_NAME = os.getenv("EMOTION_GATE_MODEL_NAME", "searle-j/kote_for_easygoing_people")
EMOTION_GATE_THRESHOLD = float(os.getenv("EMOTION_GATE_THRESHOLD", "0.35"))
EMOTION_GATE_LABEL_THRESHOLD = float(os.getenv("EMOTION_GATE_LABEL_THRESHOLD", "0.20"))
EMOTION_GATE_TOP_K = int(os.getenv("EMOTION_GATE_TOP_K", "3"))
EMOTION_POLARITY_FILTER_ENABLED = os.getenv("EMOTION_POLARITY_FILTER_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}

FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "analysis_bc/data/cc.ko.300.bin")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
EMOTION_KEYWORD_LLM_FILTER_ENABLED = os.getenv(
    "EMOTION_KEYWORD_LLM_FILTER_ENABLED",
    "true",
).lower() in {"1", "true", "yes", "y", "on"}
EMOTION_KEYWORD_LLM_FILTER_MODEL = (
    os.getenv("EMOTION_KEYWORD_LLM_FILTER_MODEL")
    or os.getenv("ANTHROPIC_SUMMARY_MODEL")
    or os.getenv("ANTHROPIC_MODEL")
    or "claude-haiku-4-5-20251001"
)

# 운영 분석 요청 로그 저장(Postgres)
# 예: postgresql+psycopg://user:password@host:5432/dbname
ANALYSIS_DB_URL = os.getenv("ANALYSIS_DB_URL", "")

# 운영 분석 요청 로그 로컬 저장(JSONL, 선택)
# 예: logs/analysis_requests_local.jsonl
ANALYSIS_REQUEST_LOG_LOCAL_PATH = os.getenv("ANALYSIS_REQUEST_LOG_LOCAL_PATH", "")
ANALYSIS_REQUEST_LOG_LOCAL_DAILY_SPLIT = os.getenv("ANALYSIS_REQUEST_LOG_LOCAL_DAILY_SPLIT", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
ANALYSIS_REQUEST_LOG_LOCAL_RETENTION_DAYS = int(os.getenv("ANALYSIS_REQUEST_LOG_LOCAL_RETENTION_DAYS", "30"))
