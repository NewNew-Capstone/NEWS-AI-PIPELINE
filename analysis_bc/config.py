# analysis_bc/config.py
import os

from dotenv import load_dotenv

load_dotenv()

REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", 6380))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

QDRANT_EMOTION_COLLECTION      = "emotion_words"
EMOTION_SIMILARITY_THRESHOLD   = 0.85

QDRANT_ANONYMOUS_COLLECTION    = "anonymous_patterns"
ANONYMOUS_SIMILARITY_THRESHOLD = 0.85

REDIS_ANONYMOUS_KEY   = "pattern:anonymous:list"
REDIS_SPECULATIVE_KEY = "pattern:speculative:list"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")