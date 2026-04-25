# analysis_bc/config.py
import os

from dotenv import load_dotenv

load_dotenv()

REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", 6380))
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

QDRANT_EMOTION_COLLECTION      = "emotion_words"
EMOTION_VECTOR_SIZE            = 300   # FastText cc.ko.300
EMOTION_SIMILARITY_THRESHOLD   = 0.80  # FastText cosine은 더 분별력 있어 낮춰도 됨

FASTTEXT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH", "analysis_bc/data/cc.ko.300.bin")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")