"""Qdrant 컬렉션 초기화 스크립트.

서버 최초 실행 전 1회 실행:
    python scripts/init_qdrant.py

포함 컬렉션:
    - emotion_words    : KNU 감정사전 부정어 임베딩
    - anonymous_patterns: 익명출처/추측 표현 임베딩
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.config import (
    QDRANT_ANONYMOUS_COLLECTION,
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)

SENTI_WORD_PATH = "analysis_bc/data/SentiWord_info.json"
ANONYMOUS_PATTERNS_PATH = "analysis_bc/data/anonymous_patterns.json"
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
VECTOR_SIZE = 768
BATCH_SIZE = 100


def init_emotion_words(client: QdrantClient, model: SentenceTransformer) -> None:
    print("감정사전 로드 중...", flush=True)
    with open(SENTI_WORD_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    negative = [e for e in entries if int(e["polarity"]) <= -1]
    print(f"  전체: {len(entries)}개, 부정 필터링 후: {len(negative)}개", flush=True)

    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_EMOTION_COLLECTION in existing:
        print(f"기존 컬렉션 '{QDRANT_EMOTION_COLLECTION}' 삭제 후 재생성", flush=True)
        client.delete_collection(QDRANT_EMOTION_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_EMOTION_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"컬렉션 '{QDRANT_EMOTION_COLLECTION}' 생성 완료", flush=True)

    print("임베딩 생성 및 적재 중...", flush=True)
    for batch_start in range(0, len(negative), BATCH_SIZE):
        batch = negative[batch_start : batch_start + BATCH_SIZE]
        word_roots = [e["word_root"] for e in batch]
        embeddings = model.encode(word_roots, show_progress_bar=False)

        points = [
            PointStruct(
                id=batch_start + i,
                vector=embeddings[i].tolist(),
                payload={
                    "word": entry["word"],
                    "word_root": entry["word_root"],
                    "polarity": entry["polarity"],
                },
            )
            for i, entry in enumerate(batch)
        ]
        client.upsert(collection_name=QDRANT_EMOTION_COLLECTION, points=points)
        print(f"  {batch_start + len(batch)}/{len(negative)} 적재 완료", flush=True)

    print("emotion_words 완료!", flush=True)


def init_anonymous_patterns(client: QdrantClient, model: SentenceTransformer) -> None:
    print("익명출처/추측 표현 로드 중...", flush=True)
    with open(ANONYMOUS_PATTERNS_PATH, encoding="utf-8") as f:
        patterns = json.load(f)
    print(f"  패턴 수: {len(patterns)}개", flush=True)

    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_ANONYMOUS_COLLECTION in existing:
        print(f"기존 컬렉션 '{QDRANT_ANONYMOUS_COLLECTION}' 삭제 후 재생성", flush=True)
        client.delete_collection(QDRANT_ANONYMOUS_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_ANONYMOUS_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"컬렉션 '{QDRANT_ANONYMOUS_COLLECTION}' 생성 완료", flush=True)

    print("임베딩 생성 및 적재 중...", flush=True)
    for batch_start in range(0, len(patterns), BATCH_SIZE):
        batch = patterns[batch_start : batch_start + BATCH_SIZE]
        phrases = [e["phrase"] for e in batch]
        embeddings = model.encode(phrases, show_progress_bar=False)

        points = [
            PointStruct(
                id=batch_start + i,
                vector=embeddings[i].tolist(),
                payload={
                    "phrase": entry["phrase"],
                    "label_type": entry["label_type"],
                },
            )
            for i, entry in enumerate(batch)
        ]
        client.upsert(collection_name=QDRANT_ANONYMOUS_COLLECTION, points=points)
        print(f"  {batch_start + len(batch)}/{len(patterns)} 적재 완료", flush=True)

    print("anonymous_patterns 완료!", flush=True)


def main() -> None:
    print("임베딩 모델 로드 중...", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Qdrant 연결 중...", flush=True)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    init_emotion_words(client, model)

    if not Path(ANONYMOUS_PATTERNS_PATH).exists():
        print(
            "anonymous_patterns.json 없음 → 스킵 (mine_anonymous_patterns.py 먼저 실행)",
            flush=True,
        )
    else:
        init_anonymous_patterns(client, model)

    print("완료!", flush=True)


if __name__ == "__main__":
    main()
