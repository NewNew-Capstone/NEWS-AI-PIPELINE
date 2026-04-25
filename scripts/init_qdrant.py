"""Qdrant 컬렉션 초기화 스크립트.

서버 최초 실행 전 1회 실행:
    python scripts/init_qdrant.py

포함 컬렉션:
    - emotion_words: KNU 감정사전 부정어 임베딩
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fasttext
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.config import (
    FASTTEXT_MODEL_PATH,
    EMOTION_VECTOR_SIZE,
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)

SENTI_WORD_PATH = "analysis_bc/data/SentiWord_info.json"
BATCH_SIZE = 100

# 감정 태깅 블랙리스트: Qdrant index에서 제외할 word_root 목록
# 추가 기준:
#   1. 동음이의어: SentiWord에 부정어로 등록됐지만 뉴스에서는 다른 의미로 주로 쓰이는 단어
#      예) 시장 → 시장하다(배고프다)는 부정어이나, 뉴스에서는 시장(市場)으로 쓰임
#   2. 오탐 확인: test_tagger.py 실행 후 중립 문장에서 잘못 태깅된 단어
#   3. 일상에서 거의 안 쓰이는 고어/사투리 감정어로 인한 오탐
_EMOTION_BLACKLIST: frozenset[str] = frozenset({
    "시장",  # 시장하다(배고프다) → 뉴스에서는 시장(市場)으로 쓰여 오탐 발생
    "인사",  # 인사불성이(정신을 잃다) → 뉴스에서는 인사(人事)로 쓰여 오탐 발생
    "경상",  # 경상(부상) → 뉴스에서는 경상수지, 경상도로 쓰여 오탐 발생
    "잘",    # 잘망스럽게, 잘아지고 등의 어근 → 뉴스에서는 부사 '잘'로 쓰여 오탐 발생
    "해지",  # 해지거나(햇빛에 바래다) → 뉴스에서는 계약 해지로 쓰여 오탐 발생
})


def init_emotion_words(client: QdrantClient, ft_model: "fasttext.FastText._FastText") -> None:
    print("감정사전 로드 중...", flush=True)
    with open(SENTI_WORD_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    def _is_korean(text: str) -> bool:
        return bool(re.search(r"[가-힣]", text))

    negative = [
        e for e in entries
        if int(e["polarity"]) <= -1 and _is_korean(e["word_root"])
    ]
    print(f"  전체: {len(entries)}개, 부정 필터링 후: {len(negative)}개 (이모티콘 제외)", flush=True)

    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_EMOTION_COLLECTION in existing:
        print(f"기존 컬렉션 '{QDRANT_EMOTION_COLLECTION}' 삭제 후 재생성", flush=True)
        client.delete_collection(QDRANT_EMOTION_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_EMOTION_COLLECTION,
        vectors_config=VectorParams(size=EMOTION_VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"컬렉션 '{QDRANT_EMOTION_COLLECTION}' 생성 완료 (벡터 크기: {EMOTION_VECTOR_SIZE})", flush=True)

    # word와 word_root를 모두 index: (word, word_root) 중복 제거 후 적재
    texts_to_index: list[tuple[str, dict]] = []  # (embed_text, payload)
    seen: set[str] = set()
    for entry in negative:
        for text in (entry["word"], entry["word_root"]):
            if text not in seen and _is_korean(text) and entry["word_root"] not in _EMOTION_BLACKLIST:
                seen.add(text)
                texts_to_index.append((text, {
                    "word": entry["word"],
                    "word_root": entry["word_root"],
                    "polarity": entry["polarity"],
                }))

    print(f"FastText 임베딩 생성 및 적재 중... (총 {len(texts_to_index)}개)", flush=True)
    for batch_start in range(0, len(texts_to_index), BATCH_SIZE):
        batch = texts_to_index[batch_start : batch_start + BATCH_SIZE]

        points = []
        for i, (text, payload) in enumerate(batch):
            vec = ft_model.get_word_vector(text)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            points.append(
                PointStruct(
                    id=batch_start + i,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )
        client.upsert(collection_name=QDRANT_EMOTION_COLLECTION, points=points)
        print(f"  {batch_start + len(batch)}/{len(texts_to_index)} 적재 완료", flush=True)

    print("emotion_words 완료!", flush=True)


def main() -> None:
    print(f"FastText 모델 로드 중... ({FASTTEXT_MODEL_PATH})", flush=True)
    ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    print("Qdrant 연결 중...", flush=True)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    init_emotion_words(client, ft_model)

    print("완료!", flush=True)


if __name__ == "__main__":
    main()
