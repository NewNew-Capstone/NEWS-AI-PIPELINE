"""Qdrant emotion_words_v2 컬렉션 생성/적재 스크립트.

사용 예시:
    .venv/bin/python scripts/init_qdrant_emotion_v2.py \
      --input analysis_bc/data/eval/processed/kote/emotion_seed_for_qdrant.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import fasttext
import httpx
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_bc.config import (
    EMOTION_VECTOR_SIZE,
    FASTTEXT_MODEL_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
)

DEFAULT_INPUT = Path("analysis_bc/data/eval/processed/kote/emotion_seed_for_qdrant.jsonl")
DEFAULT_COLLECTION = "emotion_words_v2"
BATCH_SIZE = 100
FASTTEXT_SERVER_URL = os.getenv("FASTTEXT_SERVER_URL", "").rstrip("/")


def _embed_words(
    words: list[str],
    ft_model: "fasttext.FastText._FastText | None",
) -> list[np.ndarray]:
    if FASTTEXT_SERVER_URL:
        resp = httpx.post(
            f"{FASTTEXT_SERVER_URL}/embed",
            json={"words": words},
            timeout=30.0,
        )
        resp.raise_for_status()
        return [np.array(v) for v in resp.json()["vectors"]]

    if ft_model is None:
        raise RuntimeError("FastText model is not loaded")

    return [ft_model.get_word_vector(word) for word in words]


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and populate emotion_words_v2 collection.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"input 파일이 없습니다: {args.input}")

    if FASTTEXT_SERVER_URL:
        print(f"FastText server 사용: {FASTTEXT_SERVER_URL}")
        ft_model = None
    else:
        print(f"FastText 모델 로드: {FASTTEXT_MODEL_PATH}")
        ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    rows = _load_rows(args.input)
    if not rows:
        raise ValueError(f"input 파일이 비어 있습니다: {args.input}")

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    existing = [c.name for c in client.get_collections().collections]
    if args.collection in existing:
        print(f"기존 컬렉션 삭제: {args.collection}")
        client.delete_collection(args.collection)
    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=EMOTION_VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"컬렉션 생성: {args.collection}")

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start:batch_start + BATCH_SIZE]
        words = [str(r.get("embed_text") or r.get("word_root") or r.get("word")).strip() for r in batch]
        vectors = _embed_words(words, ft_model)
        points: list[PointStruct] = []
        for i, (row, vec) in enumerate(zip(batch, vectors)):
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            payload = dict(row)
            payload["embed_text"] = words[i]
            points.append(
                PointStruct(
                    id=batch_start + i,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )
        client.upsert(collection_name=args.collection, points=points)
        print(f"  {batch_start + len(batch)}/{len(rows)} upsert 완료")

    print("완료!")
    print(f"collection: {args.collection}")
    print(f"rows      : {len(rows)}")


if __name__ == "__main__":
    main()
