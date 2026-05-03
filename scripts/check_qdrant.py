"""Qdrant 컬렉션 적재 확인 스크립트.

사용법:
    python scripts/check_qdrant.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from analysis_bc.config import (
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)

REQUIRED_COLLECTIONS = [QDRANT_EMOTION_COLLECTION]
OPTIONAL_COLLECTIONS = ["anonymous_patterns"]


def check(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}

    for name in REQUIRED_COLLECTIONS:
        if name not in existing:
            print(f"[MISSING] {name} — 컬렉션 없음 (init_qdrant.py 먼저 실행)")
            continue
        _print_collection(client, name)

    for name in OPTIONAL_COLLECTIONS:
        if name in existing:
            _print_collection(client, name, optional=True)
        else:
            print(f"\n[SKIP] {name}: optional collection not present")


def _print_collection(
    client: QdrantClient,
    name: str,
    *,
    optional: bool = False,
) -> None:
    info = client.get_collection(name)
    count = info.points_count
    vector_size = info.config.params.vectors.size
    prefix = "OK/OPTIONAL" if optional else "OK"
    print(f"\n[{prefix}] {name}: {count} points, vector_size={vector_size}")

    results, _ = client.scroll(collection_name=name, limit=1, with_payload=True)
    if results:
        print(f"  sample payload: {json.dumps(results[0].payload, ensure_ascii=False)}")


def main() -> None:
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()  # 연결 확인
    except Exception as e:
        print(f"[ERROR] Qdrant 연결 실패 ({QDRANT_HOST}:{QDRANT_PORT}): {e}")
        sys.exit(1)

    print(f"Qdrant 연결 성공 ({QDRANT_HOST}:{QDRANT_PORT})")
    check(client)


if __name__ == "__main__":
    main()
