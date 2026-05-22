"""
키워드와 영상 목록을 받아 코사인 유사도 기반으로 관련도 높은 영상 ID를 반환.
모델은 최초 1회만 로드 (싱글톤).
"""
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        logger.info("video_ranker: loading ko-sroberta-multitask model")
        _MODEL = SentenceTransformer("jhgan/ko-sroberta-multitask")
        logger.info("video_ranker: model loaded")
    return _MODEL


def rank_by_cosine(
    keyword: str,
    videos: list[dict],  # [{"video_id": str, "title": str, "description": str}]
    top_n: int = 20,
) -> list[str]:
    """
    keyword와 각 영상의 (title + description) 사이의 코사인 유사도를 계산해
    유사도 높은 순으로 top_n개의 video_id를 반환.
    """
    if not videos:
        return []

    model = _get_model()

    # 키워드 임베딩
    keyword_embedding: np.ndarray = model.encode(keyword, convert_to_numpy=True)

    # 영상 텍스트 임베딩 (title + description 합산)
    texts = [
        f"{v.get('title', '')} {v.get('description', '')}".strip()
        for v in videos
    ]
    video_embeddings: np.ndarray = model.encode(texts, convert_to_numpy=True)

    # 코사인 유사도 계산 (정규화 후 내적)
    keyword_norm = keyword_embedding / (np.linalg.norm(keyword_embedding) + 1e-9)
    video_norms = video_embeddings / (
        np.linalg.norm(video_embeddings, axis=1, keepdims=True) + 1e-9
    )
    scores: np.ndarray = video_norms @ keyword_norm  # shape: (N,)

    # 유사도 내림차순 정렬 후 top_n 선택
    top_indices = np.argsort(scores)[::-1][: min(top_n, len(videos))]

    ranked_ids = [videos[i]["video_id"] for i in top_indices]
    logger.info(
        "video_ranker: keyword=%r candidates=%d top_n=%d returned=%d top_score=%.4f",
        keyword,
        len(videos),
        top_n,
        len(ranked_ids),
        float(scores[top_indices[0]]) if len(top_indices) > 0 else 0.0,
    )
    return ranked_ids
