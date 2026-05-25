"""
영상 목록을 받아 ko-sroberta 임베딩 후 K-means로 군집화.
video_ranker.py와 동일한 모델 싱글톤을 재사용.
"""
import logging
import math

import numpy as np
from sklearn.cluster import KMeans

from content_bc.modules.video_ranker import _get_model

logger = logging.getLogger(__name__)


def cluster_videos(
    videos: list[dict],  # [{"video_id": str, "title": str, "description": str}]
    n_clusters: int | None = None,
) -> list[dict]:  # [{"cluster_id": int, "video_ids": [str]}]
    """
    영상 title+description을 임베딩하여 K-means로 군집화.
    n_clusters가 None이면 영상 수 기반으로 자동 결정.
    """
    if not videos:
        return []

    if len(videos) == 1:
        return [{"cluster_id": 0, "video_ids": [videos[0]["video_id"]]}]

    model = _get_model()

    texts = [
        f"{v.get('title', '')} {v.get('description', '')}".strip()
        for v in videos
    ]
    embeddings: np.ndarray = model.encode(texts, convert_to_numpy=True)

    k = n_clusters or max(2, min(5, math.ceil(len(videos) / 6)))
    k = min(k, len(videos))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    cluster_map: dict[int, list[str]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(int(label), []).append(videos[idx]["video_id"])

    result = [
        {"cluster_id": cid, "video_ids": vids}
        for cid, vids in sorted(cluster_map.items())
    ]

    logger.info(
        "video_clusterer: videos=%d k=%d clusters=%s",
        len(videos),
        k,
        {r["cluster_id"]: len(r["video_ids"]) for r in result},
    )
    return result
