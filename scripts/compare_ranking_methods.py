"""YouTube order=relevance 방식 vs ko-sroberta 임베딩 랭킹 비교 스크립트.

실행:
    python scripts/compare_ranking_methods.py
    python scripts/compare_ranking_methods.py --keyword "반도체 패권 경쟁"
    python scripts/compare_ranking_methods.py --keyword "미중 무역 갈등" --top_n 10

비교 조건:
    A(기존): YouTube order=relevance 20개 → 순서 그대로 사용
    B(개선): YouTube order=date 50개 → ko-sroberta 코사인 유사도 재랭킹 → 상위 20개
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import html

import httpx
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from content_bc.modules.video_ranker import rank_by_cosine

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") or os.getenv("YOUTUBE_API_KEY")
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def youtube_search(keyword: str, order: str, max_results: int) -> list[dict]:
    """YouTube Data API v3 search.list 호출."""
    params = {
        "key": YOUTUBE_API_KEY,
        "q": keyword,
        "part": "snippet",
        "type": "video",
        "order": order,
        "maxResults": max_results,
        "relevanceLanguage": "ko",
    }
    resp = httpx.get(SEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [
        {
            "video_id": item["id"]["videoId"],
            "title": html.unescape(item["snippet"]["title"]),
            "description": html.unescape(item["snippet"]["description"]),
            "channel": item["snippet"]["channelTitle"],
        }
        for item in items
        if item.get("id", {}).get("videoId")
    ]


def run_comparison(keyword: str, top_n: int = 10) -> None:
    if not YOUTUBE_API_KEY:
        print("YOUTUBE_API_KEY 환경변수가 없습니다.")
        sys.exit(1)

    print(f"\n[1/3] YouTube order=relevance {top_n}개 수집 중...")
    relevance_videos = youtube_search(keyword, order="relevance", max_results=top_n)

    print(f"[2/3] YouTube order=date 50개 수집 중...")
    date_videos = youtube_search(keyword, order="date", max_results=50)

    print(f"[3/3] ko-sroberta 재랭킹 중... (모델 첫 실행 시 다운로드 발생)")
    sroberta_top_ids = rank_by_cosine(keyword, date_videos, top_n=top_n)

    date_map = {v["video_id"]: v for v in date_videos}
    sroberta_top = [date_map[vid_id] for vid_id in sroberta_top_ids if vid_id in date_map]

    # ── 유사도 점수 계산 ─────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    kw_vec = model.encode(keyword, convert_to_numpy=True)
    kw_norm = kw_vec / (np.linalg.norm(kw_vec) + 1e-9)

    def sim_score(v: dict) -> float:
        text = f"{v['title']} {v['description']}".strip()
        vec = model.encode(text, convert_to_numpy=True)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return float(vec @ kw_norm)

    # ── 출력 ─────────────────────────────────────────────────────────────────
    col = 44
    sep = "─" * (col * 2 + 22)
    print(f"\n검색어: 「{keyword}」")
    print(f"{'─' * len(sep)}")
    print(f"{'[A] YouTube order=relevance':<{col + 12}}{'[B] ko-sroberta 임베딩 랭킹'}")
    print(f"{'─' * len(sep)}")

    a_scores, b_scores = [], []
    for i, (a, b) in enumerate(zip(relevance_videos, sroberta_top), 1):
        a_sc = sim_score(a)
        b_sc = sim_score(b)
        a_scores.append(a_sc)
        b_scores.append(b_sc)
        a_title = a["title"][:col - 2]
        b_title = b["title"][:col - 2]
        print(f" {i:2}. {a_title:<{col}} ({a_sc:.3f})  →  {b_title:<{col}} ({b_sc:.3f})")

    print(f"{'─' * len(sep)}")
    avg_a, avg_b = np.mean(a_scores), np.mean(b_scores)
    pct = (avg_b - avg_a) / avg_a * 100
    print(f"평균 유사도: {avg_a:.3f} → {avg_b:.3f}  (+{avg_b - avg_a:.3f}, 약 {pct:.0f}% 개선)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="미중 무역 갈등")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()
    run_comparison(args.keyword, args.top_n)
