import logging
import os
import re
import tempfile
import threading
import time
import yt_dlp
from deep_translator import GoogleTranslator

# 백그라운드 배치용: 동시 1개 + 3초 딜레이
_TRANSCRIPT_SEMAPHORE = threading.Semaphore(1)
# 단건 우선순위용: 백그라운드와 독립적으로 실행
_PRIORITY_SEMAPHORE = threading.Semaphore(1)

LANG_MAP = {
    "KR": ["ko", "ko-KR"],
    "US": ["en", "en-US"],
    "JP": ["ja", "ja-JP"],
}

MAX_CHUNK = 4500
logger = logging.getLogger(__name__)


def _deduplicate(segments: list[str]) -> str:
    if not segments:
        return ""
    result = segments[0].split()
    for seg in segments[1:]:
        words = seg.split()
        max_overlap = min(len(result), len(words))
        overlap = 0
        for k in range(max_overlap, 0, -1):
            if result[-k:] == words[:k]:
                overlap = k
                break
        result.extend(words[overlap:])
    return " ".join(result)


def _translate_to_korean(text: str) -> str:
    chunks = [text[i:i + MAX_CHUNK] for i in range(0, len(text), MAX_CHUNK)]
    translated = [GoogleTranslator(source="auto", target="ko").translate(chunk) for chunk in chunks]
    return " ".join(translated)


def _parse_vtt(vtt_text: str) -> list[str]:
    vtt_text = re.sub(r"<[^>]+>", "", vtt_text)
    lines = vtt_text.splitlines()
    segments = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        segments.append(line)
    return segments


def load_transcript(video_id: str, region_code: str = "US", priority: bool = False) -> str:
    semaphore = _PRIORITY_SEMAPHORE if priority else _TRANSCRIPT_SEMAPHORE
    with semaphore:
        if not priority:
            time.sleep(3)  # 백그라운드 배치만 딜레이, 우선순위 요청은 즉시 실행

        langs = LANG_MAP.get(region_code, ["en"])
        url = f"https://www.youtube.com/watch?v={video_id}"

        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = os.environ.get("YOUTUBE_COOKIE_PATH", "")
            ydl_opts = {
                "skip_download": True,
                "writeautomaticsub": True,
                "writesubtitles": True,
                "subtitleslangs": langs,
                "subtitlesformat": "vtt",
                "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "ignore_no_formats_error": True,
                "sleep_interval": 2,
                "sleep_interval_requests": 1,
            }
            if cookie_path and os.path.exists(cookie_path):
                ydl_opts["cookiefile"] = cookie_path
                logger.info("쿠키 파일 적용: %s", cookie_path)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            vtt_files = [f for f in os.listdir(tmpdir) if f.endswith(".vtt")]
            if not vtt_files:
                raise ValueError(f"자막 없음: video_id={video_id}, region={region_code}")

            selected = None
            for lang in langs:
                for f in vtt_files:
                    if lang in f:
                        selected = f
                        break
                if selected:
                    break
            if not selected:
                selected = vtt_files[0]

            with open(os.path.join(tmpdir, selected), "r", encoding="utf-8") as f:
                raw_text = f.read()

        segments = _parse_vtt(raw_text)
        text = _deduplicate(segments)

        if region_code != "KR":
            try:
                return _translate_to_korean(text)
            except Exception:
                return text

        return text
