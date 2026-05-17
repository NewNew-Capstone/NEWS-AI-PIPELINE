import logging

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, NotTranslatable
from deep_translator import GoogleTranslator

LANG_MAP = {
    "KR": ["ko", "ko-KR"],
    "US": ["en", "en-US"],
    "JP": ["ja", "ja-JP"],
}

MAX_CHUNK = 4500
FALLBACK_LANGS = ["ko", "en", "ko-KR", "en-US"]
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


def load_transcript(video_id: str, region_code: str = "US") -> str:
    normalized_region = (region_code or "US").upper()
    preferred = LANG_MAP.get(normalized_region, ["en"])
    langs = list(dict.fromkeys(preferred + FALLBACK_LANGS))
    ytt = YouTubeTranscriptApi()
    try:
        transcript = ytt.fetch(video_id, languages=langs)
    except NoTranscriptFound:
        # 마지막 fallback: 언어 우선순위 없이 유튜브가 제공하는 기본 자막 시도
        transcript = ytt.fetch(video_id)

    if normalized_region != "KR":
        try:
            translated = transcript.translate("ko")
            segments = [t.text for t in translated]
        except NotTranslatable:
            raw = _deduplicate([t.text for t in transcript])
            return _translate_to_korean(raw)
        except Exception:
            logger.warning(
                "transcript.translate failed; fallback to deep translator video_id=%s region=%s",
                video_id,
                normalized_region,
                exc_info=True,
            )
            raw = _deduplicate([t.text for t in transcript])
            return _translate_to_korean(raw)
    else:
        segments = [t.text for t in transcript]

    return _deduplicate(segments)
