# 실제 유튜브 자막 가져오는 함수
# region 코드에 따라 언어 선택
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator

LANG_MAP = {
    "KR": ["ko", "ko-KR"],
    "US": ["en", "en-US"],
    "JP": ["ja", "ja-JP"],
}

MAX_CHUNK = 4500


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
    langs = LANG_MAP.get(region_code, ["en"])
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id, languages=langs)

    if region_code != "KR":
        try:
            translated = transcript.translate("ko")
            segments = [t.text for t in translated]
        except Exception:
            raw = _deduplicate([t.text for t in transcript])
            return _translate_to_korean(raw)
    else:
        segments = [t.text for t in transcript]

    return _deduplicate(segments)
