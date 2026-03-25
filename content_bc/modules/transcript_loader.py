# 실제 유튜브 자막 가져오는 함수 
# region 코드에 따라 언어 선택 
from youtube_transcript_api import YouTubeTranscriptApi

LANG_MAP = {
    "KR": ["ko", "ko-KR"],
    "US": ["en", "en-US"],
    "JP": ["ja", "ja-JP"],
}


def load_transcript(video_id: str, region_code: str = "US") -> str:
    langs = LANG_MAP.get(region_code, ["en"])
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id, languages=langs)
    return " ".join([t.text for t in transcript])
