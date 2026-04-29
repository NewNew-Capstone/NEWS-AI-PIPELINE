from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

from analysis_bc.config import (
    ANALYSIS_REQUEST_LOG_LOCAL_DAILY_SPLIT,
    ANALYSIS_REQUEST_LOG_LOCAL_PATH,
    ANALYSIS_REQUEST_LOG_LOCAL_RETENTION_DAYS,
)
from analysis_bc.request_log_db import get_analysis_db_engine

logger = logging.getLogger(__name__)


class AnalysisRequestLogRepository:
    """분석 요청 문장 로그를 DB에 적재한다."""

    def __init__(
        self,
        engine: Engine | None = None,
        local_log_path: str = ANALYSIS_REQUEST_LOG_LOCAL_PATH,
        use_default_engine: bool = True,
        local_daily_split: bool = ANALYSIS_REQUEST_LOG_LOCAL_DAILY_SPLIT,
        local_retention_days: int = ANALYSIS_REQUEST_LOG_LOCAL_RETENTION_DAYS,
    ) -> None:
        self.engine = get_analysis_db_engine() if engine is None and use_default_engine else engine
        self.local_log_path = local_log_path.strip()
        self.local_daily_split = local_daily_split
        self.local_retention_days = local_retention_days
        self.last_sink = "disabled"

    def enabled(self) -> bool:
        return self.engine is not None or bool(self.local_log_path)

    def build_payload(
        self,
        *,
        source_endpoint: str,
        target_id: int | None,
        transcript_id: int | None,
        language: str | None,
        target_type: str | None,
        country: str | None,
        sentences: list[str],
    ) -> dict:
        cleaned = [s.strip() for s in sentences if isinstance(s, str) and s.strip()]
        return {
            "source_endpoint": source_endpoint,
            "target_id": target_id,
            "transcript_id": transcript_id,
            "language": language,
            "target_type": target_type,
            "country": country,
            "sentence_count": len(cleaned),
            "sentences_json": json.dumps(cleaned, ensure_ascii=False),
            "created_at": datetime.now(timezone.utc),
        }

    def insert_request_log(
        self,
        *,
        source_endpoint: str,
        target_id: int | None,
        transcript_id: int | None,
        language: str | None,
        target_type: str | None,
        country: str | None,
        sentences: list[str],
    ) -> bool:
        payload = self.build_payload(
            source_endpoint=source_endpoint,
            target_id=target_id,
            transcript_id=transcript_id,
            language=language,
            target_type=target_type,
            country=country,
            sentences=sentences,
        )

        if self.engine is None:
            return self._append_local_log(payload)

        stmt = text(
            """
            INSERT INTO analysis_request_log (
                source_endpoint,
                target_id,
                transcript_id,
                language,
                target_type,
                country,
                sentence_count,
                sentences_json,
                created_at
            )
            VALUES (
                :source_endpoint,
                :target_id,
                :transcript_id,
                :language,
                :target_type,
                :country,
                :sentence_count,
                CAST(:sentences_json AS jsonb),
                :created_at
            )
            """
        )
        try:
            with self.engine.begin() as conn:
                conn.execute(stmt, payload)
            self.last_sink = "db"
            return True
        except Exception:
            # DB 실패 시 로컬 파일 로깅으로 폴백
            logger.warning("analysis request DB logging failed; fallback to local file", exc_info=True)
            ok = self._append_local_log(payload)
            self.last_sink = "local_fallback" if ok else "failed"
            return ok

    def _append_local_log(self, payload: dict) -> bool:
        if not self.local_log_path:
            self.last_sink = "disabled"
            return False

        local_path = self._resolve_local_log_path(payload.get("created_at"))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._prune_old_local_logs(base_path=Path(self.local_log_path))

        sentences: list[str] = []
        raw_sentences = payload.get("sentences_json")
        if isinstance(raw_sentences, str):
            try:
                decoded = json.loads(raw_sentences)
                if isinstance(decoded, list):
                    sentences = [s for s in decoded if isinstance(s, str)]
            except json.JSONDecodeError:
                sentences = []

        created_at = payload.get("created_at")
        created_at_str = created_at.isoformat() if hasattr(created_at, "isoformat") else None
        line = {
            "created_at": created_at_str,
            "source_endpoint": payload.get("source_endpoint"),
            "target_id": payload.get("target_id"),
            "transcript_id": payload.get("transcript_id"),
            "language": payload.get("language"),
            "target_type": payload.get("target_type"),
            "country": payload.get("country"),
            "sentence_count": payload.get("sentence_count"),
            "sentences": [{"sentence_text": s} for s in sentences],
        }
        with local_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.last_sink = "local"
        return True

    def _resolve_local_log_path(self, created_at: object) -> Path:
        base_path = Path(self.local_log_path)
        if not self.local_daily_split:
            return base_path
        if not hasattr(created_at, "strftime"):
            return base_path
        day = created_at.strftime("%Y%m%d")
        return base_path.with_name(f"{base_path.stem}-{day}{base_path.suffix}")

    def _prune_old_local_logs(self, base_path: Path) -> None:
        if not self.local_daily_split or self.local_retention_days <= 0:
            return
        cutoff = datetime.now(timezone.utc).date().toordinal() - self.local_retention_days
        pattern = f"{base_path.stem}-*{base_path.suffix}"
        for path in base_path.parent.glob(pattern):
            parts = path.stem.rsplit("-", 1)
            if len(parts) != 2:
                continue
            day = parts[1]
            if len(day) != 8 or not day.isdigit():
                continue
            try:
                file_day = datetime.strptime(day, "%Y%m%d").date().toordinal()
            except ValueError:
                continue
            if file_day < cutoff:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("failed to prune local request log file: %s", path, exc_info=True)
