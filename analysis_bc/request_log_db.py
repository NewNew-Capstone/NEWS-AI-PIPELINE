from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoSuchModuleError

from analysis_bc.config import ANALYSIS_DB_URL

_ENGINE: Engine | None = None
logger = logging.getLogger(__name__)


def get_analysis_db_engine() -> Engine | None:
    """운영 로그 적재용 DB 엔진을 지연 생성해 반환한다."""
    global _ENGINE
    if not ANALYSIS_DB_URL:
        return None
    if _ENGINE is None:
        try:
            _ENGINE = create_engine(
                ANALYSIS_DB_URL,
                future=True,
                pool_pre_ping=True,
            )
        except (ModuleNotFoundError, NoSuchModuleError):
            logger.warning(
                "ANALYSIS_DB_URL is set but DB driver is unavailable; request logging disabled",
                exc_info=True,
            )
            return None
    return _ENGINE
