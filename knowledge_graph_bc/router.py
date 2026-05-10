from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from knowledge_graph_bc.config import Neo4jConfigError
from knowledge_graph_bc.neo4j_client import get_neo4j_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg", tags=["knowledge_graph"])


def _service_unavailable(exc: Exception) -> HTTPException:
    logger.warning("Neo4j operation failed: %s", exc, exc_info=True)
    return HTTPException(
        status_code=503,
        detail={
            "status": "failed",
            "message": str(exc),
        },
    )


@router.get("/health")
def health() -> dict:
    try:
        return get_neo4j_client().verify_connectivity()
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc


@router.post("/schema/init")
def init_schema() -> dict:
    try:
        return get_neo4j_client().init_schema()
    except (Neo4jConfigError, OSError) as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise _service_unavailable(exc) from exc
