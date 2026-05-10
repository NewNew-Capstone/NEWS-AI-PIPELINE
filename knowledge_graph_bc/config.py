from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class Neo4jConfigError(RuntimeError):
    """Raised when Neo4j Aura configuration is incomplete."""


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    connect_timeout: float = 10.0
    max_retry_time: float = 15.0


def get_neo4j_config() -> Neo4jConfig:
    uri = os.getenv("NEO4J_URI", "").strip()
    username = os.getenv("NEO4J_USERNAME", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"

    missing = [
        name
        for name, value in (
            ("NEO4J_URI", uri),
            ("NEO4J_USERNAME", username),
            ("NEO4J_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        joined = ", ".join(missing)
        raise Neo4jConfigError(f"Missing required Neo4j environment variables: {joined}")

    return Neo4jConfig(
        uri=uri,
        username=username,
        password=password,
        database=database,
        connect_timeout=float(os.getenv("NEO4J_CONNECT_TIMEOUT", "10")),
        max_retry_time=float(os.getenv("NEO4J_MAX_RETRY_TIME", "15")),
    )
