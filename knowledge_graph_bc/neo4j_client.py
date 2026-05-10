from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from knowledge_graph_bc.config import Neo4jConfig, Neo4jConfigError, get_neo4j_config
from knowledge_graph_bc.schema import SCHEMA_CONSTRAINTS


def _get_graph_database() -> Any:
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as exc:
        raise Neo4jConfigError(
            "Missing Python dependency 'neo4j'. Run pip install -r requirements.txt."
        ) from exc
    return GraphDatabase


class Neo4jClient:
    def __init__(self, config: Neo4jConfig | None = None) -> None:
        self.config = config or get_neo4j_config()
        graph_database = _get_graph_database()
        self.driver: Any = graph_database.driver(
            self.config.uri,
            auth=(self.config.username, self.config.password),
            connection_timeout=self.config.connect_timeout,
            max_transaction_retry_time=self.config.max_retry_time,
        )

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> dict[str, Any]:
        records = self.execute_read("RETURN 1 AS ok")
        ok = records[0]["ok"] if records else None
        return {
            "status": "ok" if ok == 1 else "failed",
            "database": self.config.database,
            "ok": ok,
        }

    def execute_read(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, dict(parameters or {}))
            return [record.data() for record in result]

    def execute_write(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, dict(parameters or {}))
            return [record.data() for record in result]

    def init_schema(self) -> dict[str, Any]:
        for query in SCHEMA_CONSTRAINTS:
            self.execute_write(query)
        return {
            "status": "ok",
            "database": self.config.database,
            "constraints_applied": len(SCHEMA_CONSTRAINTS),
        }


@lru_cache
def get_neo4j_client() -> Neo4jClient:
    return Neo4jClient()
