from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_graph_bc.config import Neo4jConfig
from knowledge_graph_bc.neo4j_client import Neo4jClient
from knowledge_graph_bc.schema import SCHEMA_CONSTRAINTS


def test_neo4j_client_uses_config_for_driver() -> None:
    config = Neo4jConfig(
        uri="neo4j+s://example.databases.neo4j.io",
        username="neo4j",
        password="secret",
        database="neo4j",
    )

    graph_database = MagicMock()

    with patch(
        "knowledge_graph_bc.neo4j_client._get_graph_database",
        return_value=graph_database,
    ):
        Neo4jClient(config)

    graph_database.driver.assert_called_once_with(
        "neo4j+s://example.databases.neo4j.io",
        auth=("neo4j", "secret"),
        connection_timeout=10.0,
        max_transaction_retry_time=15.0,
    )


def test_schema_constraints_are_idempotent() -> None:
    assert SCHEMA_CONSTRAINTS
    assert all("IF NOT EXISTS" in query for query in SCHEMA_CONSTRAINTS)


def test_schema_constraints_cover_spring_issue_labels() -> None:
    joined_constraints = "\n".join(SCHEMA_CONSTRAINTS)

    assert "FOR (i:IssueCluster)" in joined_constraints
    assert "FOR (i:Issue)" in joined_constraints
    assert "FOR (i:IssueNode)" in joined_constraints


def test_init_schema_runs_all_constraints() -> None:
    config = Neo4jConfig(
        uri="neo4j+s://example.databases.neo4j.io",
        username="neo4j",
        password="secret",
        database="neo4j",
    )

    with patch("knowledge_graph_bc.neo4j_client._get_graph_database", return_value=MagicMock()):
        client = Neo4jClient(config)

    client.execute_write = MagicMock(return_value=[])

    result = client.init_schema()

    assert result["status"] == "ok"
    assert result["constraints_applied"] == len(SCHEMA_CONSTRAINTS)
    assert client.execute_write.call_count == len(SCHEMA_CONSTRAINTS)
