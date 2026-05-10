from __future__ import annotations

import pytest

from knowledge_graph_bc.config import Neo4jConfigError, get_neo4j_config


def test_neo4j_config_requires_core_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    with pytest.raises(Neo4jConfigError) as exc:
        get_neo4j_config()

    message = str(exc.value)
    assert "NEO4J_URI" in message
    assert "NEO4J_USERNAME" in message
    assert "NEO4J_PASSWORD" in message


def test_neo4j_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://example.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")

    config = get_neo4j_config()

    assert config.uri == "neo4j+s://example.databases.neo4j.io"
    assert config.username == "neo4j"
    assert config.password == "secret"
    assert config.database == "neo4j"
