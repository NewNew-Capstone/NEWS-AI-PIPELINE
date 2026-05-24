from __future__ import annotations

SCHEMA_CONSTRAINTS: tuple[str, ...] = (
    """
    CREATE CONSTRAINT video_id_unique IF NOT EXISTS
    FOR (v:Video)
    REQUIRE v.video_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT channel_id_unique IF NOT EXISTS
    FOR (c:Channel)
    REQUIRE c.channel_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT entity_key_unique IF NOT EXISTS
    FOR (e:Entity)
    REQUIRE e.entity_key IS UNIQUE
    """,
    """
    CREATE CONSTRAINT issue_cluster_id_unique IF NOT EXISTS
    FOR (i:IssueCluster)
    REQUIRE i.issue_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT issue_id_unique IF NOT EXISTS
    FOR (i:Issue)
    REQUIRE i.issue_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT issue_node_id_unique IF NOT EXISTS
    FOR (i:IssueNode)
    REQUIRE i.issue_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT analysis_result_id_unique IF NOT EXISTS
    FOR (a:AnalysisResult)
    REQUIRE a.analysis_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT realtime_ingest_job_request_id_unique IF NOT EXISTS
    FOR (j:RealtimeIngestJob)
    REQUIRE j.request_id IS UNIQUE
    """,
)
