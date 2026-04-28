from analysis_bc.enums import (
    BiasKeywordType,
    EvidenceType,
    JobStatus,
    JobType,
    SentenceLabelType,
    TargetType,
)


def test_job_type_values() -> None:
    assert JobType.TRANSCRIPT_FETCH == "TRANSCRIPT_FETCH"
    assert JobType.VIDEO_SUMMARY == "VIDEO_SUMMARY"
    assert JobType.VIDEO_BIAS_ANALYSIS == "VIDEO_BIAS_ANALYSIS"
    assert JobType.ISSUE_COMPARE_ANALYSIS == "ISSUE_COMPARE_ANALYSIS"
    assert len(list(JobType)) == 4


def test_job_status_values() -> None:
    assert JobStatus.PENDING == "PENDING"
    assert JobStatus.RUNNING == "RUNNING"
    assert JobStatus.SUCCESS == "SUCCESS"
    assert JobStatus.FAILED == "FAILED"
    assert len(list(JobStatus)) == 4


def test_target_type_values() -> None:
    assert TargetType.YOUTUBE_VIDEO == "YOUTUBE_VIDEO"
    assert TargetType.ISSUE_CLUSTER == "ISSUE_CLUSTER"
    assert TargetType.ISSUE_CLUSTER_VIDEO == "ISSUE_CLUSTER_VIDEO"
    assert len(list(TargetType)) == 3


def test_sentence_label_type_values() -> None:
    assert SentenceLabelType.FACT_LIKE == "FACT_LIKE"
    assert SentenceLabelType.OPINION_LIKE == "OPINION_LIKE"
    assert SentenceLabelType.EMOTIONALLY_LOADED == "EMOTIONALLY_LOADED"
    assert len(list(SentenceLabelType)) == 3


def test_evidence_type_values() -> None:
    assert EvidenceType.OPINION == "OPINION"
    assert EvidenceType.EMOTION == "EMOTION"
    assert EvidenceType.SPECULATION == "SPECULATION"
    assert len(list(EvidenceType)) == 3


def test_bias_keyword_type_values() -> None:
    
    assert BiasKeywordType.EMOTION == "EMOTION"
    assert BiasKeywordType.FRAME == "FRAME"
    assert BiasKeywordType.TOPIC == "TOPIC"
    assert len(list(BiasKeywordType)) == 3
