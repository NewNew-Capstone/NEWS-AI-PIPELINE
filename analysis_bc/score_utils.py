from __future__ import annotations


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_score(value: float, digits: int = 4) -> float:
    return round(clamp_score(value), digits)
