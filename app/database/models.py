"""
Database record models (plain dataclasses — no heavy ORM dependency
needed for a local SQLite app).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnalysisRecord:
    id: Optional[int]
    created_at: str
    word_count: int
    ai_probability: float
    confidence: str
    verdict: str
    model_used: str
    text_preview: str  # first ~200 chars, always stored for history display
    full_text: Optional[str]  # None if privacy setting disables full storage
    result_json: str  # full DetectionResult.to_dict() serialized as JSON
    humanize_result_json: Optional[str] = None
