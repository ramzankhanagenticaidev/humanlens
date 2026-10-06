"""
SQLite database layer.

Uses stdlib `sqlite3` directly (no paid database, no heavy ORM). All
queries are parameterized — never string-formatted — to avoid SQL
injection even though inputs are local-only.

Privacy: if `settings.store_full_text` is False, only a short preview
and the numeric results are stored; the full submitted text is
discarded after analysis.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from app.config import settings
from app.database.models import AnalysisRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    ai_probability REAL NOT NULL,
    confidence TEXT NOT NULL,
    verdict TEXT NOT NULL,
    model_used TEXT NOT NULL,
    text_preview TEXT NOT NULL,
    full_text TEXT,
    result_json TEXT NOT NULL,
    humanize_result_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);
"""


class Database:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------------
    def save_analysis(
        self,
        word_count: int,
        ai_probability: float,
        confidence: str,
        verdict: str,
        model_used: str,
        full_text: str,
        result_dict: dict,
        humanize_result_dict: Optional[dict] = None,
    ) -> int:
        preview = (full_text or "")[:200]
        stored_text = full_text if settings.store_full_text else None
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO analyses
                    (created_at, word_count, ai_probability, confidence, verdict,
                     model_used, text_preview, full_text, result_json, humanize_result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    word_count,
                    ai_probability,
                    confidence,
                    verdict,
                    model_used,
                    preview,
                    stored_text,
                    json.dumps(result_dict),
                    json.dumps(humanize_result_dict) if humanize_result_dict else None,
                ),
            )
            return int(cur.lastrowid)

    def attach_humanize_result(self, analysis_id: int, humanize_result_dict: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE analyses SET humanize_result_json = ? WHERE id = ?",
                (json.dumps(humanize_result_dict), analysis_id),
            )

    def list_history(self, limit: int = 50, offset: int = 0) -> List[AnalysisRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, word_count, ai_probability, confidence, verdict,
                       model_used, text_preview, full_text, result_json, humanize_result_json
                FROM analyses
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_analysis(self, analysis_id: int) -> Optional[AnalysisRecord]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, created_at, word_count, ai_probability, confidence, verdict,
                       model_used, text_preview, full_text, result_json, humanize_result_json
                FROM analyses WHERE id = ?
                """,
                (analysis_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def delete_analysis(self, analysis_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            return cur.rowcount > 0

    def stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as total,
                       AVG(ai_probability) as avg_prob,
                       AVG(word_count) as avg_words
                FROM analyses
                """
            ).fetchone()
        return {
            "total_analyses": row["total"] or 0,
            "avg_ai_probability": round(row["avg_prob"], 4) if row["avg_prob"] is not None else None,
            "avg_word_count": round(row["avg_words"], 1) if row["avg_words"] is not None else None,
        }

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AnalysisRecord:
        return AnalysisRecord(
            id=row["id"],
            created_at=row["created_at"],
            word_count=row["word_count"],
            ai_probability=row["ai_probability"],
            confidence=row["confidence"],
            verdict=row["verdict"],
            model_used=row["model_used"],
            text_preview=row["text_preview"],
            full_text=row["full_text"],
            result_json=row["result_json"],
            humanize_result_json=row["humanize_result_json"],
        )


_db_singleton: Optional[Database] = None


def get_db() -> Database:
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = Database()
    return _db_singleton
