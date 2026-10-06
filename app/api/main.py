"""
FastAPI backend for HumanLens.

Endpoints:
    POST   /analyze
    POST   /analyze/paragraphs
    POST   /humanize
    GET    /history
    GET    /history/{id}
    DELETE /history/{id}
    GET    /health
    GET    /stats

All input is validated through Pydantic schemas. Text length is capped
(see app/config.py) to avoid pathological inputs. No shell/eval/exec of
any user input anywhere in this file or anything it calls.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.api.middleware import RequestContextMiddleware, SecurityMiddleware
from app.config import settings
from app.database.db import get_db
from app.detector.detector import get_detector
from app.humanizer.engine import get_humanizer_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("humanlens.api")

APP_VERSION = "1.1.0"
IS_PRODUCTION = settings.environment.lower() == "production"

app = FastAPI(
    title="HumanLens API",
    description="AI writing forensics + natural writing assistant — fully local, no paid APIs.",
    version=APP_VERSION,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
)

# Order matters: Starlette applies middleware in reverse of add order, so
# CORS (added last) runs outermost — it needs to see every response,
# including 401/429s from SecurityMiddleware, to attach CORS headers.
app.add_middleware(RequestContextMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.api_key:
    logger.info("API key authentication is ENABLED.")
else:
    logger.warning(
        "API key authentication is DISABLED (HUMANLENS_API_KEY not set). "
        "Fine for local development; set HUMANLENS_API_KEY before exposing this API beyond localhost."
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    sensitivity: Optional[float] = Field(None, ge=0.0, le=1.0)
    save_to_history: bool = True

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if len(v) > settings.max_chars_for_analysis:
            raise ValueError(
                f"Text is too large. Maximum allowed length is {settings.max_chars_for_analysis} characters."
            )
        return v


class HumanizeRequest(BaseModel):
    text: str
    use_ollama: Optional[bool] = None
    ollama_model: Optional[str] = Field(
        None, description="Override the configured Ollama model for this request, e.g. 'qwen2.5-coder:3b'."
    )
    analysis_id: Optional[int] = None  # attach result to an existing history entry

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if len(v) > settings.max_chars_for_analysis:
            raise ValueError(
                f"Text is too large. Maximum allowed length is {settings.max_chars_for_analysis} characters."
            )
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    detector = get_detector()
    return {
        "status": "ok",
        "model_trained": detector.model.is_trained,
    }


@app.get("/version")
def version():
    return {
        "version": APP_VERSION,
        "environment": settings.environment,
        "api_key_required": bool(settings.api_key),
        "rate_limit_per_minute": settings.rate_limit_per_minute,
    }


@app.get("/stats")
def stats():
    return get_db().stats()


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        detector = get_detector()
        result = detector.analyze(req.text, sensitivity=req.sensitivity)
        result_dict = result.to_dict()

        analysis_id = None
        if req.save_to_history:
            analysis_id = get_db().save_analysis(
                word_count=result.num_words,
                ai_probability=result.ai_probability,
                confidence=result.confidence,
                verdict=result.verdict,
                model_used=result.model_used,
                full_text=req.text,
                result_dict=result_dict,
            )
        result_dict["analysis_id"] = analysis_id
        return result_dict
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("analyze failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/analyze/paragraphs")
def analyze_paragraphs(req: AnalyzeRequest):
    try:
        detector = get_detector()
        result = detector.analyze(req.text, sensitivity=req.sensitivity)
        return {"paragraphs": [p.__dict__ for p in result.paragraphs]}
    except Exception as exc:  # pragma: no cover
        logger.exception("paragraph analysis failed")
        raise HTTPException(status_code=500, detail=f"Paragraph analysis failed: {exc}") from exc


@app.get("/ollama/status")
def ollama_status():
    """Diagnostic endpoint: is an Ollama server reachable, and what
    models are pulled locally? Useful for debugging why local-LLM
    humanization isn't working."""
    from app.humanizer.ollama_client import get_ollama_client

    client = get_ollama_client()
    available = client.is_available()
    return {
        "host": client.host,
        "reachable": available,
        "configured_default_model": client.model,
        "models_pulled": client.list_models() if available else [],
    }


@app.post("/humanize")
def humanize(req: HumanizeRequest):
    try:
        engine = get_humanizer_engine()
        result = engine.humanize(req.text, use_ollama=req.use_ollama, ollama_model=req.ollama_model)
        result_dict = result.to_dict()

        if req.analysis_id is not None:
            try:
                get_db().attach_humanize_result(req.analysis_id, result_dict)
            except Exception:
                logger.warning("Could not attach humanize result to analysis_id=%s", req.analysis_id)

        return result_dict
    except Exception as exc:  # pragma: no cover
        logger.exception("humanize failed")
        raise HTTPException(status_code=500, detail=f"Humanization failed: {exc}") from exc


@app.get("/history")
def history(limit: int = 50, offset: int = 0):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    records = get_db().list_history(limit=limit, offset=offset)
    return {"items": [_record_summary(r) for r in records]}


@app.get("/history/{analysis_id}")
def history_detail(analysis_id: int):
    record = get_db().get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    import json

    return {
        "id": record.id,
        "created_at": record.created_at,
        "word_count": record.word_count,
        "ai_probability": record.ai_probability,
        "confidence": record.confidence,
        "verdict": record.verdict,
        "model_used": record.model_used,
        "text_preview": record.text_preview,
        "full_text": record.full_text,
        "result": json.loads(record.result_json),
        "humanize_result": json.loads(record.humanize_result_json) if record.humanize_result_json else None,
    }


@app.delete("/history/{analysis_id}")
def delete_history(analysis_id: int):
    deleted = get_db().delete_analysis(analysis_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"deleted": True, "id": analysis_id}


def _record_summary(record) -> dict:
    return {
        "id": record.id,
        "created_at": record.created_at,
        "word_count": record.word_count,
        "ai_probability": record.ai_probability,
        "confidence": record.confidence,
        "verdict": record.verdict,
        "model_used": record.model_used,
        "text_preview": record.text_preview,
    }
