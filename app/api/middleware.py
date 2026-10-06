"""
Production middleware for the FastAPI app.

Three concerns, kept deliberately simple and dependency-free:

1. API key auth — optional. If HUMANLENS_API_KEY is set, every request
   (except /health) must send it via the `X-API-Key` header. Leave the
   env var unset for local single-user development and the check is
   skipped entirely.

2. Rate limiting — a simple in-memory sliding-window limiter keyed by
   API key (if present) or client IP. This is intentionally NOT
   distributed/Redis-backed: it's correct for a single-process
   deployment (including the Docker setup in this repo) and resets on
   restart. For a multi-instance production deployment behind a load
   balancer, swap this for a shared store (Redis, etc.) — the
   `RateLimiter` class is isolated so that's a one-file change.

3. Request ID + structured access logging — every request gets a short
   ID that's included in logs and returned as `X-Request-ID`, so a
   single request's log lines can be traced end to end.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import settings

logger = logging.getLogger("humanlens.api")

# Paths that never require an API key or count against rate limits.
EXEMPT_PATHS = {"/health"}


class RateLimiter:
    """Fixed-client sliding-window counter. Not thread-safe across
    multiple worker processes (see module docstring); fine for a single
    uvicorn worker, which is what `run.py` and the provided Dockerfile use."""

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, client_key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        window_start = now - 60.0
        hits = self._hits[client_key]
        while hits and hits[0] < window_start:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True


_rate_limiter = RateLimiter(settings.rate_limit_per_minute)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID and logs method/path/status/duration for every call."""

    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        start = time.monotonic()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = (time.monotonic() - start) * 1000
            status = response.status_code if response else 500
            logger.info(
                "%s %s -> %s (%.1fms) [%s]",
                request.method, request.url.path, status, duration_ms, request_id,
            )
            if response is not None:
                response.headers["X-Request-ID"] = request_id


class SecurityMiddleware(BaseHTTPMiddleware):
    """API key check + rate limiting, in that order. Both are skipped for
    EXEMPT_PATHS so uptime monitors don't need a key or count against limits."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        if settings.api_key:
            provided = request.headers.get("X-API-Key", "")
            if provided != settings.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid API key. Send it via the X-API-Key header."},
                )
            client_key = provided
        else:
            client_key = request.client.host if request.client else "unknown"

        if not _rate_limiter.allow(client_key):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded ({settings.rate_limit_per_minute} requests/minute). "
                              "Please slow down and try again shortly."
                },
            )

        return await call_next(request)
