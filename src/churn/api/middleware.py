"""HTTP middleware for the churn prediction API.

Logs request path, method, status code and wall-clock latency in milliseconds
for every request. Uses the standard ``logging`` module so output integrates
with any handler configured by the caller (structlog, JSON, plain text).
"""

from __future__ import annotations

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status code and latency for every HTTP request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response
