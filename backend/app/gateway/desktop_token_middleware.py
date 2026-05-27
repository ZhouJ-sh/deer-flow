from __future__ import annotations

import os
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

DESKTOP_TOKEN_HEADER = "X-DeerFlow-Desktop-Token"


class DesktopTokenMiddleware(BaseHTTPMiddleware):
    """Reject direct Gateway API calls in packaged desktop mode.

    This is a transport guard only. It does not replace DeerFlow user auth.
    """

    def __init__(self, app: ASGIApp, *, enabled: bool = False, token_file: str = "") -> None:
        super().__init__(app)
        self.enabled = enabled
        self.token_file = Path(token_file) if token_file else None
        self.expected_token = self._load_expected_token() if enabled else ""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if not self.enabled or self._is_exempt(request):
            return await call_next(request)

        supplied = request.headers.get(DESKTOP_TOKEN_HEADER, "")
        if not supplied or not self._token_matches(supplied):
            return JSONResponse(status_code=403, content={"detail": "Desktop token required"})

        return await call_next(request)

    def _token_matches(self, supplied: str) -> bool:
        return secrets.compare_digest(self.expected_token.encode(), supplied.encode())

    @staticmethod
    def _is_exempt(request: Request) -> bool:
        path = request.url.path.rstrip("/") or "/"
        return request.method == "OPTIONS" or path == "/health"

    def _load_expected_token(self) -> str:
        if self.token_file is None:
            raise RuntimeError("Desktop token file is required when DEER_FLOW_DESKTOP=1")
        try:
            stat_result = self.token_file.stat()
        except OSError as exc:
            raise RuntimeError(f"Desktop token file is unreadable: {self.token_file}") from exc

        if os.name != "nt" and stat_result.st_mode & 0o077:
            raise RuntimeError("Desktop token file permissions must not allow group/other access")

        try:
            token = self.token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"Desktop token file is unreadable: {self.token_file}") from exc
        if not token:
            raise RuntimeError("Desktop token file is empty")
        return token
