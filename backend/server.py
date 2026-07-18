"""FastAPI server application for AssistArena.

Exposes REST APIs for stadium discovery, copilot consultation, liveness checks,
and serves static web resources under secure headers and rate limiting.
"""

import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend import copilot, database
from backend.limiter import (
    RATE_LIMIT_CEILING,  # noqa: F401
    RateLimitBucket,  # noqa: F401
    RedisRateLimitBucket,  # noqa: F401
    initialize_rate_limiter,
)
from backend.models import (
    ConsultRequest,
    ConsultResponse,
    ServiceHealth,
    StadiumList,
    StadiumSummary,
)

_initialize_rate_limiter = initialize_rate_limiter

logger = logging.getLogger("assistarena")

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

_SECURE_HTTP_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; "
        "object-src 'none'; img-src 'self' data:; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
}

rate_limiter = initialize_rate_limiter()

app = FastAPI(
    title="AssistArena API",
    description="Operations and accessibility spectator copilot for FIFA World Cup 2026.",
    version="1.0.0",
)
app.state.rate_limiter = rate_limiter


@app.middleware("http")
async def inject_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Middleware attaching strict browser security profiles to responses."""
    response = await call_next(request)
    for h, v in _SECURE_HTTP_HEADERS.items():
        response.headers[h] = v
    return response


def require_rate_quota(request: Request) -> None:
    """Raise 429 status code if requesting IP has exceeded allotment limits."""
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.acquire(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait a moment before trying again.",
        )


@app.get("/healthz")
@app.get("/api/healthz")
def check_health() -> ServiceHealth:
    """Return health details and active connection profile mode."""
    return ServiceHealth(
        status="ok",
        llm="live" if copilot.is_api_key_set() else "offline",
    )


def _format_stadium_list(raw_stadiums: list[database.Stadium]) -> StadiumList:
    """Map full records to light summary entities."""
    return StadiumList(
        stadiums=[
            StadiumSummary(
                id=s["id"],
                stadiumName=s["stadiumName"],
                city=s["city"],
                country=s["country"],
                spectatorCapacity=s["spectatorCapacity"],
            )
            for s in raw_stadiums
        ],
    )


@app.get("/api/stadiums")
def list_stadiums() -> StadiumList:
    """List details for all host stadiums."""
    return _format_stadium_list(database.list_all_stadiums())


@app.get("/api/stadiums/search")
def search_stadiums(
    q: Annotated[str, Query(min_length=1, max_length=64)],
) -> StadiumList:
    """Search stadiums matching keyword queries."""
    return _format_stadium_list(database.query_stadiums(q))


@app.get("/api/stadiums/{stadium_id}")
def get_stadium(stadium_id: str) -> database.Stadium:
    """Fetch complete profile record for a stadium."""
    stadium = database.find_stadium_by_id(stadium_id)
    if stadium is None:
        raise HTTPException(status_code=404, detail=f"No stadium found matching {stadium_id!r}.")
    return stadium


@app.post("/api/consult")
def consult_copilot(
    body: ConsultRequest,
    _quota: Annotated[None, Depends(require_rate_quota)],
) -> ConsultResponse:
    """Process consultation queries returning aggregated response payload."""
    res = copilot.consult(
        body.query,
        profile=body.profile.model_dump(),
        history=[turn.model_dump() for turn in body.history],
    )
    return ConsultResponse(
        answer=res.text,
        mode=res.mode,
        stadium_id=body.profile.stadium_id,
    )


@app.post("/api/consult/stream")
def consult_copilot_stream(
    body: ConsultRequest,
    _quota: Annotated[None, Depends(require_rate_quota)],
) -> StreamingResponse:
    """Process consultation queries returning NDJSON data chunks."""
    events = copilot.consult_stream(
        body.query,
        profile=body.profile.model_dump(),
        history=[turn.model_dump() for turn in body.history],
    )
    stadium_id = body.profile.stadium_id

    def _render_frames() -> Iterator[str]:
        """Convert stream events into NDJSON line payloads."""
        try:
            for kind, content in events:
                if kind == "meta":
                    frame = {"type": "meta", "mode": content, "stadium_id": stadium_id}
                else:
                    frame = {"type": "delta", "text": content}
                yield json.dumps(frame, ensure_ascii=False) + "\n"
        except Exception:  # noqa: BLE001
            yield json.dumps({"type": "error"}) + "\n"

    return StreamingResponse(_render_frames(), media_type="application/x-ndjson")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve main static index file."""
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")
