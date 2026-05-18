"""FastAPI application entry point for the SentinelAI service."""

import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict
from collections import deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.database import SessionLocal, init_db
from api.routers import analytics, health, machines, predict
from api.schemas.response import ErrorResponse
from src.utils.config import InferenceConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_RATE_LIMIT = 100
_RATE_WINDOW_SECONDS = 60.0
_STATIC_DIR = Path("static")

app = FastAPI(
    title="SentinelAI — Aviation Turbofan Engine Predictive Maintenance",
    description=(
        "Predictive maintenance API for aviation turbofan engines, "
        "trained on the NASA C-MAPSS turbofan degradation dataset."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_request_log: Dict[str, Deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit_middleware(
    request: Request, call_next: Callable
) -> Any:
    """Reject clients exceeding 100 requests per minute.

    Args:
        request: The incoming request.
        call_next: The downstream handler.

    Returns:
        Either a 429 JSON response or the downstream response.
    """
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    history = _request_log[client]
    while history and now - history[0] > _RATE_WINDOW_SECONDS:
        history.popleft()
    if len(history) >= _RATE_LIMIT:
        logger.warning("Rate limit exceeded for %s", client)
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                detail="Rate limit exceeded.",
                code=429,
                timestamp=datetime.utcnow(),
            ).model_dump(mode="json"),
        )
    history.append(now)
    return await call_next(request)


@app.middleware("http")
async def request_context_middleware(
    request: Request, call_next: Callable
) -> Any:
    """Assign a request_id, then log the request at INFO level.

    A UUID4 ``request_id`` is attached to ``request.state`` so handlers
    and the exception handler can reference it, and echoed back in the
    ``X-Request-ID`` response header for client-side correlation.

    Args:
        request: The incoming request.
        call_next: The downstream handler.

    Returns:
        The downstream response.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "req=%s %s %s -> %d %.1fms",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return a standardized error payload for unhandled exceptions.

    Args:
        request: The request that triggered the exception.
        exc: The raised exception.

    Returns:
        A JSON response wrapping an :class:`ErrorResponse`.
    """
    status_code = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", str(exc)) or "Internal server error."
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "req=%s unhandled error on %s %s: %s",
        request_id,
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            detail=str(detail),
            code=status_code,
            timestamp=datetime.utcnow(),
        ).model_dump(mode="json"),
    )


def seed_demo_data() -> None:
    """Seed a demo fleet when the database holds no engine units.

    On a fresh Render deployment the SQLite database at ``/tmp`` is empty,
    which leaves the dashboard showing all zeros. This seeds a realistic
    twelve-engine fleet — four healthy, four warning and four critical
    units, each with sensor readings, predictions and alerts — so the API
    always has data to serve on a cold start. The generator is idempotent
    and any failure is logged and swallowed so the API still starts.
    """
    try:
        from api.database import Machine

        init_db()
        with SessionLocal() as session:
            machine_count = session.query(Machine).count()
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not inspect the database for demo data: %s", exc)
        return

    if machine_count > 0:
        logger.info(
            "Database already holds %d engine units; skipping demo seed.",
            machine_count,
        )
        return

    logger.info("Database empty; seeding the demo fleet.")
    try:
        from scripts.generate_demo_data import generate

        generate()
        logger.info("Demo fleet seeded: 12 engine units.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Demo data seeding failed: %s", exc)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize the database, seed demo data and warm the model cache."""
    start = time.perf_counter()
    init_db()
    seed_demo_data()

    app.state.model_load_error = None
    try:
        from src.inference.predictor import Predictor
        from src.utils.paths import resolve_model_path

        config = InferenceConfig()
        # Prefer the writable /tmp copy on read-only hosts; fall back to
        # the image or local directory via the shared resolver.
        if not Path(config.model_path).is_file():
            config.model_path = resolve_model_path("best_model.pt")
        predictor = Predictor(config)
        predictor._load_models()
        app.state.predictor = predictor
        logger.info("Prediction models loaded into cache")
    except Exception as exc:  # noqa: BLE001
        app.state.predictor = None
        # Keep the failure reason so /api/v1/health/detailed can surface
        # it without requiring access to the deployment's log stream.
        app.state.model_load_error = traceback.format_exc()
        logger.error("Failed to load prediction models: %s", exc)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info("SentinelAI startup complete in %.1fms", elapsed_ms)


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(machines.router)
app.include_router(analytics.router)

if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
