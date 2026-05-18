"""API key authentication for the SentinelAI service.

Authentication is enabled only when the ``SENTINEL_API_KEY`` environment
variable is set. When it is unset (local development, the public demo),
requests pass through so the dashboard keeps working without a key. When
it is set, every protected endpoint requires a matching ``X-API-Key``
header and rejects missing or wrong keys with HTTP 401.
"""

import hmac
import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.utils.logger import get_logger

logger = get_logger(__name__)

_API_KEY_HEADER = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_HEADER, auto_error=False)


def verify_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Validate the ``X-API-Key`` header against ``SENTINEL_API_KEY``.

    Args:
        api_key: The value of the incoming ``X-API-Key`` header, if any.

    Raises:
        HTTPException: With status 401 when authentication is enabled and
            the supplied key is missing or does not match.
    """
    expected = os.getenv("SENTINEL_API_KEY", "").strip()
    if not expected:
        # No key configured: authentication is disabled (demo / local).
        return

    if not api_key:
        logger.warning("Rejected request: missing %s header", _API_KEY_HEADER)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
            headers={"WWW-Authenticate": _API_KEY_HEADER},
        )

    # Constant-time comparison to avoid leaking the key via timing.
    if not hmac.compare_digest(api_key, expected):
        logger.warning("Rejected request: invalid %s header", _API_KEY_HEADER)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": _API_KEY_HEADER},
        )
