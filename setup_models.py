"""Ensure inference model artifacts are available before serving.

On a fresh deployment no model artifacts are present because they are
not committed to version control. This script downloads the required
model files from the Hugging Face Hub when they are missing, so the API
can load a model on startup. On a read-only host (such as Render) the
files are written to the writable ``/tmp/models_saved`` directory; a
copy already baked into a Docker image at ``/app/models_saved`` is
detected and reused without re-downloading.

It is safe to run repeatedly and always exits successfully, so it can be
chained ahead of the web server in a deployment start command. When a
file cannot be fetched the API still starts and simply returns ``503``
for prediction requests until a model becomes available.

The artifacts are hosted at:
    https://huggingface.co/ArnavS782/sentinel-ai-models

Environment variables:
    HF_MODEL_REPO: Hugging Face Hub repository id holding the artifacts.
        Defaults to ``ArnavS782/sentinel-ai-models``.
    HF_TOKEN: Optional access token for private repositories.
"""

import os

import requests

from src.utils.logger import get_logger
from src.utils.paths import TMP_MODELS_DIR, resolve_model_path

logger = get_logger(__name__)

# Download target: a writable directory on read-only deployment hosts.
_MODELS_DIR = TMP_MODELS_DIR
_HF_REPO = os.getenv("HF_MODEL_REPO", "ArnavS782/sentinel-ai-models")
_MODEL_FILES = ("best_model.pt", "production_model.pkl")
_DOWNLOAD_TIMEOUT_SECONDS = 120
_DOWNLOAD_CHUNK_SIZE = 1 << 16


def file_present(filename: str) -> bool:
    """Return whether a model artifact already exists in a known location.

    Args:
        filename: Name of the artifact file, e.g. ``best_model.pt``.

    Returns:
        ``True`` if the file is present and non-empty in the writable
        download directory, the Docker image directory or the local
        repository directory.
    """
    path = resolve_model_path(filename)
    return path.is_file() and path.stat().st_size > 0


def download_file(filename: str) -> bool:
    """Download a single model artifact directly over HTTPS.

    The Hugging Face ``resolve`` endpoint is fetched with ``requests`` so
    the download URL, HTTP status and byte count are all logged
    explicitly. This makes a failed or partial download plainly visible
    in deployment logs rather than failing silently.

    Args:
        filename: Name of the artifact file to download.

    Returns:
        ``True`` if the file was downloaded and is non-empty, otherwise
        ``False``.
    """
    url = f"https://huggingface.co/{_HF_REPO}/resolve/main/{filename}"
    destination = _MODELS_DIR / filename
    headers = {}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        logger.info("Downloading %s from %s", filename, url)
        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=_DOWNLOAD_TIMEOUT_SECONDS,
        )
        logger.info(
            "Download response for %s: HTTP %d", filename, response.status_code
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Download of {filename} failed with HTTP "
                f"{response.status_code} from {url}"
            )

        bytes_written = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
                    bytes_written += len(chunk)
        logger.info(
            "Wrote %d bytes for %s to %s", bytes_written, filename, destination
        )

        size = destination.stat().st_size
        logger.info("Verified %s on disk: %d bytes", filename, size)
        if size == 0:
            raise RuntimeError(
                f"Downloaded {filename} is empty (0 bytes) at {destination}."
            )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to download %s from %s", filename, url)
        destination.unlink(missing_ok=True)
        return False

    return True


def ensure_demo_data() -> None:
    """Seed the demo fleet when the database holds no engine units.

    On a fresh deployment ``/tmp/sentinel.db`` is empty, which leaves the
    dashboard showing all zeros. This generates a realistic twelve-engine
    fleet so the dashboard always has data to render. Any failure is
    logged and swallowed so the API can still start.
    """
    try:
        from api.database import Machine, SessionLocal, init_db

        init_db()
        session = SessionLocal()
        try:
            machine_count = session.query(Machine).count()
        finally:
            session.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not inspect the database for demo data: %s", exc)
        return

    if machine_count > 0:
        logger.info(
            "Database already has %d engine units; skipping demo seed.",
            machine_count,
        )
        return

    logger.info("Database empty; generating demo fleet data.")
    try:
        from scripts.generate_demo_data import generate

        generate()
        logger.info("Demo fleet data generated.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Demo data generation failed: %s", exc)


def main() -> None:
    """Ensure every required model artifact exists, downloading as needed."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for filename in _MODEL_FILES:
        if file_present(filename):
            logger.info(
                "%s already present at %s; skipping download.",
                filename,
                resolve_model_path(filename),
            )
            continue

        logger.info(
            "%s missing; downloading from Hugging Face repo '%s'",
            filename,
            _HF_REPO,
        )
        if not download_file(filename):
            logger.warning(
                "Could not download %s; the API may return 503 for "
                "predictions until it is available.",
                filename,
            )

    ensure_demo_data()
    logger.info("Model setup complete.")


if __name__ == "__main__":
    main()
