"""Filesystem path resolution for model artifacts.

On a managed deployment such as Render the application directory can be
read-only, so model artifacts are downloaded at startup to a writable
``/tmp`` location. This module resolves an artifact to the first
directory that actually contains it, searching the writable download
directory first, then the directory baked into a Docker image, then the
repository-local directory used during development.
"""

from pathlib import Path

# Writable download target used on read-only deployment hosts (Render).
TMP_MODELS_DIR = Path("/tmp/models_saved")
# Directory baked into the Docker image (the container WORKDIR is /app).
APP_MODELS_DIR = Path("/app/models_saved")
# Repository-relative directory used for local development.
LOCAL_MODELS_DIR = Path("models_saved")

# Order in which an existing artifact is searched for.
MODEL_SEARCH_DIRS = (TMP_MODELS_DIR, APP_MODELS_DIR, LOCAL_MODELS_DIR)


def resolve_model_path(filename: str) -> Path:
    """Resolve a model artifact to the first directory that contains it.

    Args:
        filename: Name of the artifact file, e.g. ``best_model.pt``.

    Returns:
        The first existing path across :data:`MODEL_SEARCH_DIRS`. When the
        file is absent everywhere, the ``/tmp`` download path is returned
        so callers report a consistent, writable location.
    """
    for directory in MODEL_SEARCH_DIRS:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return TMP_MODELS_DIR / filename
