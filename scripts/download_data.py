"""Download and verify the NASA C-MAPSS turbofan degradation dataset.

The script first attempts an automated download through the Kaggle CLI.
If the CLI is unavailable it prints manual instructions, then verifies
that the eight expected dataset files are present and reports their
sizes and row counts.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

_RAW_DIR = Path("data/raw")
_KAGGLE_DATASET = "behrad3d/nasa-cmaps"
_EXPECTED_FILES: List[str] = [
    f"{prefix}_FD00{index}.txt"
    for prefix in ("train", "test", "RUL")
    for index in range(1, 5)
]
_REQUIRED_FILES: List[str] = [
    f"{prefix}_FD00{index}.txt"
    for prefix in ("train", "RUL")
    for index in range(1, 5)
]


def _kaggle_available() -> bool:
    """Return whether the Kaggle CLI is installed and on the PATH.

    Returns:
        ``True`` if the ``kaggle`` executable can be located.
    """
    return shutil.which("kaggle") is not None


def _download_with_kaggle() -> bool:
    """Attempt to download and unzip the dataset via the Kaggle CLI.

    Returns:
        ``True`` if the download command completed successfully.
    """
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        _KAGGLE_DATASET,
        "-p",
        str(_RAW_DIR),
        "--unzip",
    ]
    logger.info("Running: %s", " ".join(command))
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        logger.error("Kaggle download failed to start: %s", exc)
        return False
    if result.returncode != 0:
        logger.error(
            "Kaggle download exited with code %d: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return False
    logger.info("Kaggle download completed")
    return True


def _print_manual_instructions() -> None:
    """Log step-by-step instructions for a manual dataset download."""
    logger.warning("Kaggle CLI unavailable — manual download required.")
    logger.warning("1. Install the CLI: pip install kaggle")
    logger.warning(
        "2. Create an API token at https://www.kaggle.com/settings "
        "and place kaggle.json in ~/.kaggle/"
    )
    logger.warning(
        "3. Run: kaggle datasets download -d %s -p %s --unzip",
        _KAGGLE_DATASET,
        _RAW_DIR,
    )
    logger.warning(
        "   Alternatively download the C-MAPSS data from the NASA "
        "Prognostics Data Repository and extract it into %s",
        _RAW_DIR,
    )


def _count_rows(path: Path) -> int:
    """Count the number of lines in a text file.

    Args:
        path: Path to the file.

    Returns:
        The number of newline-delimited rows.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def verify() -> bool:
    """Verify the expected dataset files and report their statistics.

    Returns:
        ``True`` if every required train and RUL file is present.
    """
    logger.info("Verifying dataset files in %s", _RAW_DIR)
    present: List[str] = []
    for name in _EXPECTED_FILES:
        path = _RAW_DIR / name
        if not path.is_file():
            logger.warning("Missing file: %s", path)
            continue
        size_kb = path.stat().st_size / 1024.0
        rows = _count_rows(path)
        present.append(name)
        logger.info("%-18s %9.1f KB %8d rows", name, size_kb, rows)

    missing_required = [
        name for name in _REQUIRED_FILES if name not in present
    ]
    if missing_required:
        logger.error("Missing required files: %s", ", ".join(missing_required))
        return False
    logger.info("All %d required dataset files verified", len(_REQUIRED_FILES))
    return True


def main() -> None:
    """Run the download workflow and verify the resulting files."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    if _kaggle_available():
        if not _download_with_kaggle():
            _print_manual_instructions()
    else:
        _print_manual_instructions()

    verify()


if __name__ == "__main__":
    main()
