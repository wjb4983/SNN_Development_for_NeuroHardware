from __future__ import annotations

import os
from pathlib import Path

DEFAULT_WINDOWS_KEY_PATH = Path(r"C:\Users\wbott\.stoptions_analyzer\api_key.txt")
DEFAULT_POSIX_KEY_PATH = Path.home() / ".stoptions_analyzer" / "api_key.txt"


def load_massive_api_key(explicit_path: str | Path | None = None) -> str:
    """Load MASSIVE API key from env var or api_key.txt file."""

    direct = os.getenv("MASSIVE_API_KEY", "").strip()
    if direct:
        return direct

    candidates: list[Path] = []
    file_env = os.getenv("MASSIVE_API_KEY_FILE", "").strip()
    if file_env:
        candidates.append(Path(file_env))
    if explicit_path:
        candidates.append(Path(explicit_path))

    candidates.extend([DEFAULT_WINDOWS_KEY_PATH, DEFAULT_POSIX_KEY_PATH])

    for path in candidates:
        expanded = path.expanduser()
        if expanded.exists() and expanded.is_file():
            key = expanded.read_text(encoding="utf-8").strip()
            if key:
                return key

    raise FileNotFoundError(
        "MASSIVE API key not found. Set MASSIVE_API_KEY or place key in "
        f"{DEFAULT_WINDOWS_KEY_PATH} (Windows) / {DEFAULT_POSIX_KEY_PATH} (POSIX)."
    )
