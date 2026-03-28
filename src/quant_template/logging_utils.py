from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str, run_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to console and to run_dir/run.log."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)

    fh = logging.FileHandler(run_path / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
