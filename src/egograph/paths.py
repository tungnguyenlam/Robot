"""Filesystem locations. Every path can be overridden with an environment variable,
so the same code runs on amdserver, the thinkbook, and Colab or Kaggle.

    EGO_DATA_ROOT   raw datasets (default: <repo>/data, a symlink to /hdd/Robot/data on amdserver)
    EGO_INDEX_ROOT  cached sequence indexes (default: $EGO_DATA_ROOT/_index)
    EGO_MANO_DIR    folder holding MANO_RIGHT.pkl and MANO_LEFT.pkl
    EGO_RUNS_ROOT   experiment outputs (default: <repo>/runs)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v).expanduser() if v else default


def data_root() -> Path:
    return _env_path("EGO_DATA_ROOT", REPO_ROOT / "data")


def index_root() -> Path:
    return _env_path("EGO_INDEX_ROOT", data_root() / "_index")


def runs_root() -> Path:
    return _env_path("EGO_RUNS_ROOT", REPO_ROOT / "runs")


def mano_dir() -> Path:
    candidates = [
        os.environ.get("EGO_MANO_DIR"),
        REPO_ROOT / "assets" / "mano",
        Path.home() / "Robot" / "stablehand" / "mano_v1_2" / "models",  # amdserver
    ]
    for c in candidates:
        if c and (Path(c).expanduser() / "MANO_RIGHT.pkl").exists():
            return Path(c).expanduser()
    raise FileNotFoundError(
        "MANO models not found. Set EGO_MANO_DIR to a folder that contains MANO_RIGHT.pkl and MANO_LEFT.pkl."
    )
