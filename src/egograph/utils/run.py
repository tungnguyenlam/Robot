"""Experiment run directories: ``runs/<YYYYMMDD-HHMMSS>-<name>/`` holds a config
snapshot, the git state, and the run's outputs. WORKLOG entries link to these."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from egograph.paths import REPO_ROOT, runs_root


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def create_run_dir(name: str, cfg: dict[str, Any] | None = None) -> Path:
    run_dir = runs_root() / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    if cfg is not None:
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "git.json").write_text(json.dumps(git_state(), indent=2))
    return run_dir
