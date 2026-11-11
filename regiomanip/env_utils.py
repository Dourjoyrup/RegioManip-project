"""Small helpers for loading workspace environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def _candidate_env_paths() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    return [
        repo_root / ".env",
        Path.cwd() / ".env",
    ]


def load_workspace_env() -> None:
    """Load simple KEY=VALUE pairs from a workspace `.env` file.

    Existing environment variables keep priority so shell-exported values still
    override repository-local defaults.
    """

    for env_path in _candidate_env_paths():
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ[key] = value
        return
