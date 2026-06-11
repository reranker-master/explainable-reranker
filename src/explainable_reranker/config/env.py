from __future__ import annotations

import os
from pathlib import Path


def load_project_dotenv(start: Path | None = None) -> list[Path]:
    """Load `.env` and `.env.local` from the nearest project directory.

    Existing process environment variables win. Values from `.env.local` win
    over `.env` for variables that were not already exported by the shell.
    This intentionally implements only the small dotenv subset we need:
    `KEY=value`, optional `export`, comments, and quoted values.
    """

    root = _find_env_root(start or Path.cwd())
    if root is None:
        return []

    files = [path for path in (root / ".env", root / ".env.local") if path.exists()]
    original_env = set(os.environ)
    merged: dict[str, str] = {}
    for path in files:
        merged.update(_parse_dotenv(path))

    for key, value in merged.items():
        if key not in original_env:
            os.environ[key] = value
    return files


def _find_env_root(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        if (directory / ".env").exists() or (directory / ".env.local").exists():
            return directory
    return None


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key or not _valid_key(key):
            continue
        values[key] = _clean_value(value.strip())
    return values


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    comment = value.find(" #")
    if comment != -1:
        value = value[:comment].rstrip()
    return value


def _valid_key(key: str) -> bool:
    return key.replace("_", "").isalnum() and not key[0].isdigit()
