"""Load KEY=VALUE pairs from the repo-root .env (never print values)."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path | None:
    origins = []
    if start is not None:
        origins.append(Path(start).resolve())
    origins.append(Path.cwd().resolve())
    origins.append(Path(__file__).resolve())
    seen: set[Path] = set()
    for origin in origins:
        for folder in (origin, *origin.parents):
            if folder in seen:
                continue
            seen.add(folder)
            if (folder / "pyproject.toml").is_file():
                return folder
    return None


def parse_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        out[key] = value
    return out


def load_dotenv(path: Path, *, override: bool = False) -> dict[str, str]:
    loaded = parse_dotenv(path)
    for key, value in loaded.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def describe_kaggle_auth() -> dict[str, object]:
    """Status only — never includes token text."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    access = Path.home() / ".kaggle" / "access_token"
    if not token:
        prefix = "none"
    elif token.startswith("KGAT_"):
        prefix = "KGAT_"
    else:
        prefix = "other"
    return {
        "token_set": bool(token),
        "token_prefix": prefix,
        "access_token_file": access.is_file(),
    }


def load_repo_env(*, start: Path | None = None, override: bool = False) -> Path | None:
    """Load ``<repo>/.env`` into ``os.environ``. Existing vars win unless override."""
    root = find_repo_root(start)
    if root is None:
        return None
    path = root / ".env"
    if not path.is_file():
        return None
    load_dotenv(path, override=override)
    return path
