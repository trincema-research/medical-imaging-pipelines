"""Download a Kaggle competition dump. Used by year-specific data/download.py CLIs."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path


def _download_via_api(slug: str, dest: Path, *, force: bool) -> bool:
    """Classic username/key via ~/.kaggle/kaggle.json. Returns False if unused."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        return False
    print(f"kaggle API: {slug} -> {dest}", flush=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(slug, path=str(dest), quiet=False, force=force)
    return True


def require_kaggle_cli() -> str:
    exe = shutil.which("kaggle")
    if exe is None:
        raise SystemExit(
            "kaggle CLI not found. Install with: pip install -e '.[kaggle]'\n"
            "Then: kaggle auth login  OR  ~/.kaggle/kaggle.json (username/key)."
        )
    return exe


def download_competition(
    slug: str,
    dest: Path,
    *,
    force: bool = False,
    unzip: bool = True,
) -> Path:
    """
    Download ``slug`` into ``dest``.

    Prefers the Python API (legacy ``kaggle.json`` username/key), then the CLI
    (``kaggle auth login`` / ``KAGGLE_API_TOKEN``). Accept competition rules
    in the browser first or the API returns 403.
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    used_api = False
    try:
        used_api = _download_via_api(slug, dest, force=force)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"kaggle API failed ({exc}); trying CLI.", flush=True)
        used_api = False
    if not used_api:
        exe = require_kaggle_cli()
        cmd = [exe, "competitions", "download", "-c", slug, "-p", str(dest)]
        if force:
            cmd.append("-f")
        print(" ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Kaggle download failed (exit {exc.returncode}). "
                "Run `kaggle auth login` or use ~/.kaggle/kaggle.json, "
                "and accept the competition rules on kaggle.com."
            ) from exc
    if unzip:
        _unzip_all(dest)
    return dest


def _unzip_all(dest: Path) -> None:
    zips = sorted(dest.glob("*.zip"))
    for archive in zips:
        print(f"Unzip {archive.name} -> {dest}", flush=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        archive.unlink()
