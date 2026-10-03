"""Download / unzip a Kaggle competition dump. Used by year-specific data CLIs."""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from common.utils.env import describe_kaggle_auth, load_repo_env
from common.utils.log import data, die, disk_free, fmt_bytes, ok, step, warn


def _download_via_api(slug: str, dest: Path, *, force: bool) -> bool:
    """Classic username/key via ~/.kaggle/kaggle.json. Returns False if unused."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        return False
    step("kaggle Python API (legacy kaggle.json)")
    data(slug=slug, dest=dest)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(slug, path=str(dest), quiet=False, force=force)
    return True


def require_kaggle_cli() -> str:
    exe = shutil.which("kaggle")
    if exe is None:
        die(
            "kaggle CLI not found. Install with: pip install -e '.[kaggle]'\n"
            "Then copy .env.example to .env and set KAGGLE_API_TOKEN."
        )
    data(kaggle_cli=exe)
    return exe


def _token_hint() -> str:
    return (
        "Copy .env.example to .env and set KAGGLE_API_TOKEN "
        "(https://www.kaggle.com/settings/api — Generate New Token, not the old kaggle.json key). "
        "Accept competition rules first."
    )


def ensure_kaggle_token() -> None:
    """Load repo .env, then require KAGGLE_API_TOKEN or ~/.kaggle/access_token."""
    step("auth: load .env / KAGGLE_API_TOKEN")
    env_path = load_repo_env()
    info = describe_kaggle_auth()
    data(env_file=env_path or "(none)", **info)
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    access = Path.home() / ".kaggle" / "access_token"
    if token or (access.is_file() and access.stat().st_size > 0):
        ok("auth ready (token present; value not logged)")
        return
    die(f"No usable Kaggle token. {_token_hint()}")


def find_zips(dest: Path) -> list[Path]:
    return sorted(p for p in Path(dest).glob("*.zip") if p.is_file())


def find_partials(dest: Path) -> list[Path]:
    return sorted(p for p in Path(dest).glob("*.kaggle-partial") if p.is_file())


def require_no_partials(dest: Path) -> None:
    dest = Path(dest)
    partials = find_partials(dest)
    if not partials:
        return
    names = ", ".join(p.name for p in partials)
    sizes = ", ".join(fmt_bytes(p.stat().st_size) for p in partials)
    data(partials=names, partial_size=sizes)
    die(
        f"Download failed or is incomplete: {names} under {dest}. "
        "The Kaggle CLI is still transferring or was interrupted. "
        "Re-run: python -m rsna2024_lumbar.data.download"
    )


def unzip_competition(dest: Path, *, delete_zip: bool = True) -> list[Path]:
    """Extract every ``*.zip`` under ``dest``. Refuses ``*.kaggle-partial`` leftovers."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    step("unzip: refuse leftover *.kaggle-partial")
    require_no_partials(dest)
    zips = find_zips(dest)
    if not zips:
        die(
            f"No .zip under {dest}. "
            "Run python -m rsna2024_lumbar.data.download first "
            "(or wait if a *.kaggle-partial is still growing)."
        )
    data(zip_count=len(zips), dest=dest, delete_zip=delete_zip)
    done: list[Path] = []
    for archive in zips:
        size = fmt_bytes(archive.stat().st_size)
        step(f"unzip {archive.name}")
        data(archive=archive.name, size=size, dest=dest)
        try:
            with zipfile.ZipFile(archive) as zf:
                members = zf.namelist()
                preview = ", ".join(members[:5])
                extra = f" … +{len(members) - 5} more" if len(members) > 5 else ""
                data(members=len(members), first=f"{preview}{extra}")
                zf.extractall(dest)
        except zipfile.BadZipFile as exc:
            die(
                f"Unzip failed: {archive.name} is corrupt or incomplete ({exc}). "
                "Delete it and re-run download with --force."
            )
        except OSError as exc:
            die(f"Unzip failed: {archive.name}: {exc}")
        ok(f"Unzip OK: {archive.name}")
        if delete_zip:
            archive.unlink()
            data(removed=archive.name)
        done.append(archive)
    return done


def download_competition(
    slug: str,
    dest: Path,
    *,
    force: bool = False,
    unzip: bool = True,
    delete_zip: bool = True,
) -> Path:
    """
    Download ``slug`` into ``dest``.

    Auth: repo-root ``.env`` ``KAGGLE_API_TOKEN``, then ``$env:KAGGLE_API_TOKEN``,
    then ``~/.kaggle/access_token``. Accept competition rules in the browser first
    or the API returns 403.

    Exits non-zero if the CLI fails, a ``*.kaggle-partial`` is left behind,
    no zip is written, or unzip hits a corrupt archive.
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    ensure_kaggle_token()
    step("resolve kaggle CLI")
    exe = require_kaggle_cli()
    cmd = [exe, "competitions", "download", "-c", slug, "-p", str(dest)]
    if force:
        cmd.append("-f")
    step("kaggle competitions download")
    data(cmd=" ".join(cmd), dest=dest, disk_free=disk_free(dest), force=force)
    try:
        completed = subprocess.run(cmd, check=False)
    except OSError as exc:
        die(f"Kaggle download failed: could not start the CLI ({exc}).")
    data(cli_exit=completed.returncode)
    if completed.returncode != 0:
        die(
            f"Kaggle download failed (exit {completed.returncode}). "
            f"{_token_hint()} "
            "Also check disk space / network. "
            f"If {dest} has a *.kaggle-partial file, re-run download to resume."
        )
    step("check download artifacts")
    require_no_partials(dest)
    zips = find_zips(dest)
    zip_info = ", ".join(f"{p.name} ({fmt_bytes(p.stat().st_size)})" for p in zips) or "(none)"
    data(zips=zip_info, train_csv=(dest / "train.csv").is_file())
    if not zips and not (dest / "train.csv").is_file():
        die(f"Download failed: no .zip written under {dest}.")
    ok("Download OK")
    if unzip and zips:
        step("auto-unzip after download")
        unzip_competition(dest, delete_zip=delete_zip)
    elif unzip and not zips:
        warn("no zip to unzip (files may already be extracted)")
    return dest
