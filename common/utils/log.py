"""Stdout + file traces: START / STEP / DATA / OK / DONE / NEXT / WARN / FAIL."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import NoReturn

from common.utils.env import find_repo_root

_files: list[Path] = []
_component = ""

YEAR_TRACE_FOLDERS = {
    "rsna2024": "rsna2024_lumbar",
}


def fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def component_name(script: str) -> str:
    return script.rsplit(".", 1)[-1]


def infer_traces_dir(script: str, start: Path | None = None) -> Path | None:
    root = find_repo_root(start)
    if root is None:
        return None
    year = script.split(".", 1)[0]
    folder = YEAR_TRACE_FOLDERS.get(year)
    if folder:
        return root / folder / "traces"
    return root / "traces"


def trace_paths(traces_dir: Path, script: str) -> list[Path]:
    """Per-component file + combined pipeline.log."""
    component = component_name(script)
    traces_dir = Path(traces_dir)
    return [traces_dir / f"{component}.log", traces_dir / "pipeline.log"]


def configure(
    script: str,
    *,
    traces_dir: Path | None = None,
    log_file: Path | None = None,
    force: bool = False,
) -> list[Path]:
    """
    Send traces to stdout and, unless under pytest, to files.

    Default files (RSNA 2024): ``rsna2024_lumbar/traces/<component>.log``
    and ``rsna2024_lumbar/traces/pipeline.log``. Pass ``force=True`` in tests.
    """
    global _files, _component
    _component = component_name(script)
    _files = []
    under_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    if log_file is not None:
        _files = [Path(log_file)]
    elif under_pytest and not force:
        _files = []
    else:
        directory = Path(traces_dir) if traces_dir is not None else infer_traces_dir(script)
        if directory is not None:
            _files = trace_paths(directory, script)
    for path in _files:
        path.parent.mkdir(parents=True, exist_ok=True)
    return list(_files)


def _emit(kind: str, msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {kind:<5} {msg}"
    print(line, flush=True)
    payload = line + "\n"
    for path in _files:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload)


def start(script: str, *, traces_dir: Path | None = None, force: bool = False, **kwargs) -> None:
    configure(script, traces_dir=traces_dir, force=force)
    if _files:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner = f"\n-------- {stamp}  {_component}  {script} --------\n"
        for path in _files:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(banner)
    _emit("START", script)
    if kwargs:
        data(**kwargs)


def step(msg: str) -> None:
    _emit("STEP", msg)


def data(msg: str | None = None, **kwargs) -> None:
    bits = []
    if msg:
        bits.append(msg)
    bits.extend(f"{key}={value}" for key, value in kwargs.items())
    _emit("DATA", "  ".join(bits))


def ok(msg: str) -> None:
    _emit("OK", msg)


def done(msg: str) -> None:
    _emit("DONE", msg)


def next_step(msg: str) -> None:
    _emit("NEXT", msg)


def warn(msg: str) -> None:
    _emit("WARN", msg)


def fail(msg: str) -> None:
    _emit("FAIL", msg)


def die(msg: str, code: int | str = 1) -> NoReturn:
    """Log FAIL and exit. ``code`` may be the SystemExit message (pytest matches it)."""
    fail(msg)
    if isinstance(code, str):
        raise SystemExit(code)
    if code == 1:
        raise SystemExit(msg)
    raise SystemExit(code)


def disk_free(path: Path) -> str:
    target = path if path.exists() else path.parent
    try:
        return fmt_bytes(shutil.disk_usage(str(target)).free)
    except OSError:
        return "unknown"
