"""Download the released inference-only S1 specialist checkpoints."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from rsna2018_cxr.severity.constants import GITHUB_REPO, MODELS_DIR, RELEASE_TAG, SPECIALIST_CHECKPOINTS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download published RSNA 2018 S1 specialist weights.")
    p.add_argument("--dest", type=Path, default=MODELS_DIR)
    p.add_argument("--tag", default=RELEASE_TAG)
    p.add_argument("--repo", default=GITHUB_REPO)
    return p.parse_args(argv)


def _url(repo: str, tag: str, name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    for spec in SPECIALIST_CHECKPOINTS:
        name = spec["name"]
        path = dest / name
        url = _url(args.repo, args.tag, name)
        print(f"GET {url}")
        urllib.request.urlretrieve(url, path)
        print(f"  -> {path} ({path.stat().st_size} bytes)")
    sidecar = dest / "download.json"
    sidecar.write_text(
        json.dumps({"repo": args.repo, "tag": args.tag, "files": [s["name"] for s in SPECIALIST_CHECKPOINTS]}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
