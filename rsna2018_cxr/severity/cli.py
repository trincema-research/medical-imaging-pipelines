"""CLI: python -m rsna2018_cxr.severity {train,nas,retrain,export-weights,download-weights}"""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "train": "rsna2018_cxr.severity.train_vit",
    "nas": "rsna2018_cxr.severity.nas",
    "retrain": "rsna2018_cxr.severity.retrain",
    "export-weights": "rsna2018_cxr.severity.export_weights",
    "download-weights": "rsna2018_cxr.severity.download_weights",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        names = " | ".join(COMMANDS)
        print(f"usage: python -m rsna2018_cxr.severity {{{names}}}")
        return 0
    command = args[0]
    if command not in COMMANDS:
        print(f"unknown command {command!r}. choose from {sorted(COMMANDS)}", file=sys.stderr)
        return 2
    return int(importlib.import_module(COMMANDS[command]).main(args[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
