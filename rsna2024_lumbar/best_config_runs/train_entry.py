"""Run bundled training with severity metrics enabled."""

from __future__ import annotations

import os
import sys

from rsna2024_lumbar.nas.paths import BUNDLE_DIR, REPO_ROOT
from rsna2024_lumbar.best_config_runs.paths import pipeline_env


def main(argv: list[str] | None = None) -> None:
    env = pipeline_env()
    os.environ.update(env)
    for path in (str(REPO_ROOT.resolve()), str(BUNDLE_DIR.resolve())):
        if path not in sys.path:
            sys.path.insert(0, path)
    import train_vit_lumbar as tv

    tv.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
