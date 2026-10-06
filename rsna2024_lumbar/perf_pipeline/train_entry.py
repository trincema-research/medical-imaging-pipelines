"""Run bundled training with severity metrics enabled."""

from __future__ import annotations

import os
import sys

from rsna2024_lumbar.nas.paths import BUNDLE_DIR
from rsna2024_lumbar.perf_pipeline.paths import pipeline_env


def main(argv: list[str] | None = None) -> None:
    env = pipeline_env()
    os.environ.update(env)
    bundle = str(BUNDLE_DIR.resolve())
    if bundle not in sys.path:
        sys.path.insert(0, bundle)
    import train_vit_lumbar as tv

    tv.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
