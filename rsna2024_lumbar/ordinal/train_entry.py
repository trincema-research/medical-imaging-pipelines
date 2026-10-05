"""Run bundled training with ordinal metrics (``RSNA2024_ORDINAL_METRICS=1``)."""

from __future__ import annotations

import os
import sys

from rsna2024_lumbar.ordinal.paths import BUNDLE_DIR, refit_env


def main(argv: list[str] | None = None) -> None:
    env = refit_env()
    os.environ.update(env)
    bundle = str(BUNDLE_DIR.resolve())
    if bundle not in sys.path:
        sys.path.insert(0, bundle)
    import train_vit_lumbar as tv

    tv.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
