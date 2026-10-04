#!/usr/bin/env python3
"""Patch bundled vit_nas_lumbar.py to accept --model-type efficientnet3d (idempotent)."""

from __future__ import annotations

import sys
from pathlib import Path


def patch(text: str) -> str:
    if "MODEL_TYPE_EFFICIENTNET3D" in text and "efficientnet3d requires a 3D input layout" in text:
        return text

    if "MODEL_TYPE_EFFICIENTNET3D" not in text:
        text = text.replace(
            "    MODEL_TYPE_EFFICIENTNET,\n    MODEL_TYPE_MAXVIT,",
            "    MODEL_TYPE_EFFICIENTNET,\n    MODEL_TYPE_EFFICIENTNET3D,\n    MODEL_TYPE_MAXVIT,",
            1,
        )
        text = text.replace(
            "            MODEL_TYPE_EFFICIENTNET,\n            MODEL_TYPE_MAXVIT,",
            "            MODEL_TYPE_EFFICIENTNET,\n            MODEL_TYPE_EFFICIENTNET3D,\n            MODEL_TYPE_MAXVIT,",
            1,
        )

    if 'return "runs/lumbar_nas_efficientnet3d"' not in text:
        text = text.replace(
            '    if model_type == MODEL_TYPE_EFFICIENTNET:\n        return "runs/lumbar_nas_efficientnet"\n    if model_type == MODEL_TYPE_MAXVIT:',
            '    if model_type == MODEL_TYPE_EFFICIENTNET:\n        return "runs/lumbar_nas_efficientnet"\n    if model_type == MODEL_TYPE_EFFICIENTNET3D:\n        return "runs/lumbar_nas_efficientnet3d"\n    if model_type == MODEL_TYPE_MAXVIT:',
            1,
        )

    old_3d = "    if model_type in (MODEL_TYPE_VIT3D, MODEL_TYPE_CONVNEXT3D, MODEL_TYPE_MAXVIT3D):"
    new_3d = (
        "    if model_type in (\n"
        "        MODEL_TYPE_VIT3D,\n"
        "        MODEL_TYPE_CONVNEXT3D,\n"
        "        MODEL_TYPE_EFFICIENTNET3D,\n"
        "        MODEL_TYPE_MAXVIT3D,\n"
        "    ):"
    )
    if old_3d in text:
        text = text.replace(old_3d, new_3d, 1)

    if "MODEL_TYPE_EFFICIENTNET3D" in text and "return list(space.vit_variants)" in text:
        # Ensure efficientnet3d is handled before the vit fallback.
        marker = "def backbone_variants_for_model"
        if marker in text and "model_type == MODEL_TYPE_EFFICIENTNET3D" not in text.split(marker)[1].split("def build_trial_configs")[0]:
            raise RuntimeError(
                "efficientnet3d missing from backbone_variants_for_model; replace vit_nas_lumbar.py from repo."
            )

    if "model_type == MODEL_TYPE_EFFICIENTNET3D" not in text.split("backbone_variants_for_model")[1][:800]:
        text = text.replace(
            "    if model_type == MODEL_TYPE_EFFICIENTNET:\n        return list(space.effnet2d_variants)\n    if model_type == MODEL_TYPE_MAXVIT:",
            "    if model_type == MODEL_TYPE_EFFICIENTNET:\n        return list(space.effnet2d_variants)\n    if model_type == MODEL_TYPE_EFFICIENTNET3D:\n        return list(space.cnn3d_variants)\n    if model_type == MODEL_TYPE_MAXVIT:",
            1,
        )

    guard = (
        '    if args.model_type == MODEL_TYPE_EFFICIENTNET3D and args.input_layout == INPUT_LAYOUT_2D:\n'
        '        raise SystemExit(\n'
        '            "model-type efficientnet3d requires a 3D input layout (default: 3d_level_stack)."\n'
        "        )\n"
    )
    if "efficientnet3d requires a 3D input layout" not in text:
        text = text.replace(
            '    if args.model_type == MODEL_TYPE_EFFICIENTNET and args.resolved_input_layout != INPUT_LAYOUT_2D:\n        raise SystemExit("model-type efficientnet requires input_layout=2d.")\n    if args.model_type == MODEL_TYPE_MAXVIT',
            '    if args.model_type == MODEL_TYPE_EFFICIENTNET and args.resolved_input_layout != INPUT_LAYOUT_2D:\n        raise SystemExit("model-type efficientnet requires input_layout=2d.")\n'
            + guard
            + "    if args.model_type == MODEL_TYPE_MAXVIT",
            1,
        )

    if "MODEL_TYPE_EFFICIENTNET3D" not in text:
        raise RuntimeError("Patch did not apply; vit_nas_lumbar.py layout differs from expected.")
    return text


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rsna2024_lumbar/nas/training_bundle/vit_nas_lumbar.py")
    path = root.resolve()
    original = path.read_text(encoding="utf-8")
    updated = patch(original)
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        print(f"Patched {path}")
    else:
        print(f"Already OK: {path}")


if __name__ == "__main__":
    main()
