"""Adapter for the already-trained VGG11 lung U-Net."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from rsna2018_cxr.segmentation.data.images import assert_native_resolution, to_unit_float
from rsna2018_cxr.segmentation.masks.postprocess import postprocess_merged, split_lungs
from rsna2018_cxr.segmentation.models.types import LungPrediction
from rsna2018_cxr.segmentation.models.vgg_unet import UNetVGG11


class ExistingUNetBackend:
    def __init__(
        self,
        checkpoint: Path,
        threshold: float = 0.5,
        device: str | None = None,
    ):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"U-Net checkpoint not found: {path}")
        self.threshold = float(threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = UNetVGG11(n_channels=1, n_classes=1, use_batch_norm=True, bilinear=True)
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint = str(path)
        self.version = f"epoch-{payload.get('epoch', 'na')}" if isinstance(payload, dict) else "state-dict"

    def predict_batch(self, images: list[np.ndarray]) -> list[LungPrediction]:
        started = time.perf_counter()
        originals = [np.asarray(image) for image in images]
        for image in originals:
            assert_native_resolution(image)
        batch = np.stack([to_unit_float(image) for image in originals], axis=0)
        tensor = torch.from_numpy(batch[:, None, :, :]).to(self.device, non_blocking=True)
        use_amp = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=use_amp):
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)[:, 0].detach().float().cpu().numpy()
        elapsed = (time.perf_counter() - started) * 1000.0
        per_image = elapsed / max(1, len(originals))
        predictions = []
        for image, probability in zip(originals, probabilities):
            if probability.shape != image.shape[:2]:
                raise RuntimeError(
                    f"Model returned {probability.shape}, expected original {image.shape[:2]}."
                )
            restored = probability.astype(np.float32, copy=False)
            binary = probability >= self.threshold
            merged = postprocess_merged(binary, threshold=0.5)
            left, right = split_lungs(merged, radiological=True)
            predictions.append(
                LungPrediction(
                    left_lung=left,
                    right_lung=right,
                    merged_lungs=merged,
                    probability_map=restored,
                    runtime_ms=per_image,
                    model_name="existing_unet_vgg11",
                    model_version=self.version,
                    metadata={
                        "checkpoint": self.checkpoint,
                        "threshold": self.threshold,
                        "input_size": [int(image.shape[0]), int(image.shape[1])],
                        "resize": None,
                        "normalization": "divide_by_255",
                    },
                )
            )
        return predictions

    def predict(self, image: np.ndarray) -> LungPrediction:
        return self.predict_batch([image])[0]
