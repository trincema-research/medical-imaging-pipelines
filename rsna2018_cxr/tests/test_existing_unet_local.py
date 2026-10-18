"""Load the local VGG11 lung checkpoint when it is on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CHECKPOINT = Path(r"F:\CXR_segmentation\models\unet_vgg11_1024_best.pth")


@pytest.mark.local
def test_existing_unet_restores_original_shape():
    pytest.importorskip("torch")
    if not CHECKPOINT.is_file():
        pytest.skip("trained lung U-Net checkpoint is not on this machine")
    from rsna2018_cxr.segmentation.models.existing_unet import ExistingUNetBackend

    backend = ExistingUNetBackend(CHECKPOINT)
    image = np.full((1024, 1024), 30, dtype=np.uint8)
    prediction = backend.predict(image)
    assert prediction.merged_lungs.shape == (1024, 1024)
    assert prediction.left_lung.shape == prediction.right_lung.shape == (1024, 1024)
    assert prediction.probability_map is not None
    assert prediction.probability_map.shape == (1024, 1024)
    assert prediction.metadata["resize"] is None
    with pytest.raises(ValueError):
        backend.predict(np.full((96, 128), 30, dtype=np.uint8))
    assert prediction.model_name == "existing_unet_vgg11"
