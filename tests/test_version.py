# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Test basic package functionality."""

import numpy as np
import pytest

from rfdetr_plus import RFDETR2XLarge, RFDETRXLarge


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_class", "resolution"),
    [
        (RFDETRXLarge, 700),
        (RFDETR2XLarge, 880),
    ],
)
def test_model_inference(model_class, resolution) -> None:
    """Test that we can instantiate RF-DETR+ models and run inference."""
    # Instantiate and run inference
    rf_detr = model_class()
    dummy_image = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)

    # Run inference - this verifies the model can be instantiated and used
    predictions = rf_detr.predict(dummy_image, conf_threshold=0.1)

    # Verify predictions were returned
    assert predictions is not None
