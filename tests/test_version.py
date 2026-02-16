# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Test basic package functionality."""

import numpy as np
import pytest

from rfdetr_plus import RFDETRXLarge


@pytest.mark.gpu
def test_model_inference() -> None:
    """Test that we can instantiate RF-DETR+ XLarge and run inference."""
    # Instantiate and run inference
    rf_detr = RFDETRXLarge()
    dummy_image = np.random.randint(0, 255, (700, 700, 3), dtype=np.uint8)

    # Run inference - this verifies the model can be instantiated and used
    predictions = rf_detr.predict(dummy_image, conf_threshold=0.1)

    # Verify predictions were returned
    assert predictions is not None
