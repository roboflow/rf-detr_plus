# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the RF-DETR+ model weights registry."""

import pytest

from rfdetr_plus.assets.model_weights import ModelWeights


class TestModelWeights:
    @pytest.mark.parametrize("model", [pytest.param(m, id=m.filename) for m in ModelWeights])
    def test_all_models_download_from_weights_cdn(self, model: ModelWeights) -> None:
        """Every platform-licensed weight file is served from the RF-DETR weights CDN host."""
        assert model.url.startswith("https://repo.roboflow.com/rfdetr/"), model.url
