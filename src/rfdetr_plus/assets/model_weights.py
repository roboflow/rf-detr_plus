# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
RF-DETR+ Model weights registry.

Provides ModelWeights enum for platform-licensed large-scale models,
compatible with rf-detr's asset structure introduced in version 1.4.3.
"""

from rfdetr.assets.model_weights import ModelWeightAsset, ModelWeightsBase


class ModelWeights(ModelWeightsBase):
    """
    Enumeration of RF-DETR+ platform-licensed model assets.

    Inherits from rf-detr's ModelWeightsBase to ensure compatibility.

    Each enum member's value is a ModelWeightAsset instance containing:
    - filename: The local filename for the model weights
    - url: The download URL
    - md5_hash: The expected MD5 hash for integrity validation

    Example:
        >>> from rfdetr_plus.assets import ModelWeights
        >>> asset = ModelWeights.RF_DETR_XLARGE
        >>> asset.filename
        'rf-detr-xlarge.pth'
        >>> asset.url
        'https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xlarge.pth'
    """

    # Platform-Licensed Detection Models (XLarge and 2XLarge)
    # These models are subject to the Platform Model License 1.0
    RF_DETR_XLARGE = ModelWeightAsset(
        "rf-detr-xlarge.pth",
        "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xlarge.pth",
        None,  # MD5 hash to be added when available
    )
    RF_DETR_XXLARGE = ModelWeightAsset(
        "rf-detr-xxlarge.pth",
        "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xxlarge.pth",
        None,  # MD5 hash to be added when available
    )

    # All methods inherited from ModelWeightsBase:
    # - from_filename(filename) -> Optional[ModelWeightAsset]
    # - get_url(filename) -> Optional[str]
    # - get_md5(filename) -> Optional[str]
    # - list_models() -> list[str]
