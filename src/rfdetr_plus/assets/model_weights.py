# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
RF-DETR+ Model weights registry.

Provides ModelWeights enum for platform-licensed models,
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
        "6ddf834f2bc5bed3214a82f9b0aaeed7",
    )
    RF_DETR_XXLARGE = ModelWeightAsset(
        "rf-detr-xxlarge.pth",
        "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xxlarge.pth",
        "e3204689c1f0280427e4c33e6a2ac6cd",
    )

    # Platform-Licensed Real-Time Detection Models (Atto, Femto and Pico, PE-Core-T backbone)
    # These models are subject to the Platform Model License 1.0
    RF_DETR_ATTO = ModelWeightAsset(
        "rf-detr-atto.pth",
        "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-atto.pth",
        "89e793e6246ac5808e2f2769ae9b4fe0",
    )
    RF_DETR_FEMTO = ModelWeightAsset(
        "rf-detr-femto.pth",
        "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-femto.pth",
        "f1265b9cf62c8d30ae94918b794ac69e",
    )
    RF_DETR_PICO = ModelWeightAsset(
        "rf-detr-pico.pth",
        "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-pico.pth",
        "ab8da50b62f1f6b3ed234e1292139b72",
    )

    # All methods inherited from ModelWeightsBase:
    # - from_filename(filename) -> Optional[ModelWeightAsset]
    # - get_url(filename) -> Optional[str]
    # - get_md5(filename) -> Optional[str]
    # - list_models() -> list[str]
