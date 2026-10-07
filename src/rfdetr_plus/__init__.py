# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR+ extension package.

Platform-licensed detection models for the RF-DETR family: the large-scale DINOv2 models (XLarge and 2XLarge) and
the real-time PE-Core-T models (Atto, Femto and Pico).
"""

__version__ = "1.1.0"
__author__ = "Roboflow, Inc"
__author_email__ = "develop@roboflow.com"
__license__ = "PML-1.0"
__url__ = "https://github.com/roboflow/rf-detr_plus"
__docs__ = "https://rfdetr.roboflow.com"

from rfdetr_plus.models import RFDETR2XLarge, RFDETRAtto, RFDETRFemto, RFDETRPico, RFDETRXLarge

__all__ = [
    "RFDETR2XLarge",
    "RFDETRAtto",
    "RFDETRFemto",
    "RFDETRPico",
    "RFDETRXLarge",
]
