# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Skip markers shared by several test modules; a gate used by one module lives in that module."""

import pytest
from rfdetr.export.imports import _IS_ONNX_INSTALLED

_requires_onnx = pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed")
