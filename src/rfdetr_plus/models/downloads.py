# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Legacy model downloads dictionary.

DEPRECATED: Use rfdetr_plus.assets.ModelWeights instead.
This dictionary is maintained for backward compatibility only.
"""

from rfdetr_plus.assets import ModelWeights

# Legacy dictionary for backward compatibility
# New code should use rfdetr_plus.assets.ModelWeights
PLATFORM_MODELS = {asset.filename: asset.url for asset in ModelWeights}
