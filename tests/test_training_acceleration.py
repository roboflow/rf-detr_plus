# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GPU training smoke tests for RF-DETR+'s ``compile`` acceleration flag.

This proves ``ModelConfig.compile`` survives an actual ``RFDETRXLarge.train()`` step on CUDA.
``cuda_graphs`` remains covered as a config round-trip in ``tests/test_config.py`` when upstream
releases the field; a train smoke test cannot claim capture/replay without an upstream activation
oracle. Only ``RFDETRXLarge`` is exercised to respect the GPU-runner time budget.
"""

from pathlib import Path

import pytest

from rfdetr_plus.models.detection import RFDETRXLarge


@pytest.mark.gpu
def test_xlarge_train_smoke_with_compile(
    synthetic_roboflow_dataset: Path,
    tmp_path: Path,
) -> None:
    """RFDETRXLarge.train() completes one step on CUDA with ``compile=True`` enabled.

    Builds the model from scratch (no pretrained weight download beyond the DINOv2 backbone) with
    the constructor-only ``compile`` flag set, then runs a single epoch over a tiny synthetic
    one-class dataset. It asserts only completion, not convergence or speed.
    """
    model = RFDETRXLarge(
        device="cuda",
        num_classes=1,
        pretrain_weights=None,
        accept_platform_model_license=True,
        compile=True,
    )

    model.train(
        dataset_dir=str(synthetic_roboflow_dataset),
        output_dir=str(tmp_path / "output"),
        epochs=1,
        batch_size=1,
        num_workers=0,
    )
