# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""GPU training smoke tests for rfdetr's compile/cuda_graphs acceleration flags on RF-DETR+ models.

These prove ``ModelConfig.compile``/``cuda_graphs`` don't just parse (see ``tests/test_config.py``
for the CPU-only field pass-through checks) but survive an actual ``RFDETRXLarge.train()`` step on
CUDA. Only ``RFDETRXLarge`` is exercised here, not ``RFDETR2XLarge``, to respect the CI GPU-runner
time budget upstream's own ``tests/run_smoke_all_models.py`` calls out for these heavyweight models.
"""

from pathlib import Path

import pytest
from rfdetr.config import ModelConfig

from rfdetr_plus.models.detection import RFDETRXLarge


@pytest.mark.gpu
@pytest.mark.parametrize(
    "extra_kwargs",
    [
        pytest.param({"compile": True}, id="compile"),
        pytest.param(
            {"cuda_graphs": True},
            marks=pytest.mark.skipif(
                "cuda_graphs" not in ModelConfig.model_fields,
                reason="rfdetr cuda_graphs field is unreleased upstream (develop #1479/#1481)",
            ),
            id="cuda_graphs",
        ),
    ],
)
def test_xlarge_train_smoke_with_acceleration_flag(
    synthetic_roboflow_dataset: Path,
    tmp_path: Path,
    extra_kwargs: dict[str, bool],
) -> None:
    """RFDETRXLarge.train() completes one step on CUDA with compile/cuda_graphs enabled.

    Builds the model from scratch (no pretrained weight download beyond the DINOv2 backbone) with
    the acceleration flag set at construction time (both fields are constructor-only ModelConfig
    fields, not train() kwargs), then runs a single epoch over a tiny synthetic one-class dataset.
    Only asserts the run completes without error; it is not a convergence or speed test.
    """
    model = RFDETRXLarge(
        device="cuda",
        num_classes=1,
        pretrain_weights=None,
        accept_platform_model_license=True,
        **extra_kwargs,
    )

    model.train(
        dataset_dir=str(synthetic_roboflow_dataset),
        output_dir=str(tmp_path / "output"),
        epochs=1,
        batch_size=1,
        num_workers=0,
    )
