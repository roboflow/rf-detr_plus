# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
import tempfile
from pathlib import Path

import pytest
import torch
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64
from rfdetr.detr import RFDETR
from rfdetr.engine import evaluate
from rfdetr.models import build_criterion_and_postprocessors
from rfdetr.util.misc import collate_fn

from rfdetr_plus import RFDETR2XLarge, RFDETRXLarge


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples"),
    [
        pytest.param(RFDETRXLarge, 0.77, 0.74, 500, id="xlarge"),
        pytest.param(RFDETR2XLarge, 0.78, 0.74, 500, id="2xlarge"),
    ],
)
def test_coco_detection_inference_benchmark(
    request: pytest.FixtureRequest,
    download_coco_val: tuple[Path, Path],
    model_cls: type[RFDETR],
    threshold_map: float,
    threshold_f1: float,
    num_samples: int | None,
) -> None:
    """Benchmark COCO detection inference for RF-DETR+ XLarge and 2XLarge models.

    This GPU-marked test runs inference on the COCO val2017 split using the
    specified RF-DETR+ detection model, computes COCO metrics via the base
    RF-DETR evaluation pipeline, and asserts that the achieved mAP@50 and F1
    scores meet or exceed the provided minimum thresholds. Evaluation statistics
    are also written to a JSON file to assist with debugging benchmark runs.

    Args:
        request: Pytest request object, used to access the parametrized test ID.
        download_coco_val: Tuple of (images_root, annotations_path) for COCO
            val2017 data.
        model_cls: RF-DETR model class to benchmark (e.g., RFDETRXLarge,
            RFDETR2XLarge).
        threshold_map: Minimum acceptable mAP@50 value for the benchmark.
        threshold_f1: Minimum acceptable F1 score for the benchmark.
        num_samples: Optional number of validation samples to evaluate; if None,
            evaluates on the full validation set.

    Returns:
        None. Raises an assertion error if the model fails to meet the accuracy
        thresholds.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images_root, annotations_path = download_coco_val

    rfdetr = model_cls(device=device)
    config = rfdetr.model_config
    args = rfdetr.model.args
    if not hasattr(args, "eval_max_dets"):
        args.eval_max_dets = 500

    transforms = make_coco_transforms_square_div_64(
        image_set="val",
        resolution=config.resolution,
        patch_size=config.patch_size,
        num_windows=config.num_windows,
    )
    val_dataset = CocoDetection(images_root, annotations_path, transforms=transforms)
    if num_samples is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, list(range(min(num_samples, len(val_dataset)))))
    import os

    data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        sampler=torch.utils.data.SequentialSampler(val_dataset),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=os.cpu_count() or 1,
    )
    base_ds = get_coco_api_from_dataset(val_dataset)
    criterion, postprocess = build_criterion_and_postprocessors(args)

    rfdetr.model.model.eval()
    with torch.no_grad():
        stats, _ = evaluate(
            rfdetr.model.model,
            criterion,
            postprocess,
            data_loader,
            base_ds,
            torch.device(device),
            args=args,
        )

    # Dump results JSON for debugging
    # Use env var COCO_BENCHMARK_DEBUG_DIR to specify a permanent folder, otherwise use temp
    test_id = request.node.callspec.id
    debug_dir = os.environ.get("COCO_BENCHMARK_DEBUG_DIR", tempfile.gettempdir())
    debug_path = Path(debug_dir) / f"coco_inference_stats_detection_{test_id}_nb-spl-{num_samples or 'all'}.json"
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    with open(debug_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dumped stats to {debug_path}")

    results = stats["results_json"]
    map_val = results["map"]
    f1_val = results["f1_score"]

    print(f"COCO val2017 [{test_id}]: mAP@50={map_val:.4f}, F1={f1_val:.4f}")
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"
