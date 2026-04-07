# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Test basic package functionality."""

# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64
from rfdetr.detr import RFDETR
from rfdetr.evaluation.coco_eval import CocoEvaluator
from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
from rfdetr.evaluation.matching import build_matching_data, init_matching_accumulator, merge_matching_data
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy
from rfdetr.utilities.tensors import collate_fn

from rfdetr_plus import RFDETR2XLarge, RFDETRXLarge


@pytest.mark.parametrize(
    ("model_cls", "threshold_map", "threshold_f1", "num_samples"),
    [
        pytest.param(RFDETRXLarge, 0.7, 0.7, 20, id="xlarge-CPU"),
        pytest.param(RFDETR2XLarge, 0.7, 0.7, 20, id="2xlarge-CPU"),
        pytest.param(RFDETRXLarge, 0.77, 0.74, 200, marks=pytest.mark.gpu, id="xlarge-GPU"),
        pytest.param(RFDETR2XLarge, 0.78, 0.74, 200, marks=pytest.mark.gpu, id="2xlarge-GPU"),
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
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(num_samples, len(val_dataset))))
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
    _, postprocess = build_criterion_and_postprocessors(args)

    coco_evaluator = CocoEvaluator(base_ds, ["bbox"], args.eval_max_dets)
    f1_accumulator = init_matching_accumulator()

    rfdetr.model.model.eval()
    with torch.no_grad():
        for samples, targets in data_loader:
            # Move current batch to the same device as the model.
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Run model forward pass and decode detections at original image sizes.
            outputs = rfdetr.model.model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results_all = postprocess(outputs, orig_target_sizes)

            # Update COCO mAP evaluator (expects predictions keyed by image_id).
            res = {target["image_id"].item(): output for target, output in zip(targets, results_all)}
            coco_evaluator.update(res)

            # Build F1 matching targets in the format expected by build_matching_data.
            matching_targets: list[dict[str, torch.Tensor]] = []
            for target in targets:
                # F1 matching expects absolute xyxy GT boxes.
                height, width = target["orig_size"].tolist()
                scale = target["boxes"].new_tensor([width, height, width, height])
                boxes = box_cxcywh_to_xyxy(target["boxes"]) * scale
                entry: dict[str, torch.Tensor] = {"boxes": boxes, "labels": target["labels"]}
                if "iscrowd" in target:
                    entry["iscrowd"] = target["iscrowd"]
                matching_targets.append(entry)

            # Accumulate per-class TP/FP/ignore stats for confidence-threshold sweep.
            batch_matching = build_matching_data(list(results_all), matching_targets)
            f1_accumulator = merge_matching_data(f1_accumulator, batch_matching)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_bbox = coco_evaluator.coco_eval["bbox"]
    # COCO stats index 1 is AP@IoU=0.50
    _COCO_AP50_STATS_INDEX = 1
    map_val = float(coco_bbox.stats[_COCO_AP50_STATS_INDEX])

    _EMPTY_CLASS_DATA = {
        "scores": np.array([], dtype=np.float32),
        "matches": np.array([], dtype=np.int64),
        "ignore": np.array([], dtype=bool),
        "total_gt": 0,
    }
    _NUM_CONF_THRESHOLDS = 101
    if f1_accumulator:
        # Class IDs are sparse in COCO; evaluate only observed classes.
        class_ids = sorted(f1_accumulator.keys())
        per_class_list = [f1_accumulator[class_id] for class_id in class_ids]
        classes_with_gt = [
            index for index, class_id in enumerate(class_ids) if f1_accumulator[class_id]["total_gt"] > 0
        ]
    else:
        per_class_list = [_EMPTY_CLASS_DATA]
        classes_with_gt = []
    conf_thresholds = np.linspace(0.0, 1.0, _NUM_CONF_THRESHOLDS)
    sweep_results = sweep_confidence_thresholds(per_class_list, conf_thresholds, classes_with_gt)
    f1_val = float(max((r["macro_f1"] for r in sweep_results), default=0.0))

    stats = {"results_json": {"map": map_val, "f1_score": f1_val}}

    # Dump results JSON for debugging
    # Use env var COCO_BENCHMARK_DEBUG_DIR to specify a permanent folder, otherwise use temp
    test_id = request.node.callspec.id
    debug_dir = os.environ.get("COCO_BENCHMARK_DEBUG_DIR", tempfile.gettempdir())
    debug_path = Path(debug_dir) / f"coco_inference_stats_detection_{test_id}_nb-spl-{num_samples or 'all'}.json"
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    with open(debug_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dumped stats to {debug_path}")

    print(f"COCO val2017 [{test_id}]: mAP@50={map_val:.4f}, F1={f1_val:.4f}")
    assert map_val >= threshold_map, f"mAP@50 {map_val:.4f} < {threshold_map}"
    assert f1_val >= threshold_f1, f"F1 {f1_val:.4f} < {threshold_f1}"


@pytest.mark.parametrize(
    ("model_class", "resolution"),
    [
        (RFDETRXLarge, 700),
        (RFDETR2XLarge, 880),
    ],
)
def test_model_inference(model_class, resolution) -> None:
    """Test that we can instantiate RF-DETR+ models and run inference."""
    # Instantiate and run inference
    rf_detr = model_class()
    dummy_image = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)

    # Run inference - this verifies the model can be instantiated and used
    predictions = rf_detr.predict(dummy_image, conf_threshold=0.1)

    # Verify predictions were returned
    assert predictions is not None
