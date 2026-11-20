# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Accuracy-drop tests of the compact CPU INT8 export of the PE-Core-T detectors.

Each test exports a pretrained detector twice, as a plain FP32 ONNX graph and as the compact dynamic-INT8 graph of
``export_cpu_onnx``, scores both with ONNX Runtime on the same COCO val2017 images through the same pre- and
post-processing, and bounds the AP the INT8 file loses. The tests download COCO val2017 and the pretrained weights and
run several minutes of CPU inference, so they carry the ``quantization`` marker and run in their own CI workflow
(``ci-tests-quantization.yml``) rather than with the unit tests.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
from rfdetr.detr import RFDETR

from rfdetr_plus import RFDETRAtto, RFDETRFemto, RFDETRPico
from tests._markers import _requires_onnx

#: First val2017 images (sorted by id) scored by every test; large enough that the INT8 minus FP32 difference is stable
#: to a few hundredths of an AP, small enough for a CPU runner.
_NUM_IMAGES = 500

#: Confidence floor of the scored detections; COCO AP needs the low-confidence tail.
_SCORE_THRESHOLD = 0.001

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _preprocess(image, height: int, width: int) -> np.ndarray:
    """Resize and normalise a PIL image the way ``RFDETR.predict`` does, as a ``(1, 3, H, W)`` float32 array."""
    import torch
    import torchvision.transforms.functional as functional

    with torch.no_grad():
        tensor = functional.to_tensor(image.convert("RGB"))
        tensor = functional.resize(tensor, [height, width], antialias=False)
        tensor = functional.normalize(tensor, _MEAN, _STD)
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def _decode(boxes: np.ndarray, logits: np.ndarray, width: int, height: int) -> list[tuple[int, float, list[float]]]:
    """Return ``(class slot, score, [x, y, w, h])`` for the top query/class pairs above the score floor.

    Mirrors ``PostProcess``: independent per-class sigmoids, the best ``num_queries`` pairs over all classes, then
    normalised ``cxcywh`` boxes scaled to the source image.
    """
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
    flat = scores.ravel()
    top = np.argsort(-flat)[: logits.shape[0]]
    top = top[flat[top] > _SCORE_THRESHOLD]
    queries, classes = np.divmod(top, logits.shape[1])
    cx, cy, bw, bh = boxes[queries].T
    left = np.clip((cx - bw / 2) * width, 0, width)
    upper = np.clip((cy - bh / 2) * height, 0, height)
    right = np.clip((cx + bw / 2) * width, 0, width)
    lower = np.clip((cy + bh / 2) * height, 0, height)
    return [
        (int(c), float(flat[i]), [float(x), float(y), float(r - x), float(b - y)])
        for i, c, x, y, r, b in zip(top, classes, left, upper, right, lower)
    ]


def _average_precision(model_path: Path, images_root: Path, annotations_path: Path) -> float:
    """COCO box AP (IoU 0.50:0.95, percent) of an exported ONNX detector on the first ``_NUM_IMAGES`` val images."""
    import onnxruntime
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    height, width = session.get_inputs()[0].shape[2:]
    output_names = [output.name for output in session.get_outputs()]
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotations_path))
    image_ids = sorted(ground_truth.getImgIds())[:_NUM_IMAGES]

    results = []
    for image_id in image_ids:
        info = ground_truth.loadImgs(image_id)[0]
        with Image.open(images_root / info["file_name"]) as image:
            batch = _preprocess(image, height, width)
        outputs = dict(zip(output_names, session.run(None, {input_name: batch})))
        for category, score, bbox in _decode(outputs["dets"][0], outputs["labels"][0], info["width"], info["height"]):
            results.append({"image_id": image_id, "category_id": category, "bbox": bbox, "score": score})

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = COCOeval(ground_truth, ground_truth.loadRes(results), "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[0]) * 100


@_requires_onnx
@pytest.mark.quantization
class TestCompactCpuExportAccuracy:
    # Bounds sit above the drops measured on these 500 images (Atto 0.02, Femto 0.05, Pico 0.46 AP) with room for
    # numerical differences between ONNX Runtime builds; the static TensorRT INT8 recipe, by contrast, loses 7 to 9 AP.
    @pytest.mark.parametrize(
        ("model_class", "max_drop"),
        [
            pytest.param(RFDETRAtto, 0.5, id="atto"),
            pytest.param(RFDETRFemto, 0.5, id="femto"),
            pytest.param(RFDETRPico, 1.0, id="pico"),
        ],
    )
    def test_int8_ap_drop_is_within_bound(
        self, download_coco_val: tuple[Path, Path], tmp_path: Path, model_class: type[RFDETR], max_drop: float
    ) -> None:
        """The compact INT8 file loses at most ``max_drop`` AP against the FP32 graph on the same images.

        A pretrained detector is exported as plain FP32 ONNX and as the compact dynamic-INT8 graph, both scored by
        ONNX Runtime on the first val2017 images. The drop, not the absolute AP, is bounded, so the subset and the
        pre- and post-processing cancel out. A recipe change that quantizes a sensitive projection (the PE-Core-T
        ``fc2`` inputs are the known one) shows up here as a drop of several AP.
        """
        images_root, annotations_path = download_coco_val
        detector = model_class()
        reference = detector.export(output_dir=tmp_path / "fp32", format="onnx", verbose=False)
        compact = detector.export_cpu_onnx(tmp_path / "int8")

        fp32_ap = _average_precision(reference, images_root, annotations_path)
        int8_ap = _average_precision(compact, images_root, annotations_path)

        print(f"{detector.size}: FP32 AP {fp32_ap:.2f}, compact INT8 AP {int8_ap:.2f}, drop {fp32_ap - int8_ap:.2f}")
        assert fp32_ap - int8_ap <= max_drop, f"INT8 lost {fp32_ap - int8_ap:.2f} AP (FP32 {fp32_ap:.2f})"
