# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
from rfdetr.datasets._develop import (
    _COCO_URLS,
    _download_and_extract,
    _download_lock,
)
from rfdetr.utilities.reproducibility import seed_all

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def download_coco_val() -> tuple[Path, Path]:
    """Download COCO val2017 images and annotations if not already present.

    Returns:
        Tuple containing the images root directory and annotations file path.
    """
    images_root = _DATA_DIR / "val2017"
    annotations_path = _DATA_DIR / "annotations" / "instances_val2017.json"

    lock_path = _DATA_DIR / ".coco_download.lock"
    with _download_lock(lock_path):
        if not images_root.exists():
            _download_and_extract(_COCO_URLS["val2017"], _DATA_DIR)
        if not annotations_path.exists():
            _download_and_extract(_COCO_URLS["annotations"], _DATA_DIR)

    return images_root, annotations_path


@pytest.fixture(autouse=True)
def seed_everything(request: pytest.FixtureRequest) -> None:
    """Reset random, numpy, torch, and CUDA seeds before each test.

    Defaults to seed 7. Override per-test via indirect parametrize::

        @pytest.mark.parametrize("seed_everything", [42], indirect=True)
        def test_foo(seed_everything): ...

    Args:
        request: Pytest fixture request that may carry an overridden seed.
    """
    seed = request.param if hasattr(request, "param") else 7
    seed_all(seed)


@pytest.fixture
def synthetic_roboflow_dataset(tmp_path: Path) -> Path:
    """Build a minimal Roboflow-format COCO dataset (train/valid splits, one class) for training smoke tests.

    Matches the directory layout ``rfdetr.datasets.coco.build_roboflow_from_coco`` expects:
    ``<root>/<split>/_annotations.coco.json`` plus the referenced images, for
    ``split in ("train", "valid")``. Used by GPU-marked training smoke tests that exercise
    ``ModelConfig.compile``/``cuda_graphs`` end to end without depending on a real dataset
    download.

    Returns:
        Path to the dataset root directory.
    """
    root = tmp_path / "synthetic_dataset"
    category = {"id": 1, "name": "object", "supercategory": "none"}
    image_size = 128
    bbox = [16, 16, 64, 64]

    for split, num_images in (("train", 2), ("valid", 1)):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        images = []
        annotations = []
        for idx in range(num_images):
            file_name = f"{idx}.jpg"
            pixels = np.full((image_size, image_size, 3), idx, dtype=np.uint8)
            PIL.Image.fromarray(pixels).save(split_dir / file_name)
            images.append({"id": idx, "file_name": file_name, "width": image_size, "height": image_size})
            annotations.append(
                {
                    "id": idx,
                    "image_id": idx,
                    "category_id": category["id"],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
        coco = {"images": images, "annotations": annotations, "categories": [category]}
        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(coco, f)

    return root
