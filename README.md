# RF-DETR+: Large-Scale Detection Models for RF-DETR

[![version](https://badge.fury.io/py/rfdetr-plus.svg)](https://badge.fury.io/py/rfdetr-plus)
[![downloads](https://img.shields.io/pypi/dm/rfdetr-plus)](https://pypistats.org/packages/rfdetr-plus)
[![python-version](https://img.shields.io/pypi/pyversions/rfdetr-plus)](https://badge.fury.io/py/rfdetr-plus)
[![license](https://img.shields.io/badge/license-PML--1.0-blue)](https://github.com/roboflow/rf-detr-plus/blob/main/LICENSE)

[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

RF-DETR+ is the extension package for [RF-DETR](https://github.com/roboflow/rf-detr) that provides the **XLarge** and **2XLarge** detection models. These are the largest and most accurate models in the RF-DETR family, pushing state-of-the-art accuracy on [Microsoft COCO](https://cocodataset.org/#home) and [RF100-VL](https://github.com/roboflow/rf100-vl) while retaining real-time inference speeds.

RF-DETR+ models use a DINOv2 vision transformer backbone at higher resolutions and larger feature dimensions than the core RF-DETR lineup, unlocking top-tier detection accuracy for applications where precision matters most.

## Install

Install RF-DETR+ in a [**Python>=3.10**](https://www.python.org/) environment with `pip`. This will also install [`rfdetr`](https://github.com/roboflow/rf-detr) as a dependency.

```bash
pip install rfdetr-plus
```

<details>
<summary>Install from source</summary>

<br>

```bash
pip install git+https://github.com/roboflow/rf-detr-plus.git
```

</details>

## Benchmarks

RF-DETR+ XLarge and 2XLarge sit at the top of the RF-DETR accuracy/latency curve, delivering the highest COCO AP scores in the family. All latency numbers were measured on an NVIDIA T4 using TensorRT, FP16, and batch size 1.

| Size | Class | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | RF100VL AP<sub>50</sub> | RF100VL AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |                      Package / License                       |
| :--: | :-----------------: | :------------------: | :---------------------: | :---------------------: | :------------------------: | :----------: | :--------: | :--------: |:------------------------------------------------------------:|
|  N   |    `RFDETRNano`     |         67.6         |          48.4           |          85.0           |            57.7            |     2.3      |    30.5    |  384x384   | [`rfdetr`](https://github.com/roboflow/rf-detr) / Apache 2.0 |
|  S   |    `RFDETRSmall`    |         72.1         |          53.0           |          86.7           |            60.2            |     3.5      |    32.1    |  512x512   | [`rfdetr`](https://github.com/roboflow/rf-detr) / Apache 2.0 |
|  M   |   `RFDETRMedium`    |         73.6         |          54.7           |          87.4           |            61.2            |     4.4      |    33.7    |  576x576   | [`rfdetr`](https://github.com/roboflow/rf-detr) / Apache 2.0 |
|  L   |    `RFDETRLarge`    |         75.1         |          56.5           |          88.2           |            62.2            |     6.8      |    33.9    |  704x704   | [`rfdetr`](https://github.com/roboflow/rf-detr) / Apache 2.0 |
|  XL  |   `RFDETRXLarge`    |         77.4         |          58.6           |          88.5           |            62.9            |     11.5     |   126.4    |  700x700   |              `rfdetr_plus` / [PML 1.0](LICENSE)              |
| 2XL  |   `RFDETR2XLarge`   |         78.5         |          60.1           |          89.0           |            63.2            |     17.2     |   126.9    |  880x880   |             `rfdetr_plus` / [PML 1.0](LICENSE)              |

## Run Models

Install with the `plus` extra to get XL and 2XL models alongside core RF-DETR:

```bash
pip install rfdetr[plus]
```

RF-DETR+ models require you to accept the Platform Model License before use. Once accepted, usage mirrors the standard RF-DETR API -- you import directly from `rfdetr`:

```python
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRXLarge
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRXLarge(accept_platform_model_license=True)

image = Image.open(requests.get("https://media.roboflow.com/dog.jpg", stream=True).raw)
detections = model.predict(image, threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
```

### Train Models

RF-DETR+ models support fine-tuning with the same training API as core RF-DETR. You can train on your own dataset or use datasets from [Roboflow Universe](https://universe.roboflow.com/).

```python
from rfdetr import RFDETRXLarge

model = RFDETRXLarge(accept_platform_model_license=True)
model.train(dataset_dir="path/to/dataset", epochs=50, lr=1e-4)
```

## Documentation

Visit the [RF-DETR documentation website](https://rfdetr.roboflow.com) to learn more about training, export, deployment, and the full model lineup.

## License

RF-DETR+ code and model checkpoints are licensed under the Platform Model License 1.0 (PML-1.0). See [`LICENSE`](LICENSE) for details. These models require a [Roboflow](https://roboflow.com) account to run and fine-tune.

The core RF-DETR models (Nano through Large) are available under the Apache License 2.0 in the [`rfdetr`](https://github.com/roboflow/rf-detr) package.

## Acknowledgements

Our work is built upon [LW-DETR](https://arxiv.org/pdf/2406.03459), [DINOv2](https://arxiv.org/pdf/2304.07193), and [Deformable DETR](https://arxiv.org/pdf/2010.04159). Thanks to their authors for their excellent work!

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@misc{rf-detr,
    title={RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
    author={Isaac Robinson and Peter Robicheaux and Matvei Popov and Deva Ramanan and Neehar Peri},
    year={2025},
    eprint={2511.09554},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2511.09554},
}
```

## Contribute

We welcome and appreciate all contributions! If you notice any issues or bugs, have questions, or would like to suggest new features, please [open an issue](https://github.com/roboflow/rf-detr-plus/issues/new) or pull request. By sharing your ideas and improvements, you help make RF-DETR better for everyone.

<p align="center">
    <a href="https://youtube.com/roboflow"><img src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://roboflow.com"><img src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://www.linkedin.com/company/roboflow-ai/"><img src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://docs.roboflow.com"><img src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://discuss.roboflow.com"><img src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://blog.roboflow.com"><img src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605" width="3%"/></a>
</p>
