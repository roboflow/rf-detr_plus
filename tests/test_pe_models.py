# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""End-to-end tests of RFDETRAtto / RFDETRFemto / RFDETRPico through the public rfdetr API, fully offline.

Every model loads a fake release checkpoint: random weights built by rfdetr's own ``build_model`` and saved in the
release format (``{"model": state_dict}``), so no pretrained weights are downloaded.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import supervision as sv
import torch
from rfdetr import from_checkpoint
from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.detr import RFDETR
from rfdetr.export._backend import _switch_to_export_mode
from rfdetr.export.prepare import prepare_export_graph
from rfdetr.models import build_model
from rfdetr.utilities.tensors import NestedTensor
from timm.layers import resample_abs_pos_embed

from rfdetr_plus import RFDETRAtto, RFDETRFemto, RFDETRPico
from rfdetr_plus.models import RFDETRAttoConfig, RFDETRFemtoConfig, RFDETRPicoConfig

_POS_EMBED_KEY = "backbone.0.encoder.model.pos_embed"
_MODELS = {
    "atto": (RFDETRAtto, RFDETRAttoConfig),
    "femto": (RFDETRFemto, RFDETRFemtoConfig),
    "pico": (RFDETRPico, RFDETRPicoConfig),
}
_SIZES = [pytest.param(name, id=name) for name in _MODELS]


@pytest.fixture(scope="module")
def fake_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Random-weight checkpoints in the release format, one per PE-Core-T size.

    The config's ``pretrain_weights`` is set to a placeholder only so ``build_model`` skips the upstream PE-Core-T
    download (it loads encoder weights only when no detector checkpoint is configured).
    """
    root = tmp_path_factory.mktemp("fake_pe_checkpoints")
    paths = {}
    for name, (_, config_cls) in _MODELS.items():
        torch.manual_seed(0)
        config = config_cls(pretrain_weights="placeholder.pth")
        model = build_model(_namespace_from_configs(config, TrainConfig(dataset_dir=str(root))))
        paths[name] = root / f"fake-{name}.pth"
        torch.save({"model": model.state_dict()}, paths[name])
    return paths


def _load(name: str, checkpoint: Path, **kwargs) -> RFDETR:
    model_cls, _ = _MODELS[name]
    return model_cls(pretrain_weights=str(checkpoint), device="cpu", **kwargs)


@pytest.mark.parametrize("name", _SIZES)
def test_checkpoint_loads_every_weight(name: str, fake_checkpoints: dict[str, Path]) -> None:
    model = _load(name, fake_checkpoints[name])

    expected = torch.load(fake_checkpoints[name], map_location="cpu", weights_only=True)["model"]
    loaded = model.model.model.state_dict()
    assert set(loaded) == set(expected)
    assert all(torch.equal(loaded[key], expected[key]) for key in expected)


@pytest.mark.parametrize("name", _SIZES)
def test_predict_on_a_synthetic_image(name: str, fake_checkpoints: dict[str, Path]) -> None:
    model = _load(name, fake_checkpoints[name])
    image = np.random.default_rng(0).integers(0, 255, (300, 420, 3), dtype=np.uint8)

    detections = model.predict(image, threshold=0.0)

    assert isinstance(detections, sv.Detections)
    assert 0 < len(detections) <= model.model_config.num_select


@pytest.mark.parametrize(
    ("name", "resolution"),
    [pytest.param("atto", 400, id="atto-400"), pytest.param("femto", 448, id="femto-448")],
)
def test_custom_resolution_resamples_the_checkpoint_position_embedding(
    name: str, resolution: int, fake_checkpoints: dict[str, Path]
) -> None:
    model = _load(name, fake_checkpoints[name], resolution=resolution)

    config = model.model_config
    grid = resolution // config.patch_size
    stored = torch.load(fake_checkpoints[name], map_location="cpu", weights_only=True)["model"][_POS_EMBED_KEY]
    stored_grid = math.isqrt(stored.shape[1] - 1)
    expected = resample_abs_pos_embed(stored, new_size=(grid, grid), old_size=(stored_grid, stored_grid))
    assert config.positional_encoding_size == grid
    assert torch.equal(model.model.model.state_dict()[_POS_EMBED_KEY], expected)
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    assert isinstance(model.predict(image, threshold=0.0), sv.Detections)


def _prepared_export_graph(model: RFDETR, shape: tuple[int, int]):
    """The graph ``RFDETR.export`` hands every exporter, prepared exactly as rfdetr does."""
    graph = prepare_export_graph(copy.deepcopy(model.model.model), model.model_config, shape=shape, device="cpu")
    _switch_to_export_mode(graph.model)
    return graph


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        pytest.param("atto", None, id="atto-native"),
        pytest.param("atto", (440, 440), id="atto-440"),
        pytest.param("femto", None, id="femto-native"),
        pytest.param("pico", (480, 480), id="pico-480"),
    ],
)
def test_export_graph_traces_without_runtime_resampling(
    name: str, shape: tuple[int, int] | None, fake_checkpoints: dict[str, Path]
) -> None:
    """The export graph bakes position embeddings for its shape and matches the eager model there.

    ONNX has no antialiased bicubic interpolation, so the traced graph must not resample pos_embed at runtime, even
    when exporting at a shape other than the model's resolution.
    """
    model = _load(name, fake_checkpoints[name])
    resolution = model.model_config.resolution
    shape = shape or (resolution, resolution)
    graph = _prepared_export_graph(model, shape)

    with torch.no_grad():
        traced = torch.jit.trace(graph.model, graph.input_tensors, check_trace=False)
        exported = traced(graph.input_tensors)
        eager_model = model.model.model.eval()
        eager = eager_model(NestedTensor(graph.input_tensors, torch.zeros(1, *shape, dtype=torch.bool)))

    ops = str(traced.inlined_graph)
    assert "upsample_bicubic2d" not in ops
    torch.testing.assert_close(exported[0], eager["pred_boxes"], rtol=0, atol=1e-5)
    torch.testing.assert_close(exported[1], eager["pred_logits"], rtol=0, atol=1e-5)


@pytest.mark.parametrize("name", _SIZES)
def test_onnx_export(name: str, fake_checkpoints: dict[str, Path], tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    model = _load(name, fake_checkpoints[name])

    output = model.export(format="onnx", output_dir=str(tmp_path))

    assert Path(output).is_file()
    assert Path(output).stat().st_size > 0


def test_training_updates_the_position_embedding(
    fake_checkpoints: dict[str, Path], synthetic_roboflow_dataset: Path, tmp_path: Path
) -> None:
    """One multi-scale epoch on CPU trains, saves a checkpoint that reloads as RFDETRAtto, and moves pos_embed.

    Multi-scale batches run at resolutions other than the model's; pos_embed must stay the optimized parameter there.
    """
    model = _load("atto", fake_checkpoints["atto"])
    pos_embed_before = model.model.model.state_dict()[_POS_EMBED_KEY].clone()

    model.train(
        dataset_dir=str(synthetic_roboflow_dataset),
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        num_workers=0,
        device="cpu",
        use_ema=False,
        multi_scale=True,
        tensorboard=False,
        eval_backend="faster_coco_eval",  # any COCO backend works here; this one ships in rfdetr[train]
    )

    assert not torch.equal(model.model.model.state_dict()[_POS_EMBED_KEY], pos_embed_before)
    checkpoints = sorted(tmp_path.glob("checkpoint*.pth"))
    assert checkpoints
    reloaded = from_checkpoint(str(checkpoints[-1]))
    assert isinstance(reloaded, RFDETRAtto)
    assert isinstance(reloaded.model_config, ModelConfig)
    assert reloaded.model_config.encoder == "pe_core_t"


def test_backbone_lora_adapts_the_pe_attention(fake_checkpoints: dict[str, Path]) -> None:
    """``backbone_lora=True`` wraps PE-Core-T's attention qkv projections and trains only the adapters."""
    pytest.importorskip("peft")
    model = _load("atto", fake_checkpoints["atto"], backbone_lora=True)
    network = model.model.model

    encoder_trainable = [name for name, p in network.named_parameters() if ".encoder." in name and p.requires_grad]
    assert encoder_trainable
    assert all("lora_" in name and ".attn.qkv." in name for name in encoder_trainable)
    network.train()
    images = torch.randn(1, 3, 420, 420)
    network(NestedTensor(images, torch.zeros(1, 420, 420, dtype=torch.bool)))["pred_logits"].sum().backward()
    assert all(p.grad is not None for name, p in network.named_parameters() if name in set(encoder_trainable))


@pytest.mark.parametrize(
    ("compile_", "dtype"),
    [
        pytest.param(False, torch.float32, id="eager-fp32"),
        pytest.param(True, torch.float32, id="traced-fp32"),
        pytest.param(True, torch.bfloat16, id="traced-bf16"),
    ],
)
def test_inference_optimization_matches_eager_predictions(
    compile_: bool, dtype: torch.dtype, fake_checkpoints: dict[str, Path]
) -> None:
    """``RFDETR.inference()`` (export graph, optional TorchScript trace and dtype cast) reproduces eager ``predict``."""
    model = _load("femto", fake_checkpoints["femto"])
    image = np.random.default_rng(0).integers(0, 255, (384, 384, 3), dtype=np.uint8)
    eager = model.predict(image, threshold=0.0)

    model.inference(compile=compile_, dtype=dtype)
    optimized = model.predict(image, threshold=0.0)

    assert len(optimized) == len(eager)
    scores, eager_scores = np.sort(optimized.confidence)[::-1], np.sort(eager.confidence)[::-1]
    if dtype == torch.float32:
        np.testing.assert_allclose(scores, eager_scores, rtol=0, atol=1e-5)
    else:
        np.testing.assert_allclose(scores[:50], eager_scores[:50], rtol=0, atol=1e-2)
