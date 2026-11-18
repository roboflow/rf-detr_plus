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
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

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
from rfdetr.utilities.imports import _IS_PEFT_INSTALLED
from rfdetr.utilities.tensors import NestedTensor
from timm.layers import resample_abs_pos_embed

from rfdetr_plus import RFDETRAtto, RFDETRFemto, RFDETRPico
from rfdetr_plus.models import RFDETRAttoConfig, RFDETRFemtoConfig, RFDETRPicoConfig
from tests._markers import _requires_onnx

_requires_peft = pytest.mark.skipif(not _IS_PEFT_INSTALLED, reason="peft not installed")
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
        pytest.param("femto", (384, 640), id="femto-384x640"),
        pytest.param("pico", (480, 640), id="pico-480x640"),
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


@_requires_onnx
@pytest.mark.parametrize("name", _SIZES)
def test_onnx_export(name: str, fake_checkpoints: dict[str, Path], tmp_path: Path) -> None:
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

    pos_embed_after = model.model.model.state_dict()[_POS_EMBED_KEY]
    # Trained (it moved) but not resampled in place: an in-place resize changes it by ~20% within the first
    # off-resolution steps; one optimizer step at the embedding's decayed learning rate moves it far less.
    assert not torch.equal(pos_embed_after, pos_embed_before)
    assert float((pos_embed_after - pos_embed_before).norm() / pos_embed_before.norm()) < 0.01
    checkpoints = sorted(tmp_path.glob("checkpoint*.pth"))
    assert checkpoints
    reloaded = from_checkpoint(str(checkpoints[-1]))
    assert isinstance(reloaded, RFDETRAtto)
    assert isinstance(reloaded.model_config, ModelConfig)
    assert reloaded.model_config.encoder == "pe_core_t"


@_requires_peft
def test_backbone_lora_adapts_the_pe_attention(fake_checkpoints: dict[str, Path]) -> None:
    """``backbone_lora=True`` wraps PE-Core-T's attention qkv projections and trains only the adapters."""
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
    """``RFDETR.inference()`` (export graph, optionally traced) matches eager ``predict`` in fp32 and runs in bf16."""
    model = _load("femto", fake_checkpoints["femto"])
    image = np.random.default_rng(0).integers(0, 255, (384, 384, 3), dtype=np.uint8)
    eager = model.predict(image, threshold=0.0)

    predictor = model.inference(compile=compile_, dtype=dtype)
    optimized = predictor.predict(image, threshold=0.0)

    assert len(optimized) == len(eager)
    assert all(p.dtype == dtype for p in predictor._native_model.model.inference_model.parameters())
    if dtype == torch.float32:
        scores, eager_scores = np.sort(optimized.confidence)[::-1], np.sort(eager.confidence)[::-1]
        np.testing.assert_allclose(scores, eager_scores, rtol=0, atol=1e-5)
    else:
        # bf16 scores drift from fp32's by up to ~0.05, as core models' do, and vary by CPU, so they are not compared.
        assert np.isfinite(optimized.confidence).all()


_SIZE_BUDGET = 8_000_000
_MIN_LOGIT_CORRELATION = 0.98


class _CpuCase(NamedTuple):
    """Per-size recipe of the compact CPU export of a fine-tuned detector."""

    name: str
    resolution: int | None
    size_limit: int | None
    matmul_integer: int


class _CpuExport(NamedTuple):
    """One shared compact export of a custom detector, with the FP32 export of the same weights."""

    case: _CpuCase
    path: Path
    fp32_path: Path
    resolution: int


_ATTO_CASE = pytest.param(_CpuCase("atto", 400, _SIZE_BUDGET, 42), id="atto")
_CPU_CASES = [
    _ATTO_CASE,
    pytest.param(_CpuCase("femto", None, None, 37), id="femto"),
    pytest.param(_CpuCase("pico", None, None, 37), id="pico"),
]


def _custom_detector(case: _CpuCase, checkpoint: Path) -> RFDETR:
    """A three-class detector (four logits) with a learned, non-degenerate classifier and distinct class biases."""
    model = _load(case.name, checkpoint, **({} if case.resolution is None else {"resolution": case.resolution}))
    network = model.model.model
    torch.manual_seed(0)
    network.reinitialize_detection_head(4)
    model.model_config.num_classes = 3
    with torch.no_grad():
        for module_name, module in network.named_modules():
            if isinstance(module, torch.nn.Linear) and "class_embed" in module_name:
                module.weight.normal_(0, 0.02)
                module.bias.copy_(torch.tensor([2.5, -3.0, 0.25, 0.5]))
    return model


def _run_onnx(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Boxes and logits of a seeded random image."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    session = onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    image = np.random.default_rng(42).normal(size=input_info.shape).astype(np.float32)
    boxes, logits = session.run(["dets", "labels"], {input_info.name: image})
    return boxes, logits


@pytest.fixture(scope="module")
def cpu_export(
    request: pytest.FixtureRequest, fake_checkpoints: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> _CpuExport:
    """One compact export (and one FP32 export of the same weights) per size, shared by every test of that size."""
    case: _CpuCase = request.param
    model = _custom_detector(case, fake_checkpoints[case.name])
    root = tmp_path_factory.mktemp(f"cpu_export_{case.name}")
    fp32_path = Path(model.export(format="onnx", output_dir=str(root / "fp32"), verbose=False))
    path = model.export_cpu_onnx(root / "cpu", size_limit_bytes=case.size_limit)
    return _CpuExport(case, path, fp32_path, model.model_config.resolution)


@_requires_onnx
@pytest.mark.parametrize("cpu_export", _CPU_CASES, indirect=True)
class TestCpuOnnxExport:
    """The compact CPU export of a fine-tuned detector, one concern per test."""

    def test_operator_census(self, cpu_export: _CpuExport) -> None:
        import onnx

        counts = Counter(node.op_type for node in onnx.load(cpu_export.path).graph.node)

        expected = {"RotaryEmbedding": 24, "Attention": 12, "MatMulInteger": cpu_export.case.matmul_integer}
        assert {op: counts[op] for op in expected} == expected

    def test_input_matches_the_configured_resolution(self, cpu_export: _CpuExport) -> None:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(cpu_export.path), providers=["CPUExecutionProvider"])

        assert session.get_inputs()[0].shape[-2:] == [cpu_export.resolution] * 2

    def test_outputs_have_four_columns_per_query(self, cpu_export: _CpuExport) -> None:
        boxes, logits = _run_onnx(cpu_export.path)

        assert boxes.shape == logits.shape
        assert boxes.shape[0] == 1
        assert boxes.shape[-1] == 4

    def test_outputs_are_finite(self, cpu_export: _CpuExport) -> None:
        boxes, logits = _run_onnx(cpu_export.path)

        assert np.isfinite(boxes).all()
        assert np.isfinite(logits).all()

    def test_logits_track_the_fp32_export(self, cpu_export: _CpuExport) -> None:
        """INT8/FP16 storage and dynamic activation quantization keep the FP32 logits' pattern.

        Boxes are not compared: an untrained detector saturates them at the clamp limits, where one quantization step
        flips a coordinate, so they carry no signal. The logits of this seeded detector correlate at ~0.99 with FP32's
        (and a mis-scaled or mis-oriented INT8 channel drops that far below the bound); the bound leaves margin for
        CPU-to-CPU kernel differences.
        """
        _, fp32_logits = _run_onnx(cpu_export.fp32_path)
        _, compact_logits = _run_onnx(cpu_export.path)

        correlation = np.corrcoef(fp32_logits.ravel(), compact_logits.ravel())[0, 1]
        assert correlation > _MIN_LOGIT_CORRELATION


@_requires_onnx
@pytest.mark.parametrize("cpu_export", [_ATTO_CASE], indirect=True)
class TestCpuOnnxSizeBudget:
    """Atto is the size that must fit the deployment budget."""

    def test_size_stays_below_the_budget(self, cpu_export: _CpuExport) -> None:
        assert cpu_export.path.stat().st_size < _SIZE_BUDGET


def test_failed_cpu_export_preserves_previous_artifact(
    fake_checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted trace must restore ordinary export mode and leave the previous result readable."""
    from rfdetr_plus.models.pe_core import _CPU_EXPORT

    model = _load("atto", fake_checkpoints["atto"])
    artifact = tmp_path / "rfdetr-atto-cpu.onnx"
    artifact.write_bytes(b"previous export")

    def fail_trace(**kwargs: object) -> None:
        assert _CPU_EXPORT.get()
        raise ValueError("test trace failed")

    monkeypatch.setattr(model, "export", fail_trace)
    with pytest.raises(ValueError, match="test trace failed"):
        model.export_cpu_onnx(tmp_path)
    assert not _CPU_EXPORT.get()
    assert artifact.read_bytes() == b"previous export"
    assert list(tmp_path.iterdir()) == [artifact]


@_requires_onnx
@pytest.mark.parametrize("cpu_export", [_ATTO_CASE], indirect=True)
class TestCpuExportSizeBoundary:
    """``size_limit_bytes`` is strict: the file must be smaller than the limit, and a failure keeps the old artifact."""

    def test_limit_equal_to_the_size_is_rejected(
        self, cpu_export: _CpuExport, fake_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        size = cpu_export.path.stat().st_size
        model = _custom_detector(cpu_export.case, fake_checkpoints["atto"])

        with pytest.raises(ValueError, match=f"must be smaller than {size} bytes"):
            model.export_cpu_onnx(tmp_path, size_limit_bytes=size)

    def test_rejection_after_quantization_keeps_the_previous_artifact_and_leaves_no_workspace(
        self, cpu_export: _CpuExport, fake_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        artifact = tmp_path / "rfdetr-atto-cpu.onnx"
        artifact.write_bytes(b"previous")
        model = _custom_detector(cpu_export.case, fake_checkpoints["atto"])

        with pytest.raises(ValueError, match="must be smaller than"):
            model.export_cpu_onnx(tmp_path, size_limit_bytes=cpu_export.path.stat().st_size)

        assert list(tmp_path.iterdir()) == [artifact]
        assert artifact.read_bytes() == b"previous"

    @pytest.mark.parametrize("as_limit", [int, np.int64])
    def test_limit_one_byte_above_the_size_is_accepted_and_replaces_the_previous_artifact(
        self, as_limit: type, cpu_export: _CpuExport, fake_checkpoints: dict[str, Path], tmp_path: Path
    ) -> None:
        size = cpu_export.path.stat().st_size
        (tmp_path / "rfdetr-atto-cpu.onnx").write_bytes(b"previous")
        model = _custom_detector(cpu_export.case, fake_checkpoints["atto"])

        path = model.export_cpu_onnx(tmp_path, size_limit_bytes=as_limit(size + 1))

        assert path.stat().st_size == size


@pytest.mark.parametrize("name", ["femto", "pico"])
def test_cpu_export_rejects_femto_and_pico_at_the_deployment_budget(
    name: str, fake_checkpoints: dict[str, Path], tmp_path: Path
) -> None:
    """Only Atto fits 8,000,000 bytes; the larger sizes fail loudly instead of writing an over-budget file."""
    model = _load(name, fake_checkpoints[name])

    with pytest.raises(ValueError, match=f"must be smaller than {_SIZE_BUDGET} bytes"):
        model.export_cpu_onnx(tmp_path, size_limit_bytes=_SIZE_BUDGET)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("size_limit", [True, 1.5, 8_000_000.0, "8000000"])
def test_cpu_export_rejects_a_non_integer_size_limit_before_writing(
    size_limit: object, fake_checkpoints: dict[str, Path], tmp_path: Path
) -> None:
    model = _load("atto", fake_checkpoints["atto"])

    with pytest.raises(TypeError):
        model.export_cpu_onnx(tmp_path, size_limit_bytes=size_limit)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("size_limit", [0, -1])
def test_cpu_export_rejects_a_non_positive_size_limit_before_writing(
    size_limit: int, fake_checkpoints: dict[str, Path], tmp_path: Path
) -> None:
    model = _load("atto", fake_checkpoints["atto"])

    with pytest.raises(ValueError, match="positive"):
        model.export_cpu_onnx(tmp_path, size_limit_bytes=size_limit)

    assert list(tmp_path.iterdir()) == []


def _traced_graph_export(
    attention_nodes: int, matmul_names: list[str], *, custom_domain: bool = True, rotary: bool = True
):
    """A stand-in for ``RFDETR.export`` that writes a tiny graph with the given operator inventory.

    The base TorchScript exporter emits attention and rotary nodes in the private ``rfdetr_pe`` domain, so the stand-in
    does too unless ``custom_domain`` or ``rotary`` is switched off.
    """

    def export(*, output_dir: str, **kwargs: object) -> str:
        import onnx
        from onnx import TensorProto, helper

        domain = "rfdetr_pe" if custom_domain else ""
        nodes = [
            helper.make_node("Attention", ["x", "x", "x"], [f"a{i}"], domain=domain) for i in range(attention_nodes)
        ]
        if rotary:
            nodes.append(helper.make_node("RotaryEmbedding", ["x", "x", "x"], ["rot"], domain=domain))
        nodes += [helper.make_node("MatMul", ["x", "x"], [f"m{i}"], name=name) for i, name in enumerate(matmul_names)]
        outputs = [helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, [2, 2]) for node in nodes]
        graph = helper.make_graph(
            nodes, "traced", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])], outputs
        )
        opsets = [helper.make_opsetid("", 23)]
        if custom_domain:
            opsets.append(helper.make_opsetid("rfdetr_pe", 1))
        model = helper.make_model(graph, opset_imports=opsets, ir_version=10)
        path = Path(output_dir) / "traced.onnx"
        onnx.save(model, path)
        return str(path)

    return export


_ENCODER_PROJECTIONS = [
    f"/encoder/blocks.{block}/{projection}/MatMul"
    for block in range(12)
    for projection in ("attn/qkv", "attn/proj", "mlp/fc1")
]


@_requires_onnx
class TestCpuExportGraphGuards:
    """The export refuses a traced graph whose operator inventory differs from the recipe, before quantizing."""

    def test_missing_fused_attention_operators_are_rejected(
        self, fake_checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _load("femto", fake_checkpoints["femto"])
        monkeypatch.setattr(model, "export", _traced_graph_export(1, _ENCODER_PROJECTIONS))

        with pytest.raises(ValueError, match="Expected one fused attention operator in each PE block"):
            model.export_cpu_onnx(tmp_path)

    @pytest.mark.parametrize(
        ("stand_in", "missing"),
        [
            pytest.param(
                _traced_graph_export(12, _ENCODER_PROJECTIONS, custom_domain=False), "Attention", id="default-domain"
            ),
            pytest.param(
                _traced_graph_export(12, _ENCODER_PROJECTIONS, rotary=False), "RotaryEmbedding", id="no-rotary"
            ),
        ],
    )
    def test_trace_without_the_custom_nodes_is_rejected(
        self,
        stand_in: object,
        missing: str,
        fake_checkpoints: dict[str, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _load("femto", fake_checkpoints["femto"])
        monkeypatch.setattr(model, "export", stand_in)

        with pytest.raises(ValueError, match=f"no rfdetr_pe::{missing} node"):
            model.export_cpu_onnx(tmp_path)

    def test_missing_encoder_projections_are_rejected(
        self, fake_checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _load("femto", fake_checkpoints["femto"])
        monkeypatch.setattr(model, "export", _traced_graph_export(12, []))

        with pytest.raises(ValueError, match="Expected three encoder projections in each of twelve PE blocks, found 0"):
            model.export_cpu_onnx(tmp_path)

    @pytest.mark.parametrize(("name", "expected"), [("atto", 42), ("femto", 37), ("pico", 37)])
    def test_missing_size_specific_projections_are_rejected(
        self,
        name: str,
        expected: int,
        fake_checkpoints: dict[str, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _load(name, fake_checkpoints[name])
        monkeypatch.setattr(model, "export", _traced_graph_export(12, _ENCODER_PROJECTIONS))

        with pytest.raises(ValueError, match=f"Expected {expected} linear projections for rfdetr-{name}, found 36"):
            model.export_cpu_onnx(tmp_path)


@pytest.mark.parametrize("missing", ["onnx", "onnxruntime.quantization"])
def test_cpu_export_names_the_extra_when_an_optional_dependency_is_missing(
    missing: str, fake_checkpoints: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _load("atto", fake_checkpoints["atto"])
    monkeypatch.setitem(sys.modules, missing, None)

    with pytest.raises(ImportError, match=re.escape('pip install "rfdetr-plus[onnx]"')):
        model.export_cpu_onnx(tmp_path)
