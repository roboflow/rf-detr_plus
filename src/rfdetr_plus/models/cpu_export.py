# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CPU ONNX export for the fixed PE-Core-T detector family."""

# Exporter coupling: this module post-processes the graph that the base ``RFDETR.export(format="onnx")`` writes.
# That export calls ``torch.onnx.export`` with ``dynamo=False`` (the TorchScript exporter), which is what makes two
# assumptions hold. First, the custom ``symbolic()`` methods in ``pe_core`` emit ``rfdetr_pe::Attention`` and
# ``rfdetr_pe::RotaryEmbedding`` nodes. Second, node names carry TorchScript module scopes
# (``/encoder/blocks.N/...``, ``/projector/...``, ``/decoder/layers.N/...``), which the projection regexes and
# compaction scopes below select on. If the base export moves to the dynamo exporter, or a module is renamed, export
# fails loudly instead of writing a degraded file: ``_check_custom_ops`` rejects a trace without the custom nodes,
# ``_select_projections`` and the ``MatMulInteger`` count reject drifted projection names, and ``_check_compacted``
# rejects drifted compaction scopes.

from __future__ import annotations

import numbers
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx import GraphProto, ModelProto, TensorProto
    from rfdetr.detr import RFDETR


@dataclass(frozen=True)
class _CPURecipe:
    """Graph recipe for one PE-Core-T size, applied on top of the 36 encoder attention and ``fc1`` projections.

    Attributes:
        extra_projections: Regex over traced node names selecting the MatMuls quantized besides those 36.
        projections: Total number of MatMuls quantized to dynamic INT8.
        min_dec_layers: Smallest ``model_config.dec_layers`` that ``extra_projections`` can match.
        projector_matmuls: Whether the 1x1 projector convolutions are rewritten as NHWC MatMuls.
        pack_int8: Whether the remaining large filters are stored as INT8 codes instead of FP16.
        compacted_scopes: Scopes in which at least one filter must be compacted (Atto has no decoder layers).
    """

    extra_projections: str
    projections: int
    min_dec_layers: int
    projector_matmuls: bool
    pack_int8: bool
    compacted_scopes: tuple[str, ...]


#: Node-name scopes whose large FP32 filters are stored compactly, in match priority order.
_COMPACT_SCOPES = ("/encoder/", "/projector/", "/decoder/")


#: Export recipe per supported ``detector.size``. Femto and Pico quantize one projection of decoder ``layers.1``,
#: which exists only with at least two decoder layers; Atto quantizes ``fc2`` of the last six encoder blocks instead.
_RECIPES: dict[str, _CPURecipe] = {
    "rfdetr-atto": _CPURecipe(
        extra_projections=r"/encoder/blocks\.(?:[6-9]|10|11)/mlp/fc2/MatMul$",
        projections=42,
        min_dec_layers=0,
        projector_matmuls=True,
        pack_int8=True,
        compacted_scopes=("/projector/",),
    ),
    "rfdetr-femto": _CPURecipe(
        extra_projections=r"/decoder/layers\.1/linear2/MatMul$",
        projections=37,
        min_dec_layers=2,
        projector_matmuls=False,
        pack_int8=False,
        compacted_scopes=("/projector/", "/decoder/"),
    ),
    "rfdetr-pico": _CPURecipe(
        extra_projections=r"/decoder/layers\.1/linear1/MatMul$",
        projections=37,
        min_dec_layers=2,
        projector_matmuls=True,
        pack_int8=False,
        compacted_scopes=("/projector/", "/decoder/"),
    ),
}

#: Extensions stripped from a caller-supplied ``output_name``; the same set the base ``RFDETR.export`` strips, so a
#: dot that belongs to the name itself (``"detector-v1.2"``) is kept.
_ARTIFACT_EXTENSIONS = frozenset(
    {".aimodel", ".bin", ".engine", ".mlmodel", ".mlpackage", ".onnx", ".plan", ".pte", ".tflite", ".trt", ".xml"}
)


def _output_stem(size: str, output_name: str | None) -> str:
    """Return the artifact file stem, following the full-stem semantics of the base ``export(output_name=...)``.

    Args:
        size: Model variant name used for the default ``{size}-cpu`` stem.
        output_name: Caller-supplied full stem, or ``None``/empty for the default. Only the basename is kept
            (directory components are dropped) and a known export extension such as ``.onnx`` is stripped.

    Returns:
        The stem of the ``.onnx`` file to write.

    Raises:
        ValueError: If ``output_name`` reduces to an empty stem.
    """
    if not output_name:
        return f"{size}-cpu"
    stem, extension = os.path.splitext(re.split(r"[\\/]", output_name)[-1])
    if extension.lower() not in _ARTIFACT_EXTENSIONS:
        stem += extension
    if not stem:
        raise ValueError("output_name must resolve to a non-empty filename stem")
    return stem


def _check_custom_ops(model: ModelProto) -> None:
    """Require the ``rfdetr_pe`` attention and rotary nodes that only the TorchScript exporter emits.

    Args:
        model: Graph traced by the base ONNX export.

    Raises:
        ValueError: If the graph has no ``rfdetr_pe::Attention`` or no ``rfdetr_pe::RotaryEmbedding`` node.
    """
    found = {node.op_type for node in model.graph.node if node.domain == "rfdetr_pe"}
    missing = sorted({"Attention", "RotaryEmbedding"} - found)
    if missing:
        raise ValueError(
            f"The traced graph has no rfdetr_pe::{missing[0]} node; compact CPU export needs the TorchScript ONNX "
            "exporter (torch.onnx.export with dynamo=False) that the base RFDETR.export uses"
        )


def _upgrade_attention(model: ModelProto) -> ModelProto:
    """Upgrade the traced graph before resolving the two symbolic ops introduced in ONNX 23."""
    from onnx import version_converter

    model = version_converter.convert_version(model, 23)
    for node in model.graph.node:
        if node.domain == "rfdetr_pe" and node.op_type in {"Attention", "RotaryEmbedding"}:
            node.domain = ""
    for opset in list(model.opset_import):
        if opset.domain == "rfdetr_pe":
            model.opset_import.remove(opset)
    return model


def _compaction_candidates(graph: GraphProto, weights: dict[str, TensorProto]) -> dict[str, str]:
    """Map each FP32 Conv/MatMul filter inside a compactable scope to the first scope of the node that uses it."""
    from onnx import TensorProto

    candidates: dict[str, str] = {}
    for node in graph.node:
        scope = next((scope for scope in _COMPACT_SCOPES if scope in node.name), None)
        if (
            scope is not None
            and node.op_type in {"Conv", "MatMul"}
            and len(node.input) > 1
            and node.input[1] in weights
            and weights[node.input[1]].data_type == TensorProto.FLOAT
        ):
            candidates.setdefault(node.input[1], scope)
    return candidates


def _compact_weights(model: ModelProto, *, pack_int8: bool) -> Counter[str]:
    """Store large floating-point filters compactly and reconstruct their FP32 values at session setup.

    Args:
        model: Quantized model, edited in place.
        pack_int8: Store filters as symmetric per-channel INT8 codes instead of FP16.

    Returns:
        Number of filters rewritten per scope in ``_COMPACT_SCOPES``; a scope with no rewritten filter is absent,
        so callers can fail closed when scope names drift.

    Raises:
        ValueError: If a filter holds nonfinite or FP16-overflowing values, or a storage name is already taken.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    weights = {tensor.name: tensor for tensor in graph.initializer}
    candidates = _compaction_candidates(graph, weights)
    compacted: Counter[str] = Counter()
    casts = []
    used = set(weights) | {name for node in graph.node for name in (*node.input, *node.output)}
    for name in sorted(candidates):
        tensor = weights[name]
        values = numpy_helper.to_array(tensor)
        if values.size < 4096:
            continue
        # Every path below either rewrites this filter or raises, so the count reflects rewritten filters only.
        compacted[candidates[name]] += 1
        if pack_int8:
            # Symmetric per-output-channel storage. Cast and scale only constant weights
            # so the runtime can fold reconstruction at session creation.
            axis = 1 if values.ndim == 2 else 0
            axes = tuple(i for i in range(values.ndim) if i != axis)
            scales = np.max(np.abs(values), axis=axes, keepdims=True) / 127
            scales[scales == 0] = 1
            if not np.isfinite(scales).all():
                raise ValueError(f"Nonfinite weight in {name}")
            packed, scale, expanded = (name + suffix for suffix in ("_cpu_i8", "_cpu_scale", "_cpu_expand"))
            if used.intersection((packed, scale, expanded)):
                raise ValueError(f"Cannot allocate storage names for {name}")
            codes = np.rint(values / scales).clip(-127, 127).astype(np.int8)
            tensor.CopyFrom(numpy_helper.from_array(codes, packed))
            graph.initializer.append(numpy_helper.from_array(scales, scale))
            casts.extend(
                [
                    helper.make_node("Cast", [packed], [expanded], to=TensorProto.FLOAT),
                    helper.make_node("Mul", [expanded, scale], [name]),
                ]
            )
            continue
        compact = name + "_cpu_fp16"
        if compact in used:
            raise ValueError(f"Tensor name already in use: {compact}")
        if not np.isfinite(values).all() or np.abs(values).max() > np.finfo(np.float16).max:
            raise ValueError(f"Cannot store {name} in FP16")
        tensor.CopyFrom(numpy_helper.from_array(values.astype(np.float16), compact))
        casts.append(helper.make_node("Cast", [compact], [name], to=TensorProto.FLOAT))
    for node in reversed(casts):
        graph.node.insert(0, node)
    return compacted


def _check_compacted(compacted: Counter[str], size: str, recipe: _CPURecipe) -> None:
    """Fail closed when a scope the recipe expects to shrink kept all of its filters in FP32.

    Args:
        compacted: Rewritten-filter counts per scope, as returned by ``_compact_weights``.
        size: Detector size, used in the error message.
        recipe: Export recipe for ``size``.

    Raises:
        ValueError: Naming the first expected scope in which no filter was compacted.
    """
    for scope in recipe.compacted_scopes:
        if not compacted.get(scope):
            raise ValueError(
                f"No {scope} filter was compacted for {size}; the traced node names no longer match the "
                "TorchScript scopes this export expects"
            )


def _share_constants(model: ModelProto) -> None:
    """Share identical traced rotary tables across blocks."""
    outputs = {output.name for output in model.graph.output}
    seen = {}
    aliases = {}
    kept = []
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in aliases:
                node.input[index] = aliases[name]
        if (
            node.op_type == "Constant"
            and node.output[0] not in outputs
            and len(node.attribute) == 1
            and node.attribute[0].name == "value"
        ):
            tensor = node.attribute[0].t
            key = (tensor.data_type, tuple(tensor.dims), tensor.raw_data)
            if tensor.raw_data and key in seen:
                aliases[node.output[0]] = seen[key]
                continue
            seen[key] = node.output[0]
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)


def _projector_matmuls(graph: GraphProto) -> None:
    """Rewrite the two channel projections as pointwise matrix multiplies on NHWC features."""
    from onnx import helper, numpy_helper

    tensors = {tensor.name: tensor for tensor in graph.initializer}
    nodes = []
    for node in graph.node:
        if node.op_type != "Conv" or "/projector/" not in node.name:
            nodes.append(node)
            continue
        weight = tensors[node.input[1]]
        if list(weight.dims[2:]) != [1, 1]:
            nodes.append(node)
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (
            attributes.get("group", 1) != 1
            or any(attributes.get("pads", []))
            or any(stride != 1 for stride in attributes.get("strides", []))
            or attributes.get("auto_pad", b"NOTSET") != b"NOTSET"
        ):
            raise ValueError(f"Unsupported spatial projection: {node.name}")
        prefix = node.name + "_cpu"
        matrix = prefix + "_weight"
        graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(weight)[:, :, 0, 0].T.copy(), matrix))
        nodes.append(helper.make_node("Transpose", [node.input[0]], [prefix + "_nhwc"], perm=[0, 2, 3, 1]))
        nodes.append(helper.make_node("MatMul", [prefix + "_nhwc", matrix], [prefix + "_product"], name=prefix))
        value = prefix + "_product"
        if len(node.input) > 2:
            nodes.append(helper.make_node("Add", [value, node.input[2]], [prefix + "_biased"]))
            value = prefix + "_biased"
        nodes.append(helper.make_node("Transpose", [value], list(node.output), perm=[0, 3, 1, 2]))
    del graph.node[:]
    graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input} | {output.name for output in graph.output}
    kept = [w for w in graph.initializer if w.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def _size_limit(size_limit_bytes: object) -> int | None:
    """Normalise an optional byte budget to a positive Python ``int``.

    Args:
        size_limit_bytes: ``None``, or any integral number such as ``int`` or ``numpy.int64``.

    Returns:
        ``None`` when no limit is given, else the limit as ``int``.

    Raises:
        TypeError: If the value is a ``bool`` or not an integral number (``float``, ``str``, ...).
        ValueError: If the value is zero or negative.
    """
    if size_limit_bytes is None:
        return None
    if isinstance(size_limit_bytes, bool) or not isinstance(size_limit_bytes, numbers.Integral):
        raise TypeError(f"size_limit_bytes must be an integer, got {type(size_limit_bytes).__name__}")
    limit = int(size_limit_bytes)
    if limit <= 0:
        raise ValueError(f"size_limit_bytes must be positive, got {limit}")
    return limit


def _recipe_for(detector: RFDETR) -> _CPURecipe:
    """Return the export recipe for ``detector``, rejecting an unsupported layout before the expensive trace.

    Args:
        detector: Detector to export.

    Returns:
        The recipe registered for ``detector.size``.

    Raises:
        ValueError: If ``size`` has no recipe, ``model_config.encoder`` is not the PE-Core-T encoder, or
            ``model_config.dec_layers`` is below what the recipe's selected projections need.
    """
    from rfdetr_plus.models.pe_core import PE_CORE_T_ENCODER

    size = getattr(detector, "size", None)
    recipe = _RECIPES.get(size) if isinstance(size, str) else None
    if recipe is None:
        raise ValueError(f"Compact CPU export supports size in {sorted(_RECIPES)}, got size={size!r}")
    config = detector.model_config
    if config.encoder != PE_CORE_T_ENCODER:
        raise ValueError(
            f"Compact CPU export of {size} needs model_config.encoder={PE_CORE_T_ENCODER!r}, got {config.encoder!r}"
        )
    if config.dec_layers < recipe.min_dec_layers:
        raise ValueError(
            f"Compact CPU export of {size} quantizes decoder layers.1 and needs "
            f"model_config.dec_layers >= {recipe.min_dec_layers}, got {config.dec_layers}"
        )
    return recipe


def _select_projections(graph: GraphProto, size: str, recipe: _CPURecipe) -> list[str]:
    """Return the names of the MatMuls to quantize, checking the per-block and per-recipe counts.

    Args:
        graph: Traced graph after the attention upgrade.
        size: Detector size, used in error messages.
        recipe: Export recipe for ``size``.

    Returns:
        Names of the selected MatMul nodes.

    Raises:
        ValueError: If the encoder or recipe projections are not found the expected number of times.
    """
    selected = [
        node.name
        for node in graph.node
        if node.op_type == "MatMul" and re.search(r"/encoder/blocks\.\d+/(attn/(qkv|proj)|mlp/fc1)/MatMul$", node.name)
    ]
    if len(selected) != 36:
        raise ValueError(f"Expected three encoder projections in each of twelve PE blocks, found {len(selected)}")
    selected.extend(node.name for node in graph.node if re.search(recipe.extra_projections, node.name))
    if len(selected) != recipe.projections:
        raise ValueError(f"Expected {recipe.projections} linear projections for {size}, found {len(selected)}")
    return selected


def export_cpu_onnx(
    detector: RFDETR,
    output_dir: str | Path = "output",
    *,
    size_limit_bytes: int | None = None,
    output_name: str | None = None,
) -> Path:
    """Export a loaded PE-Core-T detector to a compact ONNX file for ONNX Runtime on CPU.

    Experimental: the graph recipe, file layout and arguments may change in a minor release.

    This differs from the base ``detector.export(format="onnx", quantization="int8")``, which writes a static QDQ
    model with 8-bit weights and activations and requires ``calibration_data``. This export needs no calibration
    data: selected linear projections use dynamic INT8 (weight scales derive from the live checkpoint and ONNX
    Runtime computes activation scales at inference time), and the other large filters are stored compactly and
    expanded to FP32 when the session is created. The recipe depends on the size:

    - Atto: the encoder ``qkv``, ``proj`` and ``fc1`` projections of all twelve blocks plus ``fc2`` of blocks 6-11
      (42 MatMuls) use dynamic INT8, the 1x1 projector convolutions are rewritten as float NHWC MatMuls, and the
      remaining large filters are stored as per-channel INT8 codes.
    - Femto: the 36 encoder projections plus ``linear2`` of decoder ``layers.1`` (37 MatMuls) use dynamic INT8;
      the remaining large filters are stored in FP16.
    - Pico: the 36 encoder projections plus ``linear1`` of decoder ``layers.1`` (37 MatMuls) use dynamic INT8, the
      1x1 projector convolutions are rewritten as float NHWC MatMuls, and the remaining large filters are stored
      in FP16.

    The file targets the ONNX Runtime CPU execution provider, uses batch one at the configured model resolution,
    and uses opset 23 (``Attention`` and ``RotaryEmbedding``); the tested runtime floor is ``onnxruntime>=1.23.2``
    on the deployment host as well as on the exporting one.

    Args:
        detector: Loaded ``RFDETRAtto``, ``RFDETRFemto`` or ``RFDETRPico`` detector with the PE-Core-T encoder;
            Femto and Pico also need ``model_config.dec_layers >= 2``.
        output_dir: Directory the file is written to; created when missing.
        size_limit_bytes: Optional byte budget. When given, the complete uncompressed file must be strictly smaller
            than this many bytes. Any integral number is accepted (``int``, ``numpy.int64``, ...); ``bool`` is not.
            The file is written to a temporary path and moved into place only after every check passes, so a
            failed export, including one over budget, leaves any previous artifact at the target path in place.
        output_name: Full file stem, following the full-stem semantics of the base ``export(output_name=...)``:
            directory components (``/`` or ``\\``) are dropped, a known export extension such as ``.onnx`` is
            stripped case-insensitively, and other dots (``"atto-cpu-v1.2"``) are kept, so the file is
            ``{stem}.onnx``. ``None`` or ``""`` selects the default ``{detector.size}-cpu``.

    Returns:
        Path of the written ``.onnx`` file, ``output_dir / f"{stem}.onnx"``.

    Raises:
        TypeError: If ``size_limit_bytes`` is a ``bool`` or not an integral number. Checked before tracing.
        ValueError: Before tracing, if ``size_limit_bytes`` is zero or negative; if ``detector.size`` is not
            ``"rfdetr-atto"``, ``"rfdetr-femto"`` or ``"rfdetr-pico"``; if ``model_config.encoder`` is not the
            PE-Core-T encoder; if ``model_config.dec_layers`` is below 2 for Femto or Pico; or if ``output_name``
            reduces to an empty stem (for example ``"models/"``). After tracing, if the graph has no
            ``rfdetr_pe::Attention`` or ``rfdetr_pe::RotaryEmbedding`` node (the base export no longer uses the
            TorchScript exporter); if it does not hold exactly twelve fused ``Attention`` nodes; if the encoder
            ``qkv``/``proj``/``fc1`` projections are not exactly 36 or the size's recipe total (42 or 37) is not
            matched; if a 1x1 projector convolution has groups, padding, strides or ``auto_pad`` (Atto and Pico);
            if ONNX Runtime did not convert every selected projection to ``MatMulInteger``; if a filter holds
            nonfinite or FP16-overflowing values or its storage name is already taken; if a scope the recipe
            expects to shrink (``/projector/``, and ``/decoder/`` for Femto and Pico) has no compacted filter; or
            if the file reaches or exceeds ``size_limit_bytes``.
        ImportError: If ``onnx`` or ``onnxruntime`` is not installed; the message names the package extra to
            install. Raised only after the argument and recipe checks above.
        onnx.checker.ValidationError: If the final graph fails ``onnx.checker.check_model``.

    Examples:
        >>> from rfdetr_plus import RFDETRAtto
        >>> RFDETRAtto().export_cpu_onnx("output", output_name="atto-cpu-v1")  # doctest: +SKIP
        PosixPath('output/atto-cpu-v1.onnx')
    """
    size_limit_bytes = _size_limit(size_limit_bytes)
    recipe = _recipe_for(detector)
    stem = _output_stem(detector.size, output_name)

    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError('Compact CPU export requires pip install "rfdetr-plus[onnx]"') from exc

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{stem}.onnx"
    with TemporaryDirectory(dir=directory) as workspace:
        from rfdetr_plus.models.pe_core import _CPU_EXPORT

        token = _CPU_EXPORT.set(True)
        try:
            # The TorchScript exporter supports opset 20. Reserve the newer ops in a private
            # domain during tracing, then upgrade the complete model using the ONNX converter.
            fp32 = detector.export(output_dir=workspace, format="onnx", opset_version=20, verbose=False)
        finally:
            _CPU_EXPORT.reset(token)
        traced = onnx.load(fp32)
        _check_custom_ops(traced)
        graph = _upgrade_attention(traced)
        if sum(node.op_type == "Attention" for node in graph.graph.node) != 12:
            raise ValueError("Expected one fused attention operator in each PE block")
        selected = _select_projections(graph.graph, detector.size, recipe)
        if recipe.projector_matmuls:
            _projector_matmuls(graph.graph)
        temporary = Path(workspace) / "compact.onnx"
        quantize_dynamic(
            graph,
            temporary,
            nodes_to_quantize=selected,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=True,
            weight_type=QuantType.QInt8,
        )
        compact = onnx.load(temporary)
        if sum(node.op_type == "MatMulInteger" for node in compact.graph.node) != recipe.projections:
            raise ValueError(f"ONNX Runtime did not convert all {recipe.projections} projections for {detector.size}")
        _share_constants(compact)
        _check_compacted(_compact_weights(compact, pack_int8=recipe.pack_int8), detector.size, recipe)
        onnx.checker.check_model(compact)
        onnx.save(compact, temporary)
        byte_count = temporary.stat().st_size
        if size_limit_bytes is not None and byte_count >= size_limit_bytes:
            raise ValueError(
                f"{detector.size} uses {byte_count} bytes and must be smaller than {size_limit_bytes} bytes"
            )
        temporary.replace(target)
    return target
