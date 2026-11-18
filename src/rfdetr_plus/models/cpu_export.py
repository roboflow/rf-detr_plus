# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CPU ONNX export for the fixed PE-Core-T detector family."""

from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx import GraphProto, ModelProto
    from rfdetr.detr import RFDETR


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


def _compact_weights(model: ModelProto, *, pack_int8: bool) -> None:
    """Store large floating-point filters compactly and reconstruct their FP32 values at session setup."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    weights = {tensor.name: tensor for tensor in graph.initializer}
    candidates = {
        node.input[1]
        for node in graph.node
        if node.op_type in {"Conv", "MatMul"}
        and len(node.input) > 1
        and node.input[1] in weights
        and weights[node.input[1]].data_type == TensorProto.FLOAT
        and any(scope in node.name for scope in ("/encoder/", "/projector/", "/decoder/"))
    }
    casts = []
    used = set(weights) | {name for node in graph.node for name in (*node.input, *node.output)}
    for name in sorted(candidates):
        tensor = weights[name]
        values = numpy_helper.to_array(tensor)
        if values.size < 4096:
            continue
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


def export_cpu_onnx(
    detector: RFDETR, output_dir: str | Path = "output", *, size_limit_bytes: int | None = None
) -> Path:
    """Export the loaded PE detector with dynamic encoder INT8 and compact weight storage.

    The file uses batch one and the configured model resolution. Quantization scales for weights
    derive from the live checkpoint. ONNX Runtime computes activation scales at inference time.
    When provided, the complete uncompressed file must be strictly smaller than size_limit_bytes.
    """
    if size_limit_bytes is not None and (type(size_limit_bytes) is not int or size_limit_bytes <= 0):
        raise ValueError("size_limit_bytes must be a positive integer")

    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError('Compact CPU export requires pip install "rfdetr-plus[cpu-export]"') from exc

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{detector.size}-cpu.onnx"
    with TemporaryDirectory(dir=directory) as workspace:
        from rfdetr_plus.models.pe_core import _CPU_EXPORT

        token = _CPU_EXPORT.set(True)
        try:
            # The TorchScript exporter supports opset 20. Reserve the newer ops in a private
            # domain during tracing, then upgrade the complete model using the ONNX converter.
            fp32 = detector.export(output_dir=workspace, format="onnx", opset_version=20, verbose=False)
        finally:
            _CPU_EXPORT.reset(token)
        graph = _upgrade_attention(onnx.load(fp32))
        if sum(node.op_type == "Attention" for node in graph.graph.node) != 12:
            raise ValueError("Expected one fused attention operator in each PE block")
        selected = [
            node.name
            for node in graph.graph.node
            if node.op_type == "MatMul"
            and re.search(r"/encoder/blocks\.\d+/(attn/(qkv|proj)|mlp/fc1)/MatMul$", node.name)
        ]
        if len(selected) != 36:
            raise ValueError(f"Expected three encoder projections in each of twelve PE blocks, found {len(selected)}")
        if detector.size == "rfdetr-atto":
            selected.extend(
                node.name
                for node in graph.graph.node
                if re.search(r"/encoder/blocks\.(?:[6-9]|10|11)/mlp/fc2/MatMul$", node.name)
            )
        else:
            decoder_projection = "linear2" if detector.size == "rfdetr-femto" else "linear1"
            selected.extend(
                node.name
                for node in graph.graph.node
                if node.name.endswith(f"/decoder/layers.1/{decoder_projection}/MatMul")
            )
        expected = 42 if detector.size == "rfdetr-atto" else 37
        if len(selected) != expected:
            raise ValueError(f"Expected {expected} linear projections for {detector.size}, found {len(selected)}")
        if detector.size in {"rfdetr-atto", "rfdetr-pico"}:
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
        if sum(node.op_type == "MatMulInteger" for node in compact.graph.node) != expected:
            raise ValueError(f"ONNX Runtime did not convert all {expected} projections for {detector.size}")
        _share_constants(compact)
        _compact_weights(compact, pack_int8=detector.size == "rfdetr-atto")
        onnx.checker.check_model(compact)
        onnx.save(compact, temporary)
        byte_count = temporary.stat().st_size
        if size_limit_bytes is not None and byte_count >= size_limit_bytes:
            raise ValueError(
                f"{detector.size} uses {byte_count} bytes and must be smaller than {size_limit_bytes} bytes"
            )
        temporary.replace(target)
    return target
