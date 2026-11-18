# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests of the graph rewrites behind the compact CPU export, on small synthetic ONNX models.

Each rewrite is judged by its effect on the graph (initializer types, node types) and on what ONNX Runtime computes
from it, never by a return value.
"""

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from rfdetr_plus.models.pe_cpu_export import _compact_weights, _output_stem, _projector_matmuls, _share_constants
from tests._markers import _requires_onnx

_SIDE = 64  # a 64 x 64 weight has exactly the 4096 elements from which weights are stored compactly
_MATMUL_NAME = "/encoder/blocks.0/attn/qkv/MatMul"
_CONV_NAME = "/projector/stages.0/Conv"


def _model(nodes: list, initializers: list, inputs: list, outputs: list) -> ModelProto:
    graph = helper.make_graph(nodes, "synthetic", inputs, outputs, initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def _info(name: str, shape: list[int]):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _run(model: ModelProto, feeds: dict[str, np.ndarray]) -> np.ndarray:
    import onnxruntime

    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)[0]


def _matmul_model(weight: np.ndarray, node_name: str = _MATMUL_NAME) -> ModelProto:
    """``y = x @ weight`` for a ``[_SIDE, _SIDE]`` weight; feeding the identity as x reads the weight back."""
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name=node_name)
    return _model(
        [node], [numpy_helper.from_array(weight, "w")], [_info("x", [_SIDE, _SIDE])], [_info("y", [_SIDE, _SIDE])]
    )


def _conv_weight_model(weight: np.ndarray) -> ModelProto:
    """Pointwise Conv with weight ``[_SIDE, _SIDE, 1, 1]``; feeding the identity as a batch reads ``weight.T`` back."""
    node = helper.make_node("Conv", ["x", "w"], ["y"], name="/decoder/layers.0/Conv")
    return _model(
        [node],
        [numpy_helper.from_array(weight, "w")],
        [_info("x", [_SIDE, _SIDE, 1, 1])],
        [_info("y", [_SIDE, _SIDE, 1, 1])],
    )


def _weights(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _initializer_types(model: ModelProto) -> set[int]:
    return {tensor.data_type for tensor in model.graph.initializer}


_IDENTITY = np.eye(_SIDE, dtype=np.float32)


@_requires_onnx
class TestCompactWeightsInt8:
    def test_matmul_weight_is_stored_as_int8(self) -> None:
        model = _matmul_model(_weights((_SIDE, _SIDE)))

        _compact_weights(model, pack_int8=True)

        assert TensorProto.INT8 in _initializer_types(model)
        assert TensorProto.FLOAT not in {t.data_type for t in model.graph.initializer if t.name == "w"}

    def test_matmul_round_trip_error_is_at_most_half_a_scale_per_output_column(self) -> None:
        # Output columns differ in magnitude by 1000x, so a scale taken along the wrong axis breaks the bound.
        weight = _weights((_SIDE, _SIDE)) * np.geomspace(0.01, 10, _SIDE, dtype=np.float32)
        model = _matmul_model(weight)

        _compact_weights(model, pack_int8=True)

        error = np.abs(_run(model, {"x": _IDENTITY}) - weight).max(axis=0)
        half_scale = np.abs(weight).max(axis=0) / 127 / 2
        assert (error <= half_scale * (1 + 1e-5)).all()

    def test_conv_round_trip_error_is_at_most_half_a_scale_per_output_channel(self) -> None:
        # Output channels differ in magnitude by 1000x, so a scale taken along the wrong axis breaks the bound.
        weight = _weights((_SIDE, _SIDE, 1, 1)) * np.geomspace(0.01, 10, _SIDE, dtype=np.float32).reshape(-1, 1, 1, 1)
        model = _conv_weight_model(weight)

        _compact_weights(model, pack_int8=True)

        reconstructed = _run(model, {"x": np.eye(_SIDE, dtype=np.float32).reshape(_SIDE, _SIDE, 1, 1)})
        # Row b of the output is the weight column fed by basis vector b, so read the weight back transposed.
        error = np.abs(reconstructed[:, :, 0, 0].T - weight[:, :, 0, 0]).max(axis=1)
        half_scale = np.abs(weight[:, :, 0, 0]).max(axis=1) / 127 / 2
        assert (error <= half_scale * (1 + 1e-5)).all()

    def test_all_zero_channel_reconstructs_as_zeros(self) -> None:
        weight = _weights((_SIDE, _SIDE))
        weight[:, 3] = 0
        model = _matmul_model(weight)

        _compact_weights(model, pack_int8=True)

        reconstructed = _run(model, {"x": _IDENTITY})
        assert np.isfinite(reconstructed).all()
        assert (reconstructed[:, 3] == 0).all()


@_requires_onnx
class TestCompactWeightsFp16:
    def test_matmul_weight_is_stored_as_float16(self) -> None:
        model = _matmul_model(_weights((_SIDE, _SIDE)))

        _compact_weights(model, pack_int8=False)

        assert _initializer_types(model) == {TensorProto.FLOAT16}

    def test_round_trip_equals_the_float16_rounding_of_the_weight(self) -> None:
        weight = _weights((_SIDE, _SIDE))
        model = _matmul_model(weight)

        _compact_weights(model, pack_int8=False)

        np.testing.assert_array_equal(_run(model, {"x": _IDENTITY}), weight.astype(np.float16).astype(np.float32))


@_requires_onnx
class TestCompactWeightsSelection:
    @pytest.mark.parametrize("pack_int8", [True, False])
    def test_weight_below_the_size_threshold_keeps_its_float32_values(self, pack_int8: bool) -> None:
        weight = _weights((_SIDE - 1, _SIDE))
        node = helper.make_node("MatMul", ["x", "w"], ["y"], name=_MATMUL_NAME)
        model = _model(
            [node],
            [numpy_helper.from_array(weight, "w")],
            [_info("x", [1, _SIDE - 1])],
            [_info("y", [1, _SIDE])],
        )

        _compact_weights(model, pack_int8=pack_int8)

        stored = {t.name: t for t in model.graph.initializer}["w"]
        assert stored.data_type == TensorProto.FLOAT
        np.testing.assert_array_equal(numpy_helper.to_array(stored), weight)

    @pytest.mark.parametrize("pack_int8", [True, False])
    def test_weight_outside_the_encoder_projector_and_decoder_scopes_keeps_its_float32_values(
        self, pack_int8: bool
    ) -> None:
        model = _matmul_model(_weights((_SIDE, _SIDE)), node_name="/head/MatMul")

        _compact_weights(model, pack_int8=pack_int8)

        assert _initializer_types(model) == {TensorProto.FLOAT}


@_requires_onnx
class TestProjectorMatmuls:
    def _projector_model(self, bias: bool) -> ModelProto:
        weight = _weights((8, 6, 1, 1), seed=1)
        names = ["x", "w", "b"] if bias else ["x", "w"]
        initializers = [numpy_helper.from_array(weight, "w")]
        if bias:
            initializers.append(numpy_helper.from_array(_weights((8,), seed=2), "b"))
        node = helper.make_node("Conv", names, ["y"], name=_CONV_NAME)
        return _model([node], initializers, [_info("x", [2, 6, 5, 7])], [_info("y", [2, 8, 5, 7])])

    @pytest.mark.parametrize("bias", [True, False])
    def test_pointwise_matmul_equals_the_conv_it_replaces(self, bias: bool) -> None:
        model = self._projector_model(bias)
        features = _weights((2, 6, 5, 7), seed=3)
        expected = _run(model, {"x": features})

        _projector_matmuls(model.graph)

        np.testing.assert_allclose(_run(model, {"x": features}), expected, rtol=0, atol=1e-5)

    def test_projector_conv_becomes_a_matmul(self) -> None:
        model = self._projector_model(bias=True)

        _projector_matmuls(model.graph)

        ops = {node.op_type for node in model.graph.node}
        assert "Conv" not in ops
        assert "MatMul" in ops

    def test_replaced_conv_weight_is_dropped_from_the_initializers(self) -> None:
        model = self._projector_model(bias=False)

        _projector_matmuls(model.graph)

        assert "w" not in {tensor.name for tensor in model.graph.initializer}


@_requires_onnx
class TestShareConstants:
    @staticmethod
    def _constant(name: str, values: np.ndarray):
        return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(values, name))

    def _add_model(self, first: np.ndarray, second: np.ndarray, output_second: bool = False) -> ModelProto:
        nodes = [
            self._constant("c1", first),
            self._constant("c2", second),
            helper.make_node("Add", ["x", "c1"], ["y1"]),
            helper.make_node("Add", ["y1", "c2"], ["y2"]),
        ]
        outputs = [_info("y2", [3])] + ([_info("c2", [3])] if output_second else [])
        return _model(nodes, [], [_info("x", [3])], outputs)

    @staticmethod
    def _constants(model: ModelProto) -> list:
        return [node for node in model.graph.node if node.op_type == "Constant"]

    def test_identical_constants_collapse_into_one(self) -> None:
        values = np.array([1, 2, 3], dtype=np.float32)
        model = self._add_model(values, values.copy())
        expected = _run(model, {"x": values})

        _share_constants(model)

        assert len(self._constants(model)) == 1
        np.testing.assert_array_equal(_run(model, {"x": values}), expected)

    def test_different_constants_stay_separate(self) -> None:
        model = self._add_model(np.array([1, 2, 3], dtype=np.float32), np.array([1, 2, 4], dtype=np.float32))

        _share_constants(model)

        assert len(self._constants(model)) == 2

    def test_constant_that_is_a_graph_output_is_kept(self) -> None:
        values = np.array([1, 2, 3], dtype=np.float32)
        model = self._add_model(values, values.copy(), output_second=True)

        _share_constants(model)

        assert "c2" in {node.output[0] for node in self._constants(model)}


@_requires_onnx
class TestFailLoudGuards:
    """Weights or graphs the compact storage cannot represent faithfully raise instead of exporting silently wrong."""

    @staticmethod
    def _with_value(position: tuple[int, int], value: float) -> ModelProto:
        weight = _weights((_SIDE, _SIDE))
        weight[position] = value
        return _matmul_model(weight)

    @staticmethod
    def _with_name_in_use(suffix: str) -> ModelProto:
        model = _matmul_model(_weights((_SIDE, _SIDE)))
        model.graph.initializer.append(numpy_helper.from_array(np.zeros(1, dtype=np.float32), "w" + suffix))
        return model

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_int8_packing_rejects_a_nonfinite_weight(self, value: float) -> None:
        model = self._with_value((0, 0), value)

        with pytest.raises(ValueError, match="Nonfinite weight in w"):
            _compact_weights(model, pack_int8=True)

    @pytest.mark.parametrize("suffix", ["_cpu_i8", "_cpu_scale", "_cpu_expand"])
    def test_int8_packing_rejects_a_taken_storage_name(self, suffix: str) -> None:
        model = self._with_name_in_use(suffix)

        with pytest.raises(ValueError, match="Cannot allocate storage names for w"):
            _compact_weights(model, pack_int8=True)

    @pytest.mark.parametrize("value", [np.nan, np.inf, 1e6])
    def test_fp16_packing_rejects_a_weight_it_cannot_hold(self, value: float) -> None:
        model = self._with_value((0, 0), value)

        with pytest.raises(ValueError, match="Cannot store w in FP16"):
            _compact_weights(model, pack_int8=False)

    def test_fp16_packing_rejects_a_taken_storage_name(self) -> None:
        model = self._with_name_in_use("_cpu_fp16")

        with pytest.raises(ValueError, match="Tensor name already in use: w_cpu_fp16"):
            _compact_weights(model, pack_int8=False)

    def test_projector_rewrite_keeps_a_spatial_convolution(self) -> None:
        weight = numpy_helper.from_array(_weights((4, 4, 3, 3)), "w")
        node = helper.make_node("Conv", ["x", "w"], ["y"], name=_CONV_NAME, pads=[1, 1, 1, 1])
        model = _model([node], [weight], [_info("x", [1, 4, 5, 5])], [_info("y", [1, 4, 5, 5])])

        _projector_matmuls(model.graph)

        assert [n.op_type for n in model.graph.node] == ["Conv"]

    @pytest.mark.parametrize("attributes", [{"group": 2}, {"strides": [2, 2]}, {"auto_pad": "SAME_UPPER"}])
    def test_projector_rewrite_rejects_a_pointwise_conv_it_cannot_reduce_to_a_matmul(self, attributes: dict) -> None:
        weight = numpy_helper.from_array(_weights((4, 2, 1, 1)), "w")
        node = helper.make_node("Conv", ["x", "w"], ["y"], name=_CONV_NAME, **attributes)
        model = _model([node], [weight], [_info("x", [1, 4, 4, 4])], [_info("y", [1, 4, 4, 4])])

        with pytest.raises(ValueError, match="Unsupported spatial projection"):
            _projector_matmuls(model.graph)


class TestOutputStem:
    """The ``output_name`` argument is a full stem: directory parts are dropped and the export extension stripped."""

    @pytest.mark.parametrize("output_name", [None, ""])
    def test_default_stem_names_the_size(self, output_name: str | None) -> None:
        assert _output_stem("atto", output_name) == "atto-cpu"

    @pytest.mark.parametrize(
        ("output_name", "expected"),
        [
            ("detector", "detector"),
            ("detector.onnx", "detector"),
            ("DETECTOR.ONNX", "DETECTOR"),
            ("detector-v1.2", "detector-v1.2"),
            ("models/detector.onnx", "detector"),
            ("models\\detector", "detector"),
        ],
    )
    def test_stem_keeps_the_basename_without_the_export_extension(self, output_name: str, expected: str) -> None:
        assert _output_stem("femto", output_name) == expected

    @pytest.mark.parametrize("output_name", ["models/", "models\\"])
    def test_name_without_a_filename_is_rejected(self, output_name: str) -> None:
        with pytest.raises(ValueError, match="non-empty filename stem"):
            _output_stem("pico", output_name)
