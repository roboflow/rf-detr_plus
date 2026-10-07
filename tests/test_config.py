# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compatibility tests between rfdetr_plus's configs and base rfdetr's ModelConfig/TrainConfig.

Verifies that every field base ``rfdetr`` currently exposes on ``ModelConfig`` (including
``compile``) reaches every Plus config (XLarge, 2XLarge, Atto, Femto, Pico) through inheritance, that
shared field defaults have not silently drifted from base, that ``TrainConfig`` is used
unmodified (rfdetr_plus does not subclass it), and that constructing a Plus model with
``pretrain_weights=None`` and round-tripping it through ``RFDETR.from_checkpoint()`` both succeed
and resolve back to the same Plus model class. A ``cuda_graphs`` construction check is also present,
self-activating once upstream ``rfdetr`` releases that field (absent as of this writing).
"""

import argparse
import os
import tempfile
from pathlib import Path

import pytest
import torch
from rfdetr import from_checkpoint
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.detr import RFDETR

from rfdetr_plus.models.detection import (
    RFDETR2XLarge,
    RFDETR2XLargeConfig,
    RFDETRAtto,
    RFDETRAttoConfig,
    RFDETRFemto,
    RFDETRFemtoConfig,
    RFDETRPico,
    RFDETRPicoConfig,
    RFDETRXLarge,
    RFDETRXLargeConfig,
)
from rfdetr_plus.models.pe_core import PE_CORE_T_ENCODER

# Fields every Plus config intentionally overrides with a variant-specific value (architecture
# knobs, resolution, pretrain weights, and license). Any base ModelConfig field NOT in this set
# must keep the same default on the Plus config, or the override was accidental drift.
_OVERRIDE_FIELDS = frozenset(
    {
        "encoder",
        "hidden_dim",
        "dec_layers",
        "sa_nheads",
        "ca_nheads",
        "dec_n_points",
        "num_windows",
        "patch_size",
        "projector_scale",
        "out_feature_indexes",
        "num_classes",
        "positional_encoding_size",
        "resolution",
        "pretrain_weights",
        "license",
    }
)

# The PE-Core-T models additionally set the decoder FFN width and the query counts of their NAS-selected architectures.
_PE_OVERRIDE_FIELDS = _OVERRIDE_FIELDS | {"dim_feedforward", "num_queries", "num_select"}
_OVERRIDE_FIELDS_BY_CONFIG: dict[type[ModelConfig], frozenset[str]] = {
    RFDETRAttoConfig: _PE_OVERRIDE_FIELDS,
    RFDETRFemtoConfig: _PE_OVERRIDE_FIELDS,
    RFDETRPicoConfig: _PE_OVERRIDE_FIELDS,
}

_PLUS_CONFIG_CLASSES = [
    pytest.param(RFDETRXLargeConfig, id="xlarge"),
    pytest.param(RFDETR2XLargeConfig, id="2xlarge"),
    pytest.param(RFDETRAttoConfig, id="atto"),
    pytest.param(RFDETRFemtoConfig, id="femto"),
    pytest.param(RFDETRPicoConfig, id="pico"),
]
_PLUS_MODEL_CLASSES = [
    pytest.param(RFDETRXLarge, id="xlarge"),
    pytest.param(RFDETR2XLarge, id="2xlarge"),
    pytest.param(RFDETRAtto, id="atto"),
    pytest.param(RFDETRFemto, id="femto"),
    pytest.param(RFDETRPico, id="pico"),
]


class TestPlusConfigParity:
    """Every base ModelConfig field must be present and non-drifted on the Plus configs."""

    @pytest.mark.parametrize("config_cls", _PLUS_CONFIG_CLASSES)
    def test_has_every_base_model_config_field(self, config_cls: type[ModelConfig]) -> None:
        """The Plus config must not drop any field base ModelConfig defines.

        rfdetr_plus subclasses ModelConfig instead of maintaining its own field whitelist, so a
        new base field (e.g. ``cuda_graphs``) should always be present here too. A field missing
        here would mean the subclass shadowed the base schema instead of extending it.
        """
        missing = set(ModelConfig.model_fields) - set(config_cls.model_fields)
        assert not missing, f"{config_cls.__name__} is missing base ModelConfig fields: {missing}"

    @pytest.mark.parametrize("config_cls", _PLUS_CONFIG_CLASSES)
    def test_shared_field_defaults_match_base(self, config_cls: type[ModelConfig]) -> None:
        """Fields the Plus config does not intentionally override must keep base's default.

        Catches silent default drift: if upstream changes a shared field's default (as happened
        with several fields across recent rfdetr releases) and the Plus config was never updated
        to match, this fails instead of silently shipping a stale default.
        """
        shared_fields = set(ModelConfig.model_fields) - _OVERRIDE_FIELDS_BY_CONFIG.get(config_cls, _OVERRIDE_FIELDS)
        mismatched = {}
        for name in sorted(shared_fields):
            base_default = ModelConfig.model_fields[name].get_default(call_default_factory=True)
            plus_default = config_cls.model_fields[name].get_default(call_default_factory=True)
            if base_default != plus_default:
                mismatched[name] = (base_default, plus_default)
        assert not mismatched, f"{config_cls.__name__} defaults drifted from ModelConfig: {mismatched}"

    @pytest.mark.parametrize("config_cls", _PLUS_CONFIG_CLASSES)
    def test_override_fields_are_still_declared_by_base(self, config_cls: type[ModelConfig]) -> None:
        """Every field this test suite treats as an intentional override must still exist upstream.

        Guards the guard: if upstream ever renames or removes one of the fields Plus overrides,
        this fails loudly instead of the override silently becoming dead configuration.
        """
        overrides = _OVERRIDE_FIELDS_BY_CONFIG.get(config_cls, _OVERRIDE_FIELDS)
        missing_upstream = overrides - set(ModelConfig.model_fields)
        assert not missing_upstream, f"Override fields no longer exist on base ModelConfig: {missing_upstream}"


@pytest.mark.parametrize("model_cls", _PLUS_MODEL_CLASSES)
def test_train_config_class_is_unmodified_base(model_cls: type[RFDETR]) -> None:
    """Every Plus model must use base TrainConfig unmodified, not a shadowed subclass.

    Unlike ModelConfig, rfdetr_plus does not subclass TrainConfig at all (all Plus models are plain
    detection variants, not segmentation/keypoint ones), so every TrainConfig field is supported by
    construction. This test pins that assumption so a future accidental override is caught.
    """
    assert model_cls._train_config_class is TrainConfig


@pytest.mark.parametrize("config_cls", _PLUS_CONFIG_CLASSES)
def test_compile_field_reaches_plus_config(config_cls: type[ModelConfig]) -> None:
    """The ``compile`` flag (torch.compile) must construct cleanly and round-trip through Plus configs."""
    cfg = config_cls(compile=True)
    assert cfg.compile is True


@pytest.mark.skipif(
    "cuda_graphs" not in ModelConfig.model_fields,
    reason="rfdetr cuda_graphs field is unreleased upstream (develop #1479/#1481); "
    "this test self-activates once a release exposes it",
)
@pytest.mark.parametrize("config_cls", _PLUS_CONFIG_CLASSES)
def test_cuda_graphs_field_reaches_plus_config(config_cls: type[ModelConfig]) -> None:
    """The ``cuda_graphs`` flag must construct cleanly and round-trip through Plus configs."""
    cfg = config_cls(cuda_graphs=True)
    assert cfg.cuda_graphs is True


@pytest.mark.parametrize("model_cls", _PLUS_MODEL_CLASSES)
def test_from_checkpoint_resolves_plus_model_class(model_cls: type[RFDETR]) -> None:
    """rfdetr.from_checkpoint() must resolve back to the same Plus model class it was saved from.

    Model class resolution goes through RFDETR's _name_map/_model_map (matched on the checkpoint's
    recorded pretrain_weights filename), not through _model_config_class — that attribute only
    filters which saved config keys from_checkpoint() forwards to the constructor. This test's real
    regression guard is architecture-only construction (pretrain_weights=None): pre-1.1.0 the Plus
    configs narrowed that field's annotation to plain str (still with a default value, so not
    required to pass), so an explicit pretrain_weights=None in model_cls(...) below failed pydantic
    validation before from_checkpoint() was ever reached.

    Builds an architecture-only instance (pretrain_weights=None skips the RF-DETR checkpoint;
    the DINOv2 or PE-Core-T backbone setup may still load its own weights), saves a minimal
    training-style checkpoint from its state_dict, and reloads it through from_checkpoint().
    """
    model_instance = model_cls(pretrain_weights=None, accept_platform_model_license=True)
    num_classes = model_instance.model.args.num_classes
    checkpoint = {
        "args": argparse.Namespace(pretrain_weights=f"{model_cls.size}.pth", num_classes=num_classes),
        "model": model_instance.model.model.state_dict(),
    }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pth")
    os.close(tmp_fd)
    try:
        torch.save(checkpoint, tmp_path)
        recovered = from_checkpoint(tmp_path, accept_platform_model_license=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    assert isinstance(recovered, model_cls), (
        f"from_checkpoint returned {type(recovered).__name__}, expected {model_cls.__name__}"
    )


class TestPECoreTConfigs:
    """Architecture of the PE-Core-T models, as selected from the PE-Core-T NAS supernet."""

    @pytest.mark.parametrize(
        ("config_cls", "expected"),
        [
            pytest.param(
                RFDETRAttoConfig,
                dict(resolution=380, patch_size=20, num_windows=1, dec_layers=0, num_queries=300, num_select=300),
                id="atto",
            ),
            pytest.param(
                RFDETRFemtoConfig,
                dict(resolution=384, patch_size=16, num_windows=2, dec_layers=2, num_queries=200, num_select=200),
                id="femto",
            ),
            pytest.param(
                RFDETRPicoConfig,
                dict(resolution=560, patch_size=20, num_windows=2, dec_layers=3, num_queries=200, num_select=200),
                id="pico",
            ),
        ],
    )
    def test_size_specific_fields(self, config_cls: type[ModelConfig], expected: dict) -> None:
        config = config_cls()

        assert {key: getattr(config, key) for key in expected} == expected
        assert config.positional_encoding_size == config.resolution // config.patch_size
        assert config.resolution % (config.patch_size * config.num_windows) == 0

    @pytest.mark.parametrize("config_cls", [RFDETRAttoConfig, RFDETRFemtoConfig, RFDETRPicoConfig])
    def test_shared_pe_core_t_architecture(self, config_cls: type[ModelConfig]) -> None:
        config = config_cls()

        assert config.encoder == PE_CORE_T_ENCODER
        assert (config.hidden_dim, config.sa_nheads, config.ca_nheads, config.dec_n_points) == (128, 4, 8, 2)
        assert config.dim_feedforward == 1024
        assert config.out_feature_indexes == [2, 5, 8, 11]
        assert config.projector_scale == ["P4"]
        assert config.license == "PML-1.0"

    @pytest.mark.parametrize("config_cls", [RFDETRAttoConfig, RFDETRFemtoConfig, RFDETRPicoConfig])
    def test_non_rgb_input_is_rejected(self, config_cls: type[ModelConfig]) -> None:
        with pytest.raises(ValueError, match="num_channels=3 only"):
            config_cls(num_channels=1)

    def test_custom_resolution_resizes_the_position_grid(self) -> None:
        config = RFDETRAttoConfig(resolution=400)

        assert config.positional_encoding_size == 20
