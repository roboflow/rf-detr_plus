# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for the PE-Core-T encoder and backbone behind RFDETRAtto / RFDETRFemto / RFDETRPico.

All encoders here are built from timm's architecture without pretrained weights, so the tests run offline.
"""

import copy
from argparse import Namespace

import pytest
import torch
from rfdetr.models.backbone import build_backbone
from timm.layers import resample_abs_pos_embed

from rfdetr_plus.models.pe_core import (
    PE_CORE_T_ENCODER,
    PECoreBackbone,
    PECoreEncoder,
    apply_windowing_with_rope,
    get_pe_lr_decay_rate,
    get_pe_weight_decay_rate,
    undo_windowing,
)

_OUT_FEATURE_INDEXES = [2, 5, 8, 11]


def _encoder(patch_size: int = 20, num_windows: int = 2, grid: int = 8, **kwargs) -> PECoreEncoder:
    side = grid * patch_size
    return PECoreEncoder(
        out_feature_indexes=_OUT_FEATURE_INDEXES,
        shape=(side, side),
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=grid,
        load_pretrained_weights=False,
        **kwargs,
    )


class TestWindowing:
    """Window partitioning of tokens and their RoPE rows."""

    def test_undo_inverts_apply(self) -> None:
        tokens = torch.randn(2, 8 * 8, 5)
        rope = torch.randn(8 * 8, 4)

        windowed, _ = apply_windowing_with_rope(tokens, rope, num_windows=2, patch_size=10, height=80, width=80)

        assert windowed.shape == (2 * 4, 16, 5)
        assert torch.equal(undo_windowing(windowed, num_windows=2, patch_size=10, height=80, width=80), tokens)

    def test_rope_rows_follow_their_tokens(self) -> None:
        """Every windowed token keeps the RoPE row of its absolute grid position."""
        grid, num_windows = 6, 3
        token_ids = torch.arange(grid * grid, dtype=torch.float32)
        tokens = token_ids.view(1, -1, 1).repeat(2, 1, 1)
        rope = token_ids.view(-1, 1).repeat(1, 4)

        windowed, windowed_rope = apply_windowing_with_rope(
            tokens, rope, num_windows=num_windows, patch_size=1, height=grid, width=grid
        )

        assert windowed_rope.shape == (num_windows**2, (grid // num_windows) ** 2, 4)
        for batch in range(2):
            window_block = windowed[batch * num_windows**2 : (batch + 1) * num_windows**2, :, 0]
            assert torch.equal(window_block, windowed_rope[..., 0])


class TestPECoreEncoder:
    """Construction, runtime resolution changes, checkpoint loading and export of the encoder."""

    @pytest.mark.parametrize(
        ("patch_size", "num_windows"),
        [
            pytest.param(20, 1, id="atto-like"),
            pytest.param(16, 2, id="femto-like"),
            pytest.param(20, 2, id="pico-like"),
        ],
    )
    def test_returns_one_feature_map_per_out_index(self, patch_size: int, num_windows: int) -> None:
        encoder = _encoder(patch_size=patch_size, num_windows=num_windows).eval()
        side = 8 * patch_size

        with torch.no_grad():
            feats = encoder(torch.randn(2, 3, side, side))

        assert encoder._out_feature_channels == [192] * 4
        assert [tuple(f.shape) for f in feats] == [(2, 192, 8, 8)] * 4

    def test_trunk_matches_timm_architecture(self) -> None:
        encoder = _encoder(patch_size=20, grid=19)

        assert encoder.model.pretrained_cfg["architecture"] == "vit_pe_core_tiny_patch16_384"
        assert tuple(encoder.model.patch_embed.proj.weight.shape) == (192, 3, 20, 20)
        assert tuple(encoder.model.pos_embed.shape) == (1, 1 + 19 * 19, 192)
        assert all(block.attn.num_prefix_tokens == 0 for block in encoder.model.blocks)

    def test_unused_clip_head_is_dropped(self) -> None:
        encoder = _encoder()

        assert encoder.model.attn_pool is None
        assert isinstance(encoder.model.head, torch.nn.Identity)
        assert not [name for name in encoder.state_dict() if "attn_pool" in name or name.startswith("model.head.")]

    def test_positional_encoding_size_sets_the_position_grid(self) -> None:
        encoder = PECoreEncoder(
            out_feature_indexes=_OUT_FEATURE_INDEXES,
            shape=(320, 320),
            patch_size=20,
            num_windows=1,
            positional_encoding_size=10,
            load_pretrained_weights=False,
        )

        assert tuple(encoder.model.pos_embed.shape) == (1, 1 + 10 * 10, 192)
        assert tuple(encoder.model.patch_embed.grid_size) == (10, 10)

    def test_off_resolution_forward_leaves_state_untouched(self) -> None:
        """A forward at another resolution must not replace or resample any parameter or buffer.

        rf-detr-internal resized the trunk in place (timm ``set_input_size``) for such inputs, which swapped
        ``pos_embed`` for a new Parameter outside the optimizer and resampled it down and back up on every
        multi-scale step.
        """
        encoder = _encoder(patch_size=20, num_windows=2, grid=8).train()
        pos_embed = encoder.model.pos_embed
        pos_before = pos_embed.detach().clone()
        state_before = {k: v.clone() for k, v in encoder.state_dict().items()}
        rope_before = encoder.model.rope.get_embed().clone()

        feats = encoder(torch.randn(1, 3, 240, 240))
        sum(f.sum() for f in feats).backward()

        assert encoder.model.pos_embed is pos_embed
        assert torch.equal(encoder.model.pos_embed.detach(), pos_before)
        assert pos_embed.grad is not None
        assert pos_embed.grad.abs().sum() > 0
        assert tuple(encoder.model.patch_embed.grid_size) == (8, 8)
        assert torch.equal(encoder.model.rope.get_embed(), rope_before)
        assert all(torch.equal(v, state_before[k]) for k, v in encoder.state_dict().items())

    def test_off_resolution_uses_resampled_position_embedding(self) -> None:
        """Another resolution sees the build-grid pos_embed resampled to its grid, exactly as a trunk built there."""
        encoder = _encoder(patch_size=20, num_windows=1, grid=8).eval()
        rebuilt = _encoder(patch_size=20, num_windows=1, grid=12).eval()
        state = encoder.state_dict()
        state["model.pos_embed"] = resample_abs_pos_embed(
            state["model.pos_embed"], new_size=(12, 12), old_size=(8, 8), num_prefix_tokens=1
        )
        rebuilt.load_state_dict(state)
        images = torch.randn(1, 3, 240, 240)

        with torch.no_grad():
            got, expected = encoder(images), rebuilt(images)

        assert all(torch.equal(a, b) for a, b in zip(got, expected))

    @pytest.mark.parametrize("grid", [8, 12, 19, 24, 28])
    def test_rope_for_grid_matches_timm_feature_shape_update(self, grid: int) -> None:
        """``rope_for_grid`` must equal what timm computes after ``update_feat_shape`` (guards the timm API)."""
        encoder = _encoder(grid=8)
        reference = copy.deepcopy(encoder.model.rope)
        reference.update_feat_shape((grid, grid))

        assert torch.equal(encoder.rope_for_grid(grid, grid), reference.get_embed())

    def test_load_state_dict_resamples_position_grid(self) -> None:
        """A checkpoint from another position grid loads with pos_embed resampled to this encoder's grid."""
        source = _encoder(patch_size=20, grid=8)
        with torch.no_grad():
            source.model.pos_embed.normal_()
        target = _encoder(patch_size=20, grid=10)

        target.load_state_dict(source.state_dict())

        expected = resample_abs_pos_embed(
            source.model.pos_embed.detach(), new_size=(10, 10), old_size=(8, 8), num_prefix_tokens=1
        )
        assert torch.equal(target.model.pos_embed.detach(), expected)
        assert torch.equal(target.model.patch_embed.proj.weight, source.model.patch_embed.proj.weight)

    def test_input_not_divisible_by_window_block_raises(self) -> None:
        encoder = _encoder(patch_size=20, num_windows=2, grid=8)

        with pytest.raises(ValueError, match="divisible by 40"):
            encoder(torch.randn(1, 3, 180, 180))

    def test_export_matches_eager_at_build_resolution(self) -> None:
        encoder = _encoder(patch_size=20, num_windows=2, grid=8).eval()
        images = torch.randn(2, 3, 160, 160)
        with torch.no_grad():
            eager = encoder(images)

        encoder.export()
        with torch.no_grad():
            exported = encoder(images)

        assert all(torch.equal(a, b) for a, b in zip(eager, exported))

    def test_export_bakes_a_different_shape(self) -> None:
        """``export()`` bakes pos_embed and RoPE for ``shape`` so tracing never resamples at runtime."""
        encoder = _encoder(patch_size=20, num_windows=2, grid=8).eval()
        images = torch.randn(1, 3, 240, 240)
        with torch.no_grad():
            eager = encoder(images)

        encoder.shape = (240, 240)
        encoder.export()
        with torch.no_grad():
            exported = encoder(images)

        assert tuple(encoder.model.patch_embed.grid_size) == (12, 12)
        assert tuple(encoder.model.pos_embed.shape) == (1, 1 + 12 * 12, 192)
        assert all(torch.equal(a, b) for a, b in zip(eager, exported))

    def test_export_bakes_a_non_square_shape(self) -> None:
        """A non-square export shape must trace without runtime resampling, and match eager at that shape."""
        encoder = _encoder(patch_size=20, num_windows=2, grid=8).eval()
        images = torch.randn(1, 3, 160, 240)
        with torch.no_grad():
            eager = encoder(images)

        encoder.set_export_shape((160, 240))
        with torch.no_grad():
            traced = torch.jit.trace(encoder, images, check_trace=False)
            exported = traced(images)

        assert "upsample_bicubic2d" not in str(traced.inlined_graph)
        assert all(torch.equal(a, b) for a, b in zip(eager, exported))

    def test_second_export_shape_is_rejected(self) -> None:
        encoder = _encoder(patch_size=20, num_windows=2, grid=8)
        encoder.set_export_shape((160, 160))
        encoder.set_export_shape((160, 160))

        with pytest.raises(RuntimeError, match="already exported"):
            encoder.set_export_shape((240, 240))

    def test_rf_detr_internal_clip_head_keys_are_ignored(self) -> None:
        """rf-detr-internal checkpoints still carry PE-CLIP's attention-pool head; loading them strictly succeeds."""
        encoder = _encoder()
        state = encoder.state_dict()
        state["model.attn_pool.latent"] = torch.zeros(1, 1, 192)
        state["model.head.weight"] = torch.zeros(512, 192)

        encoder.load_state_dict(state, strict=True)

    def test_other_patch_size_checkpoint_raises_a_clear_error(self) -> None:
        state = _encoder(patch_size=20, grid=8).state_dict()

        with pytest.raises(ValueError, match=r"patch_size=20.*patch_size=16"):
            _encoder(patch_size=16, grid=8).load_state_dict(state)

    def test_gradient_checkpointing_matches_plain_backward(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import rfdetr_plus.models.pe_core as pe_core

        torch.manual_seed(0)
        plain = _encoder(patch_size=20, num_windows=2, grid=8).train()
        checkpointed = _encoder(patch_size=20, num_windows=2, grid=8, gradient_checkpointing=True).train()
        checkpointed.load_state_dict(plain.state_dict())
        images = torch.randn(1, 3, 160, 160)
        calls = []
        original_checkpoint = pe_core.checkpoint
        monkeypatch.setattr(pe_core, "checkpoint", lambda *a, **k: calls.append(1) or original_checkpoint(*a, **k))

        for encoder in (plain, checkpointed):
            sum(f.square().sum() for f in encoder(images)).backward()

        assert len(calls) == len(checkpointed.model.blocks)  # only the checkpointed encoder, once per block

        for (name, a), (_, b) in zip(plain.named_parameters(), checkpointed.named_parameters()):
            assert (a.grad is None) == (b.grad is None), name
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


class TestPELayerDecay:
    """Layer-wise learning-rate and weight-decay rules for PE-Core-T parameter names."""

    @pytest.mark.parametrize(
        ("name", "layer_id"),
        [
            pytest.param("backbone.0.encoder.model.pos_embed", 0, id="pos_embed"),
            pytest.param("backbone.0.encoder.model.patch_embed.proj.weight", 0, id="patch_embed"),
            pytest.param("backbone.0.encoder.model.cls_token", 0, id="cls_token"),
            pytest.param("backbone.0.encoder.model.blocks.0.attn.qkv.weight", 1, id="block-0"),
            pytest.param("backbone.0.encoder.model.blocks.11.mlp.fc2.bias", 12, id="block-11"),
            pytest.param("backbone.0.encoder.model.norm.weight", 14, id="final-norm"),
            pytest.param("backbone.0.encoder.model.norm_pre.weight", 14, id="pre-norm"),
        ],
    )
    def test_lr_decay_rate(self, name: str, layer_id: int) -> None:
        assert get_pe_lr_decay_rate(name, lr_decay_rate=0.8, num_layers=13) == pytest.approx(0.8 ** (14 - layer_id))

    @pytest.mark.parametrize(
        ("name", "rate"),
        [
            pytest.param("backbone.0.encoder.model.pos_embed", 0.0, id="pos_embed"),
            pytest.param("backbone.0.encoder.model.patch_embed.proj.weight", 0.0, id="patch_embed"),
            pytest.param("backbone.0.encoder.model.cls_token", 0.0, id="cls_token"),
            pytest.param("backbone.0.encoder.model.blocks.3.attn.qkv.bias", 0.0, id="bias"),
            pytest.param("backbone.0.encoder.model.blocks.3.norm1.weight", 0.0, id="norm"),
            pytest.param("backbone.0.encoder.model.blocks.3.attn.qkv.weight", 1.0, id="weight"),
        ],
    )
    def test_weight_decay_rate(self, name: str, rate: float) -> None:
        assert get_pe_weight_decay_rate(name) == rate


class TestPECoreBackbone:
    """Registration with rfdetr and the backbone's optimizer parameter groups."""

    @pytest.fixture
    def backbone(self) -> PECoreBackbone:
        joiner = build_backbone(
            encoder=PE_CORE_T_ENCODER,
            vit_encoder_num_layers=12,
            pretrained_encoder=None,
            window_block_indexes=None,
            drop_path=0.0,
            out_channels=128,
            out_feature_indexes=_OUT_FEATURE_INDEXES,
            projector_scale=["P4"],
            use_cls_token=False,
            hidden_dim=128,
            position_embedding="sine",
            freeze_encoder=False,
            layer_norm=True,
            target_shape=(160, 160),
            rms_norm=False,
            backbone_lora=False,
            force_no_pretrain=False,
            gradient_checkpointing=False,
            load_dinov2_weights=False,
            patch_size=20,
            num_windows=2,
            positional_encoding_size=8,
        )
        return joiner[0]

    def test_registered_for_the_pe_core_t_encoder(self, backbone: PECoreBackbone) -> None:
        assert type(backbone) is PECoreBackbone
        assert isinstance(backbone.encoder, PECoreEncoder)

    def test_param_groups_follow_pe_rules(self, backbone: PECoreBackbone) -> None:
        args = Namespace(
            out_feature_indexes=_OUT_FEATURE_INDEXES,
            lr_encoder=1.5e-4,
            lr_vit_layer_decay=0.8,
            lr_component_decay=0.7,
            weight_decay=1e-4,
        )

        groups = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")

        encoder_params = {f"backbone.0.{n}" for n, _ in backbone.named_parameters() if n.startswith("encoder.")}
        assert set(groups) == encoder_params
        num_layers = _OUT_FEATURE_INDEXES[-1] + 2
        for name, group in groups.items():
            expected_lr = 1.5e-4 * get_pe_lr_decay_rate(name, 0.8, num_layers) * 0.7**2
            assert group["lr"] == pytest.approx(expected_lr, rel=0, abs=0), name
            assert group["weight_decay"] == 1e-4 * get_pe_weight_decay_rate(name), name
        blocks_0 = groups["backbone.0.encoder.model.blocks.0.attn.qkv.weight"]
        assert blocks_0["lr"] == pytest.approx(1.5e-4 * 0.8**13 * 0.49)

    def test_frozen_encoder_params_are_excluded(self) -> None:
        backbone = PECoreBackbone(
            PE_CORE_T_ENCODER,
            out_channels=128,
            out_feature_indexes=_OUT_FEATURE_INDEXES,
            projector_scale=["P4"],
            freeze_encoder=True,
            layer_norm=True,
            target_shape=(160, 160),
            load_dinov2_weights=False,
            patch_size=20,
            num_windows=2,
            positional_encoding_size=8,
        )
        args = Namespace(
            out_feature_indexes=_OUT_FEATURE_INDEXES,
            lr_encoder=1.5e-4,
            lr_vit_layer_decay=0.8,
            lr_component_decay=0.7,
            weight_decay=1e-4,
        )

        assert backbone.get_named_param_lr_pairs(args) == {}
