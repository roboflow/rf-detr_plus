# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PE-Core-T encoder and backbone for the RFDETRAtto / RFDETRFemto / RFDETRPico models.

The encoder is Meta's Perception Encoder PE-Core-T vision trunk (timm ``vit_pe_core_tiny_patch16_384``) run with
windowed attention: tokens are split into ``num_windows x num_windows`` windows, and the blocks listed in
``out_feature_indexes`` attend over the full image. PE uses both a learned absolute position embedding and 2D RoPE; the
class token gets a no-op RoPE so each window's copy of it carries no position.

Importing this module registers :class:`PECoreBackbone` with ``rfdetr`` for ``ModelConfig.encoder == "pe_core_t"``.
"""

from __future__ import annotations

import math
from typing import Any

import timm
import torch
from rfdetr.models.backbone import register_backbone
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.utilities.logger import get_logger
from timm.layers import Format, resample_abs_pos_embed
from timm.models._features import feature_take_indices
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

logger = get_logger()

__all__ = [
    "PE_CORE_T_ENCODER",
    "PECoreBackbone",
    "PECoreEncoder",
    "apply_windowing_with_rope",
    "get_pe_lr_decay_rate",
    "get_pe_weight_decay_rate",
    "undo_windowing",
]

#: ``ModelConfig.encoder`` value selecting the PE-Core-T backbone.
PE_CORE_T_ENCODER = "pe_core_t"
#: timm architecture of the PE-Core-T vision trunk.
PE_CORE_T_TIMM_ARCH = "vit_pe_core_tiny_patch16_384"


def apply_windowing_with_rope(
    x: Tensor, ropes: Tensor, num_windows: int, patch_size: int, height: int, width: int
) -> tuple[Tensor, Tensor]:
    """Split patch tokens and their RoPE rows into ``num_windows x num_windows`` windows.

    Unlike DINOv2's windowing, RoPE cannot be pre-fused into the tokens, so its rows are windowed the same way as the
    tokens to keep each token's absolute position.

    Args:
        x: Patch tokens ``(B, H * W, C)`` in row-major grid order, without the class token.
        ropes: RoPE rows ``(H * W, D)`` in the same order, shared across the batch.
        num_windows: Windows per side.
        patch_size: Patch size, used to derive the token grid from ``height`` and ``width``.
        height: Input image height in pixels.
        width: Input image width in pixels.

    Returns:
        Windowed tokens ``(B * num_windows**2, H * W / num_windows**2, C)`` and windowed RoPE rows
        ``(num_windows**2, H * W / num_windows**2, D)``.
    """
    num_h_patches = height // patch_size
    num_w_patches = width // patch_size
    num_windows_squared = num_windows**2
    batch, tokens, channels = x.shape
    _, rope_dim = ropes.shape
    tokens_per_window = tokens // num_windows_squared
    num_h_patches_per_window = num_h_patches // num_windows
    num_w_patches_per_window = num_w_patches // num_windows

    x = x.view(batch, num_windows, num_h_patches_per_window, num_windows, num_w_patches_per_window, channels)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(batch * num_windows_squared, tokens_per_window, channels)

    ropes = ropes.view(num_windows, num_h_patches_per_window, num_windows, num_w_patches_per_window, rope_dim)
    ropes = ropes.permute(0, 2, 1, 3, 4)
    ropes = ropes.reshape(num_windows_squared, tokens_per_window, rope_dim)
    return x, ropes


def undo_windowing(x: Tensor, num_windows: int, patch_size: int, height: int, width: int) -> Tensor:
    """Merge windowed patch tokens back into row-major grid order.

    Args:
        x: Windowed tokens ``(B * num_windows**2, H * W / num_windows**2, C)``, without class tokens.
        num_windows: Windows per side.
        patch_size: Patch size, used to derive the token grid from ``height`` and ``width``.
        height: Input image height in pixels.
        width: Input image width in pixels.

    Returns:
        Tokens ``(B, H * W, C)``.
    """
    num_h_patches = height // patch_size
    num_w_patches = width // patch_size
    num_windows_squared = num_windows**2
    windowed_batch, tokens_per_window, channels = x.shape
    batch = windowed_batch // num_windows_squared
    num_h_patches_per_window = num_h_patches // num_windows
    num_w_patches_per_window = num_w_patches // num_windows

    x = x.view(batch, num_windows, num_windows, num_h_patches_per_window, num_w_patches_per_window, channels)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(batch, tokens_per_window * num_windows_squared, channels)


@torch.compiler.disable
def _resample_pos_embed(pos_embed: Tensor, grid: tuple[int, int], stored_grid: tuple[int, int]) -> Tensor:
    """Resample a ``(1, 1 + H * W, C)`` position embedding (class token first) from *stored_grid* to *grid*.

    Kept out of ``torch.compile`` graphs, like rfdetr's DINOv2 position-embedding interpolation: every new input size
    would otherwise recompile the encoder. Antialiasing is off on MPS, which has no antialiased bicubic kernel.
    """
    return resample_abs_pos_embed(
        pos_embed,
        new_size=list(grid),
        old_size=list(stored_grid),
        num_prefix_tokens=1,
        antialias=pos_embed.device.type != "mps",
    )


def get_pe_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Layer-wise learning-rate decay multiplier for a PE-Core-T (timm naming) parameter.

    Embeddings and the class token form layer 0, block ``i`` is layer ``i + 1``, and every other parameter (final
    norms, heads) gets the undecayed rate.

    Args:
        name: Full parameter name, e.g. ``"backbone.0.encoder.model.blocks.3.attn.qkv.weight"``.
        lr_decay_rate: Per-layer decay factor.
        num_layers: Number of decayed layers above layer 0.

    Returns:
        ``lr_decay_rate ** (num_layers + 1 - layer_id)``.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name or ".cls_token" in name or ".reg_token" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_pe_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """Weight-decay multiplier for a PE-Core-T parameter: zero for embeddings, tokens, biases and norms.

    Args:
        name: Full parameter name.
        weight_decay_rate: Multiplier for every other parameter.

    Returns:
        ``0.0`` or ``weight_decay_rate``.
    """
    if (
        ("pos_embed" in name)
        or ("rel_pos" in name)
        or ("bias" in name)
        or ("norm" in name)
        or ("cls_token" in name)
        or ("reg_token" in name)
        or ("patch_embed" in name)
    ):
        weight_decay_rate = 0.0
    return weight_decay_rate


class PECoreEncoder(nn.Module):
    """PE-Core-T trunk returning windowed-attention feature maps at ``out_feature_indexes``.

    Inputs at any resolution divisible by ``patch_size * num_windows`` are accepted, square or not. The position
    embedding is resampled to the input's patch grid and RoPE is rebuilt for it inside the forward pass; parameters and
    buffers are never modified, so multi-scale training keeps optimizing the original ``pos_embed``. Stochastic depth
    follows timm's build-time ``drop_path`` schedule; rfdetr's per-step drop-path schedule does not reach timm blocks.

    Args:
        out_feature_indexes: Blocks whose outputs are returned; these blocks use full (unwindowed) attention.
        shape: Native ``(height, width)`` input resolution, baked into the graph by :meth:`export`.
        patch_size: Patch size of the patch embedding.
        num_windows: Windows per side for windowed attention.
        positional_encoding_size: Side length, in patches, of the stored position-embedding grid.
        load_pretrained_weights: Initialize the trunk from timm's pretrained PE-Core-T weights (resampled by timm to
            ``patch_size`` and the position grid).
        drop_path: Stochastic-depth rate of the trunk blocks.
        gradient_checkpointing: Checkpoint block activations during training.
    """

    def __init__(
        self,
        out_feature_indexes: list[int],
        shape: tuple[int, int],
        patch_size: int,
        num_windows: int,
        positional_encoding_size: int,
        load_pretrained_weights: bool = False,
        drop_path: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            PE_CORE_T_TIMM_ARCH,
            pretrained=load_pretrained_weights,
            patch_size=patch_size,
            img_size=positional_encoding_size * patch_size,
            drop_path_rate=drop_path,
        )
        # The CLIP attention-pool head never runs here; drop it so it is neither stored, optimized nor exported.
        self.model.attn_pool = None
        self.model.head = nn.Identity()
        self.model.global_pool = ""
        # Patch-embed to an NHWC grid at any input size; `_pos_embed` then resamples pos_embed to that grid.
        self.model.dynamic_img_size = True
        self.model.patch_embed.strict_img_size = False
        self.model.patch_embed.output_fmt = Format.NHWC
        self.model.patch_embed.flatten = False
        # The class token gets a no-op RoPE row instead of being skipped as a prefix token, so every window's copy of it
        # and the full-attention merge of all copies line up with their RoPE rows.
        for block in self.model.blocks:
            block.attn.num_prefix_tokens = 0
        self.model.set_grad_checkpointing(gradient_checkpointing)

        self.out_feature_indexes = list(out_feature_indexes)
        self.shape = tuple(shape)
        self.patch_size = patch_size
        self.num_windows = num_windows
        self._out_feature_channels = [self.model.embed_dim] * len(self.out_feature_indexes)
        self._export = False

    @torch.compiler.disable
    def rope_for_grid(self, grid_h: int, grid_w: int) -> Tensor:
        """RoPE rows ``(grid_h * grid_w, D)`` for a patch grid, without touching the cached rows.

        Args:
            grid_h: Patch rows.
            grid_w: Patch columns.

        Returns:
            The cached rows for the stored grid; otherwise the rows timm's ``update_feat_shape`` would compute.
        """
        rope = self.model.rope
        if (grid_h, grid_w) == tuple(rope.feat_shape):
            return rope.get_embed()
        return rope._get_pos_embed_values([grid_h, grid_w], device=rope.pos_embed.device, dtype=rope.pos_embed.dtype)

    def set_export_shape(self, shape: tuple[int, int]) -> None:
        """Bake position embeddings for *shape* before export tracing (called by ``rfdetr``'s export preparation).

        Args:
            shape: ``(height, width)`` the graph is exported at.

        Raises:
            RuntimeError: If the encoder was already exported for a different shape.
        """
        shape = tuple(shape)
        if self._export and shape != self.shape:
            raise RuntimeError(
                f"This PE-Core-T encoder is already exported for shape {self.shape}; export a fresh copy of the model "
                f"for shape {shape}."
            )
        self.shape = shape
        self.export()

    def export(self) -> None:
        """Bake the position embedding and RoPE for ``shape`` so the traced forward never resamples.

        Runtime pos_embed resampling uses antialiased bicubic interpolation, which ONNX cannot express. Eager use after
        ``export()`` sees the model in its baked, export-time configuration. Idempotent.
        """
        if self._export:
            return
        grid = (self.shape[0] // self.patch_size, self.shape[1] // self.patch_size)
        stored_grid = tuple(self.model.patch_embed.grid_size)
        if grid != stored_grid:
            with torch.no_grad():
                pos_embed = _resample_pos_embed(self.model.pos_embed, grid, stored_grid)
            self.model.pos_embed = nn.Parameter(pos_embed)
            self.model.patch_embed.grid_size = grid
            self.model.rope.update_feat_shape(grid)
        self._export = True

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Adapt a checkpoint to this encoder before loading it.

        - A ``pos_embed`` saved at another position grid is resampled to this encoder's grid, as ``rfdetr`` does for
          DINOv2 position embeddings when ``resolution`` (and with it ``positional_encoding_size``) differs.
        - PE-CLIP attention-pool / head weights (still present in rf-detr-internal checkpoints) are dropped: this
          encoder never runs them.
        - A patch-embedding kernel of another patch size is rejected with a clear error.

        Raises:
            ValueError: If the checkpoint's patch size differs from this encoder's.
        """
        unused_head = (f"{prefix}model.attn_pool.", f"{prefix}model.head.")
        for unused_key in [k for k in state_dict if k.startswith(unused_head)]:
            del state_dict[unused_key]
        kernel = state_dict.get(f"{prefix}model.patch_embed.proj.weight")
        own_kernel = self.model.patch_embed.proj.weight
        if isinstance(kernel, Tensor) and kernel.shape != own_kernel.shape:
            raise ValueError(
                f"The checkpoint's PE-Core-T patch embedding {tuple(kernel.shape)} does not match this encoder's "
                f"{tuple(own_kernel.shape)}: it was trained with patch_size={kernel.shape[-1]}, but the model is "
                f"configured with patch_size={self.patch_size}. Keep the default patch_size to load these weights."
            )
        key = f"{prefix}model.pos_embed"
        incoming = state_dict.get(key)
        own = self.model.pos_embed
        if isinstance(incoming, Tensor) and incoming.ndim == 3 and incoming.shape != own.shape:
            src_side = math.isqrt(incoming.shape[1] - 1)
            if src_side * src_side + 1 == incoming.shape[1] and incoming.shape[2] == own.shape[2]:
                grid = tuple(self.model.patch_embed.grid_size)
                state_dict[key] = _resample_pos_embed(incoming, grid, (src_side, src_side))
                logger.debug("Resampled %s from a %dx%d to a %dx%d grid.", key, src_side, src_side, *grid)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        """Encode images into one ``(B, C, H / patch_size, W / patch_size)`` map per ``out_feature_indexes`` entry.

        Args:
            x: Images ``(B, 3, H, W)`` with ``H`` and ``W`` divisible by ``patch_size * num_windows``.

        Returns:
            Feature maps after the trunk's final norm, one per ``out_feature_indexes`` entry.

        Raises:
            ValueError: If ``H`` or ``W`` is not divisible by ``patch_size * num_windows``.
        """
        model = self.model
        patch_size, num_windows = self.patch_size, self.num_windows
        batch, _, height, width = x.shape
        block_size = patch_size * num_windows
        if height % block_size or width % block_size:
            raise ValueError(
                f"PE-Core-T input height and width must be divisible by {block_size} "
                f"(patch_size={patch_size} * num_windows={num_windows}), got {height}x{width}."
            )
        grid_h, grid_w = height // patch_size, width // patch_size
        take_indices, max_index = feature_take_indices(len(model.blocks), self.out_feature_indexes)

        # Class token + position embedding, as timm's `_pos_embed` does, but resampling only when the grid differs
        # from the stored one: timm also resamples non-square grids of the stored size, which would put antialiased
        # bicubic interpolation into a baked non-square export graph. RoPE is rebuilt here too because timm's
        # `get_embed` returns the cached rows for any grid.
        x = model.patch_embed(x).reshape(batch, grid_h * grid_w, -1)
        pos_embed = model.pos_embed
        stored_grid = tuple(model.patch_embed.grid_size)
        if (grid_h, grid_w) != stored_grid:
            pos_embed = _resample_pos_embed(pos_embed, (grid_h, grid_w), stored_grid)
        x = torch.cat((model.cls_token.expand(batch, -1, -1), x), dim=1) + pos_embed
        x = model.pos_drop(x)
        rope = self.rope_for_grid(grid_h, grid_w)
        x = model.norm_pre(x)

        num_windows_squared = num_windows**2
        rope_dim = rope.shape[-1]
        # RoPE rows are [sin, cos]; sin=0, cos=1 leaves the class token unrotated.
        no_op_rope = torch.cat(
            [
                torch.zeros(rope_dim // 2, device=x.device, dtype=rope.dtype),
                torch.ones(rope_dim // 2, device=x.device, dtype=rope.dtype),
            ]
        )
        if num_windows > 1:
            cls_token = x[:, :1]
            x, windowed_rope = apply_windowing_with_rope(x[:, 1:], rope, num_windows, patch_size, height, width)
            x = torch.cat((cls_token.repeat(num_windows_squared, 1, 1), x), dim=1)
            windowed_rope = torch.cat(
                (no_op_rope.unsqueeze(0).unsqueeze(0).expand(num_windows_squared, 1, rope_dim), windowed_rope), dim=1
            )
            full_rope = windowed_rope.view(num_windows_squared * windowed_rope.shape[1], rope_dim)
            windowed_rope = windowed_rope.repeat(batch, 1, 1).unsqueeze(1)
        else:
            full_rope = torch.cat((no_op_rope.unsqueeze(0), rope), dim=0)
            windowed_rope = full_rope

        intermediates = []
        for index, block in enumerate(model.blocks[: max_index + 1]):
            full_attention = num_windows > 1 and index in self.out_feature_indexes
            if full_attention:
                _, tokens, channels = x.shape
                x = x.view(batch, num_windows_squared * tokens, channels)
            block_rope = full_rope if full_attention else windowed_rope
            if model.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, rope=block_rope, use_reentrant=False)
            else:
                x = block(x, rope=block_rope)
            if full_attention:
                _, tokens, channels = x.shape
                x = x.view(batch * num_windows_squared, tokens // num_windows_squared, channels)
            if index in take_indices:
                intermediates.append(model.norm(x))

        features = []
        for y in intermediates:
            y = y[:, 1:]
            if num_windows > 1:
                y = undo_windowing(y, num_windows, patch_size, height, width)
            features.append(y.reshape(batch, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous())
        return features


class PECoreBackbone(Backbone):
    """``rfdetr`` backbone with the PE-Core-T encoder and its layer-wise learning-rate rules."""

    def _build_encoder(
        self,
        name: str,
        *,
        out_feature_indexes: list[int] | None,
        target_shape: tuple[int, int],
        gradient_checkpointing: bool,
        load_pretrained_weights: bool,
        patch_size: int,
        num_windows: int,
        positional_encoding_size: int,
        drop_path: float,
        window_block_indexes: list[int] | None,
    ) -> nn.Module:
        """Build the PE-Core-T encoder; ``out_feature_indexes`` sets the full-attention blocks, all others are windowed.

        Raises:
            ValueError: If ``out_feature_indexes`` is empty or ``window_block_indexes`` is set.
        """
        if not out_feature_indexes:
            raise ValueError("out_feature_indexes must be non-empty for the PE-Core-T encoder.")
        if window_block_indexes is not None:
            raise ValueError(
                "window_block_indexes is not supported by the PE-Core-T encoder; the blocks in out_feature_indexes "
                "use full attention and every other block is windowed."
            )
        return PECoreEncoder(
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            load_pretrained_weights=load_pretrained_weights,
            drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
        )

    def get_named_param_lr_pairs(self, args: Any, prefix: str = "backbone.0") -> dict[str, dict[str, Any]]:
        """Per-parameter learning rate and weight decay for the trainable encoder parameters.

        Layer 0 holds the embeddings, so ``num_layers`` is ``out_feature_indexes[-1] + 2``.

        Args:
            args: Namespace with ``lr_encoder``, ``lr_vit_layer_decay``, ``lr_component_decay``, ``weight_decay`` and
                ``out_feature_indexes``.
            prefix: Name prefix of this backbone inside the detector.

        Returns:
            ``{parameter name: {"params", "lr", "weight_decay"}}`` for every trainable encoder parameter.
        """
        num_layers = args.out_feature_indexes[-1] + 2
        named_param_lr_pairs = {}
        for name, param in self.named_parameters():
            name = f"{prefix}.{name}"
            if f"{prefix}.encoder" in name and param.requires_grad:
                lr = (
                    args.lr_encoder
                    * get_pe_lr_decay_rate(name, lr_decay_rate=args.lr_vit_layer_decay, num_layers=num_layers)
                    * args.lr_component_decay**2
                )
                named_param_lr_pairs[name] = {
                    "params": param,
                    "lr": lr,
                    "weight_decay": args.weight_decay * get_pe_weight_decay_rate(name),
                }
        return named_param_lr_pairs


register_backbone(PE_CORE_T_ENCODER, PECoreBackbone)
