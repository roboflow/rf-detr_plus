# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from typing import Any, Literal

from pydantic import Field, field_validator
from rfdetr.config import ModelConfig

try:
    from rfdetr.config import PathLikeStr
except ImportError:
    PathLikeStr = str
from rfdetr.detr import RFDETR

# Importing pe_core registers the "pe_core_t" encoder with rfdetr's build_backbone.
from rfdetr_plus.models.pe_core import PE_CORE_T_ENCODER


class RFDETRXLargeConfig(ModelConfig):
    encoder: Literal["dinov2_windowed_base"] = "dinov2_windowed_base"
    hidden_dim: int = 512
    dec_layers: int = 5
    sa_nheads: int = 16
    ca_nheads: int = 32
    dec_n_points: int = 4
    num_windows: int = 1
    patch_size: int = 20
    projector_scale: list[Literal["P4",]] = ["P4"]
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_classes: int = 365
    positional_encoding_size: int = 700 // 20
    resolution: int = 700
    pretrain_weights: PathLikeStr | None = "rf-detr-xlarge.pth"
    license: str = "PML-1.0"


class RFDETR2XLargeConfig(ModelConfig):
    encoder: Literal["dinov2_windowed_base"] = "dinov2_windowed_base"
    hidden_dim: int = 512
    dec_layers: int = 5
    sa_nheads: int = 16
    ca_nheads: int = 32
    dec_n_points: int = 4
    num_windows: int = 2
    patch_size: int = 20
    projector_scale: list[Literal["P4",]] = ["P4"]
    out_feature_indexes: list[int] = [3, 6, 9, 12]
    num_classes: int = 365
    positional_encoding_size: int = 880 // 20
    resolution: int = 880
    pretrain_weights: PathLikeStr | None = "rf-detr-xxlarge.pth"
    license: str = "PML-1.0"


class RFDETRXLarge(RFDETR):
    size: Literal["rfdetr-xlarge"] = "rfdetr-xlarge"
    _model_config_class = RFDETRXLargeConfig

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("accept_platform_model_license", None)
        super().__init__(**kwargs)


class RFDETR2XLarge(RFDETR):
    size: Literal["rfdetr-2xlarge"] = "rfdetr-2xlarge"
    _model_config_class = RFDETR2XLargeConfig

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("accept_platform_model_license", None)
        super().__init__(**kwargs)


class RFDETRPECoreTConfig(ModelConfig):
    """Architecture shared by the PE-Core-T models (RFDETRAtto, RFDETRFemto, RFDETRPico).

    A Perception Encoder PE-Core-T trunk with windowed attention, full attention at ``out_feature_indexes``, a single
    P4 projector and a narrow deformable decoder. The sizes differ in resolution, patch size, windows, decoder depth
    and query count.
    """

    encoder: Literal["pe_core_t"] = PE_CORE_T_ENCODER
    hidden_dim: int = 128
    sa_nheads: int = 4
    ca_nheads: int = 8
    dec_n_points: int = 2
    dim_feedforward: int = Field(default=1024, ge=1)
    projector_scale: list[Literal["P4",]] = ["P4"]
    out_feature_indexes: list[int] = [2, 5, 8, 11]
    license: str = "PML-1.0"

    @field_validator("num_channels")
    @classmethod
    def _rgb_only(cls, value: int) -> int:
        """The PE-Core-T patch embedding is RGB-only; ``rfdetr``'s channel adaptation targets DINOv2."""
        if value != 3:
            raise ValueError(f"PE-Core-T models support num_channels=3 only, got {value}.")
        return value


class RFDETRAttoConfig(RFDETRPECoreTConfig):
    dec_layers: int = 0
    num_windows: int = 1
    patch_size: int = 20
    num_queries: int = 300
    num_select: int = 300
    positional_encoding_size: int = 380 // 20
    resolution: int = 380
    pretrain_weights: PathLikeStr | None = "rf-detr-atto.pth"


class RFDETRFemtoConfig(RFDETRPECoreTConfig):
    dec_layers: int = 2
    num_windows: int = 2
    patch_size: int = 16
    num_queries: int = 200
    num_select: int = 200
    positional_encoding_size: int = 384 // 16
    resolution: int = 384
    pretrain_weights: PathLikeStr | None = "rf-detr-femto.pth"


class RFDETRPicoConfig(RFDETRPECoreTConfig):
    dec_layers: int = 3
    num_windows: int = 2
    patch_size: int = 20
    num_queries: int = 200
    num_select: int = 200
    positional_encoding_size: int = 560 // 20
    resolution: int = 560
    pretrain_weights: PathLikeStr | None = "rf-detr-pico.pth"


class RFDETRAtto(RFDETR):
    size: Literal["rfdetr-atto"] = "rfdetr-atto"
    _model_config_class = RFDETRAttoConfig

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("accept_platform_model_license", None)
        super().__init__(**kwargs)


class RFDETRFemto(RFDETR):
    size: Literal["rfdetr-femto"] = "rfdetr-femto"
    _model_config_class = RFDETRFemtoConfig

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("accept_platform_model_license", None)
        super().__init__(**kwargs)


class RFDETRPico(RFDETR):
    size: Literal["rfdetr-pico"] = "rfdetr-pico"
    _model_config_class = RFDETRPicoConfig

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("accept_platform_model_license", None)
        super().__init__(**kwargs)
