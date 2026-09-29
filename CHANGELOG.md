# Changelog

## Unreleased

### Added

- `RFDETRAtto`, `RFDETRFemto` and `RFDETRPico`: real-time detection models with Meta's Perception Encoder PE-Core-T (`timm` `vit_pe_core_tiny_patch16_384`) backbone, the architectures selected by neural architecture search over a PE-Core-T RF-DETR supernet trained on COCO. Atto runs at 380x380 with one attention window and no decoder layers, Femto at 384x384 with two windows and two decoder layers, and Pico at 560x560 with two windows and three decoder layers. Their COCO weights are the supernet's, extracted per architecture. They use the standard `rfdetr` training, inference and export APIs, like the XLarge models.
- `rfdetr_plus.models.pe_core`: the PE-Core-T encoder and backbone, registered with `rfdetr` for `ModelConfig.encoder="pe_core_t"`. Inputs at resolutions other than the model's (multi-scale training, custom `resolution`) resample the position embedding and RoPE inside the forward pass without changing any parameter, and `export()` bakes them for the export shape.

### Changed

- Minimum `rfdetr` dependency raised from `1.8.0` to `1.12.0` for its backbone registry, `ModelConfig.dim_feedforward` and export-shape hook; `timm>=1.0.27,<2` added for the PE-Core-T trunk.

## 1.1.0 — 2026-09-22

### Changed

- Minimum `rfdetr` dependency raised from `1.6.0` to `1.8.0`, ensuring compatibility with upstream `compile`/`cuda_graphs` support; docs gain compile version-gating guidance (`compile=True` is a no-op below rfdetr 1.10.0, and not `multi_scale`-safe until 1.10.1). ([#38](https://github.com/roboflow/rf-detr_plus/pull/38), [#64](https://github.com/roboflow/rf-detr_plus/pull/64))
- Config resolution moved from `get_model_config()`/`get_train_config()` method overrides to the `_model_config_class` class attribute — internal simplification, no observable behavior change (constructor overrides still resolve correctly by inheritance; base `RFDETR` already declared the same attribute at class level, so `from_checkpoint()`'s config-field filtering is unaffected). ([#64](https://github.com/roboflow/rf-detr_plus/pull/64))

### Fixed

- No-download construction restored: `pretrain_weights=None` now builds architecture without fetching weights on both Plus configs (previously failed pydantic validation, since `pretrain_weights` was narrowed to `str`). ([#64](https://github.com/roboflow/rf-detr_plus/pull/64))

## 1.0.2 — 2026-04-08

### Fixed

- Updated minimum `rfdetr` dependency from `1.4.3` to `1.6.0` and fixed deprecated import paths for compatibility with rfdetr 1.6+. ([#15](https://github.com/roboflow/rf-detr_plus/pull/15))

## 1.0.1 — 2026-02-19

### Changed

- Refactored model assets structure for `rfdetr` 1.4.3+ compatibility. ([#6](https://github.com/roboflow/rf-detr_plus/pull/6))

## 1.0.0 — 2026-02-11

Initial release. RF-DETR's XLarge and 2XLarge models split out of the core `rfdetr` package into this dedicated `rfdetr_plus` package — high-accuracy models using a DINOv2 backbone, licensed under the Platform Model License 1.0 (PML-1.0), distinct from `rfdetr`'s Apache-2.0 core (Nano/Small/Medium/Large).

### Added

- `RFDETRXLarge`, `RFDETR2XLarge` model classes, moved from `rfdetr` core.
- Explicit license acceptance required to construct Plus models: `accept_platform_model_license=True`.
