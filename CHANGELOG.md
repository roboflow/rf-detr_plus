# Changelog

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
