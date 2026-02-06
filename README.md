# RF-DETR+

Extension package for [RF-DETR](https://github.com/roboflow/rf-detr) containing Platform Model License (PML) code.

## Installation

```bash
pip install rfdetr-plus
```

Or for development:

```bash
pip install -e ".[tests]"
```

## Structure

```
rf-detr-plus/
├── .github/
│   ├── workflows/          # CI/CD pipelines
│   └── ISSUE_TEMPLATE/     # Issue templates
├── src/
│   └── rfdetr_plus/        # Main package
│       ├── __init__.py
│       └── py.typed
├── tests/                  # Test suite
├── .codecov.yml
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── pyproject.toml
```

## Development

### Code Quality

We use `pre-commit` to ensure code quality:

```bash
pre-commit install
pre-commit run --all-files
```

### Tests

[`pytest`](https://docs.pytest.org/) is used to run tests:

```bash
pytest tests/ -v
```

### License Headers

All Python files must include the following license header:

```python
# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
```
