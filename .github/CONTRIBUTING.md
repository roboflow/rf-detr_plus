# Contributing to RF-DETR+

Thank you for helping to advance RF-DETR+! Your participation is invaluable in evolving our platform—whether you're squashing bugs, refining documentation, or rolling out new features. Every contribution pushes the project forward.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Project Structure](#project-structure)
3. [Development Environment Setup](#development-environment-setup)
4. [Test-Driven Development](#test-driven-development)
5. [Code Quality and Linting](#code-quality-and-linting)
6. [CLA Signing](#cla-signing)
7. [Google-Style Docstrings and Mandatory Type Hints](#google-style-docstrings-and-mandatory-type-hints)
8. [Reporting Bugs](#reporting-bugs)
9. [Architecture & Conventions](#architecture--conventions)
10. [License](#license)

## How to Contribute

Your contributions can be in many forms—whether it's enhancing existing features, improving documentation, resolving bugs, or proposing new ideas. Here's a high-level overview to get you started:

1. [Fork the Repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo): Click the "Fork" button on our GitHub page to create your own copy.

2. [Clone Locally](https://docs.github.com/en/enterprise-server@3.11/repositories/creating-and-managing-repositories/cloning-a-repository): Download your fork to your local development environment.

3. [Create a Branch](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop): Use a descriptive name with appropriate prefix:

    ```bash
    # Branch naming convention: {type}/{issue_number}-name_or_description
    git checkout -b fix/123-authentication_bug
    git checkout -b feat/678-add_model_support
    git checkout -b docs/update_readme
    ```

    **Prefixes:** `fix/` (bug fixes), `feat/` (new features), `docs/` (documentation), `refactor/`, `test/`, `chore/`

4. Develop Your Changes: Make your updates, ensuring your commit messages clearly describe your modifications.

5. [Commit and Push](https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop): Run:

    ```bash
    git add .
    git commit -m "A brief description of your changes"
    git push -u origin your-descriptive-name
    ```

6. [Open a Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request): Submit your pull request against the main development branch. Please detail your changes and link any related issues.

Before merging, check that all tests pass and that your changes adhere to our development and documentation standards.

## Project Structure

Understanding the project structure will help you navigate the codebase and make contributions effectively.

```
rf-detr_plus/
├── .github/              # GitHub configuration
│   ├── workflows/        # CI/CD pipelines (tests, builds)
│   ├── CONTRIBUTING.md   # This file - contribution guidelines
│   ├── copilot-instructions.md  # GitHub Copilot-specific guidance
│   └── ISSUE_TEMPLATE/   # Issue templates
├── src/rfdetr_plus/      # Main package source code
│   ├── __init__.py       # Package entry point
│   └── models/           # Model implementations (XLarge, 2XLarge variants)
├── tests/                # Test suite
│   └── test_*.py         # Test files
├── pyproject.toml        # Project metadata, dependencies, tool configurations
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── README.md             # Project overview and quick start
├── LICENSE               # Platform Model License 1.0
└── AGENTS.md             # AI agent-specific technical documentation
```

**Key Directories:**

- **`src/rfdetr_plus/`** - All source code for the RF-DETR+ package

    - Contains extended model implementations (XLarge, 2XLarge variants)
    - Depends on the base `rfdetr` package (>=1.4.1)

- **`tests/`** - Test suite

    - Unit tests and integration tests
    - Use `@pytest.mark.gpu` for GPU-dependent tests

- **`.github/`** - GitHub-specific configuration

    - CI/CD workflows define automated testing and deployment
    - Contributing guidelines and issue templates

**Important Configuration Files:**

- **`pyproject.toml`** - Single source of truth for:

    - Project metadata and dependencies
    - Tool configurations (ruff, pytest, etc.)
    - Build system configuration

- **`.pre-commit-config.yaml`** - Defines pre-commit hooks for code quality

> [!TIP]
> When contributing, focus on the relevant directory for your change:
>
> - Bug fixes/features → `src/rfdetr_plus/` and `tests/`
> - CI/build issues → `.github/workflows/` or config files

## Development Environment Setup

RF-DETR+ uses **`uv`** as the package manager for dependency management. Ensure you have Python >=3.10 installed (supports 3.10, 3.11, 3.12, 3.13).

### Installing uv

```bash
pip install uv
```

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rf-detr-plus.git
cd rf-detr-plus

# Install all development dependencies
uv sync --all-groups

# Or install specific dependency groups
uv sync --group tests      # Testing dependencies only
uv sync --group build      # Build tools only
uv sync --group dev        # Development tools (pre-commit)
```

**Important:** Always run `uv sync` after pulling changes to ensure your dependencies are up to date.

### Dependency Information

RF-DETR+ extends the base RF-DETR package with additional model variants.

**Core dependency:**

- `rfdetr>=1.4.1,<2` - Base RF-DETR package

**Development groups:**

- `tests` - pytest, pytest-cov
- `build` - twine, wheel, build
- `dev` - pre-commit

See `pyproject.toml` for complete dependency specifications.

### Running Tests

> **CI Workflows as Source of Truth:** See `.github/workflows/` for the exact commands used in continuous integration.

```bash
# Run CPU tests (default for local development)
uv run --no-sync pytest src/ tests/ -m "not gpu" --cov=rfdetr_plus --cov-report=xml

# Run GPU tests (requires GPU)
uv run --no-sync pytest src/ tests/ -m gpu
```

**Development vs. PR Requirements:**

- **During development:** Tests may fail as you work through TDD cycle (write failing test → implement → fix)
- **Before opening PR:** Your final commit MUST have all tests passing
- **Before each commit:** Run `pre-commit run --all-files` to ensure code quality

### Building the Package

```bash
# Build source and wheel distributions
uv build

# Validate the build
uv run twine check --strict dist/*
```

## Test-Driven Development

We follow test-driven development practices to ensure code quality and prevent regressions.

### For Bug Fixes

1. **Write a test that replicates the issue** - The test should fail initially, demonstrating the bug
2. **Commit the failing test** (optional during development, but commit message should note "WIP" or "test for issue #XXX")
3. **Implement the fix** - Make the minimal change needed to make the test pass
4. **Verify all tests pass** - Ensure your fix doesn't break existing functionality
5. **Commit the fix** - This commit MUST have all tests passing before opening PR

**Note:** It's acceptable to have failing tests in intermediate commits during development. However, your **final commit before opening a PR must have all tests passing**. This aligns with test-driven development: first create a failing test that proves the bug exists, then fix it.

### For New Features

1. **Write tests covering all major use cases** - Think about edge cases, invalid inputs, and expected behaviors
2. **Implement the feature** - Build the feature to satisfy the test requirements
3. **Refactor if needed** - Clean up the implementation while keeping tests green

### Test Organization

**Use test classes to group related tests:**

```python
import pytest


class TestXLargeModel:
    def test_model_initialization(self):
        # Test code
        pass

    def test_inference(self):
        # Test code
        pass
```

**Use `pytest.mark.parametrize` to extend test cases:**

```python
import pytest


@pytest.mark.parametrize(
    "model_variant",
    [
        pytest.param("xlarge", id="xlarge"),
        pytest.param("2xlarge", id="2xlarge"),
    ],
)
def test_model_loading(model_variant):
    # Test code that runs for each model variant
    pass
```

**Mark GPU-required or computationally heavy tests:**

```python
import pytest


@pytest.mark.gpu  # Use this marker for GPU-dependent or heavy tests
def test_model_training():
    # Training test code
    pass
```

Tests marked with `@pytest.mark.gpu` are excluded from CPU CI workflows and run separately on GPU infrastructure.

### CI Testing

> [!NOTE]
> **CI Workflows (Source of Truth):** See `.github/workflows/` for exact commands.

Our continuous integration tests run on:

- **Operating Systems:** Ubuntu, Windows, macOS
- **Python Versions:** 3.10, 3.11, 3.12, 3.13
- **CPU Workflow:** `pytest -m "not gpu"` - Runs on all OS/Python combinations
- **GPU Workflow:** `pytest -m gpu` - Runs separately on GPU infrastructure

This ensures your changes work across all supported platforms and Python versions.

### Running Tests

```bash
# Run tests with coverage (recommended)
uv run --no-sync pytest src/ tests/ -m "not gpu" --cov=rfdetr_plus

# Run a specific test file
uv run --no-sync pytest tests/test_model.py

# Run a specific test
uv run --no-sync pytest tests/test_model.py::test_model_loading
```

## Code Quality and Linting

All code must pass linting and formatting checks before being merged. We use **pre-commit hooks** to automate this process.

> [!TIP]
> Pre-commit hooks will auto-format many issues. If pre-commit fails, review the changes it made and re-stage the files.

### Setting Up Pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Configuration:** See `.pre-commit-config.yaml` for all hooks and `pyproject.toml` for tool-specific settings (e.g., `[tool.ruff]`).

### Linting Tools

- **ruff** - Python linting and formatting
- **mdformat** - Markdown formatting
- **prettier** - YAML formatting
- **codespell** - Spell checking

## CLA Signing

In order to maintain the integrity of our project, every pull request must include a signed Contributor License Agreement (CLA). This confirms that your contributions are properly licensed. After opening your pull request, simply add a comment stating:

```text
I have read the CLA Document and I sign the CLA.
```

This step is essential before any merge can occur.

## Google-Style Docstrings and Mandatory Type Hints

For clarity and maintainability, any new functions or classes must include [Google-style docstrings](https://google.github.io/styleguide/pyguide.html) and use Python type hints. Type hints are mandatory in all function definitions, ensuring explicit parameter and return type declarations.

> [!IMPORTANT]
> Type hints are in the function signature. **Do not duplicate types in docstrings** - describe the parameter's purpose instead.

For example:

```python
def sample_function(param1: int, param2: int = 10) -> bool:
    """
    Provides a brief description of function behavior.

    Args:
        param1: Explanation of the first parameter's purpose.
        param2: Explanation of the second parameter, defaulting to 10.

    Returns:
        True if the operation succeeds, otherwise False.

    Examples:
        >>> sample_function(5, 10)
        True
    """
    return param1 == param2
```

Following this pattern helps ensure consistency throughout the codebase.

## Reporting Bugs

Bug reports are vital for continued improvement. When reporting an issue, please include a clear, minimal reproducible example that demonstrates the problem. Detailed bug reports assist us in swiftly diagnosing and addressing issues.

## Architecture & Conventions

Understanding the architecture and coding conventions will help you write consistent code that aligns with the project's patterns.

### Package Structure

RF-DETR+ is an extension package that provides additional model variants:

- Extends the base `rfdetr` package
- Provides XLarge and 2XLarge model variants
- Depends on `rfdetr>=1.4.1,<2` for all core functionality
- Licensed under Platform Model License (PML) 1.0

### Model Architecture

- XLarge and 2XLarge variants follow the same architecture patterns as base RF-DETR models
- All base architecture documentation from the `rfdetr` package applies
- Refer to [RF-DETR documentation](https://rfdetr.roboflow.com) for core architecture details

### Import Conventions

**Always use direct imports (NOT `import ... as` pattern):**

```python
# Base RF-DETR imports
from rfdetr.util.misc import get_rank, get_world_size, is_main_process
from rfdetr.util.logger import get_logger

# Logger usage
logger = get_logger()  # Default name: "rf-detr", reads LOG_LEVEL env var

# RF-DETR+ models
from rfdetr_plus.models.detection import RFDETRXLarge, RFDETR2XLarge
```

### Logging Conventions

- Use `logger.debug()` for detailed tensor/shape information (not `logger.info()`)
- Use `logger.info()` for high-level progress/status updates
- Logger reads `LOG_LEVEL` environment variable

### Subprocess Usage

When using subprocess, follow this pattern:

```python
import subprocess

result = subprocess.run(
    ["command", "arg1", "arg2"],
    check=True,  # Raise CalledProcessError on failure
    text=True,  # Return stdout/stderr as strings
    capture_output=True,
)
# Note: stderr is already a string, don't decode
```

### Security Best Practices

- **Write secure code:** Avoid injection vulnerabilities (XSS, SQL injection, command injection)
- **Validate inputs:** Especially for file paths, URLs, and user-provided data
- **No credentials:** Never commit API keys, tokens, or credentials
- **Follow OWASP best practices**

## License

By contributing to RF-DETR+, you agree that your contributions will be licensed under the Platform Model License 1.0 as specified in our [LICENSE](/LICENSE) file.

Thank you for your commitment to making RF-DETR+ better. We look forward to your pull requests and continued collaboration. Happy coding!

### License Headers

All Python files must start with the following header:

```python
# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
```
