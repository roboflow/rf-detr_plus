# RF-DETR+ - Agent Instructions

This file provides detailed technical context for AI coding agents working with RF-DETR+.

**Canonical Sources:**

- **Contribution Guidelines:** [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) - The authoritative source for all contribution practices
- **Human Documentation:** [README.md](README.md) - Project overview and usage
- **Copilot Instructions:** [.github/copilot-instructions.md](.github/copilot-instructions.md) - GitHub Copilot-specific guidance

This document supplements the contribution guidelines with detailed technical information for automated tooling.

## Agent Responsibilities

As an AI agent contributing to RF-DETR+, you are responsible for:

1. **Following test-driven development practices**

    - Write failing tests first for bug fixes
    - Write comprehensive tests for new features
    - Ensure final PR commit has all tests passing

2. **Adhering to code quality standards**

    - Run `pre-commit run --all-files` before every commit
    - Follow type hint and docstring requirements
    - Use direct imports (not `import ... as` pattern)

3. **Maintaining agentic documentation**

    - Update `AGENTS.md` when architecture patterns or technical conventions change
    - Update `.github/copilot-instructions.md` when high-level guidance changes
    - Update `.github/CONTRIBUTING.md` when human workflow is affected
    - Apply updates after receiving major feedback in PR reviews

4. **Consulting maintainers before major changes**

    - Open an issue before adding new models or significant features
    - Wait for approval on approach before implementing

5. **Writing secure, minimal code**

    - Avoid over-engineering and unnecessary abstractions
    - Write secure code (prevent injection vulnerabilities)
    - Follow existing patterns in the codebase

> [!NOTE]
> Keeping documentation current ensures consistency across agent contributions and reduces repeated feedback on the same issues.

## Build & Development Environment

> [!NOTE]
> **Canonical Reference:** See [Development Environment Setup](.github/CONTRIBUTING.md#development-environment-setup) in CONTRIBUTING.md for complete setup instructions.

### Setup

```bash
# Install uv (if not already installed)
pip install uv

# Full development environment (always use this)
uv sync --all-groups
```

**Prerequisites:** Python >=3.10 (tested on 3.10-3.13)

### Dependency Information

See `pyproject.toml` for complete dependency specifications:

- **Core:** `rfdetr>=1.5.0,<2` (Base RF-DETR package with all core dependencies)
- **Development:** `tests`, `build`, `dev` groups

**Important:** RF-DETR+ is an extension package that depends on the base `rfdetr` package. All core model architecture, training logic, and utilities come from `rfdetr`.

## Testing

> [!NOTE]
> **Canonical Reference:** See [Test-Driven Development](.github/CONTRIBUTING.md#test-driven-development) in CONTRIBUTING.md for complete guidelines.
>
> **CI Workflows (Source of Truth):** See `.github/workflows/` for exact test commands used in CI.

### Commands

```bash
# CPU tests (default for local development) - matches CI
uv run --no-sync pytest src/ tests/ -m "not gpu" --cov=rfdetr_plus --cov-report=xml

# GPU tests (requires GPU)
uv run --no-sync pytest src/ tests/ -m gpu

# Pre-commit checks (ALWAYS run before committing)
pre-commit run --all-files
```

### Testing Principles

> [!IMPORTANT]
> **Testing Requirements:**
>
> - ⚠️ **During development:** Tests may fail as you work through TDD cycle
> - ✅ **Before opening PR:** Final commit MUST have all tests passing
> - ✅ **Before each commit:** Run `pre-commit run --all-files`

**Test-Driven Development:**

1. **Bug fixes:** Write failing test → Fix code → Verify all tests pass
2. **New features:** Write comprehensive tests → Implement feature → Refactor

**Test Organization:**

- Group related tests in classes
- Use `@pytest.mark.parametrize` with `pytest.param(..., id="name")`
- Mark GPU/heavy tests with `@pytest.mark.gpu`

**CI Information:**
See [CI Testing](.github/CONTRIBUTING.md#ci-testing) in CONTRIBUTING.md for details on OS/Python version matrix and workflow configurations.

## Code Quality & Linting

> [!NOTE]
> **Canonical Reference:** See [Code Quality and Linting](.github/CONTRIBUTING.md#code-quality-and-linting) in CONTRIBUTING.md for setup and details.

### Command

```bash
# Always run full pre-commit (not individual tools)
pre-commit run --all-files
```

> [!TIP]
> Pre-commit hooks will auto-format many issues. Review changes and re-stage files.

**Configuration Files:**

- `.pre-commit-config.yaml` - Pre-commit hooks (ruff, mdformat, prettier, codespell)
- `pyproject.toml` - Ruff linting rules (`[tool.ruff]` section)

**License Header (required for all Python files):**

```python
# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
```

## Package Building

```bash
# Install build dependencies
uv sync --group build

# Build distributions
uv build

# Validate build
uv run twine check --strict dist/*
```

**Build outputs:**

- Source distribution: `dist/rfdetr_plus-*.tar.gz`
- Wheel: `dist/rfdetr_plus-*.whl`

## Project Structure

> [!NOTE]
> **Canonical Reference:** See [Project Structure](.github/CONTRIBUTING.md#project-structure) in CONTRIBUTING.md for complete project organization, directory descriptions, and configuration files.
>
> **Quick summary:** `src/rfdetr_plus/` (source code - XLarge and 2XLarge model variants), `tests/` (test suite), `.github/` (CI/CD), `pyproject.toml` (dependencies and config).
>
> RF-DETR+ is an extension package. Internal organization is simpler than the base `rfdetr` package.

## Architecture & Conventions

> [!NOTE]
> **Canonical Reference:** See [Architecture & Conventions](.github/CONTRIBUTING.md#architecture--conventions) in CONTRIBUTING.md for complete architecture patterns, import conventions, logging guidelines, and security best practices.

**Quick summary:**

- RF-DETR+ extends `rfdetr` package with XLarge and 2XLarge variants
- Always use direct imports (not `import ... as`)
- Use `logger.debug()` for detailed info, `logger.info()` for high-level status
- Follow security best practices (avoid injection vulnerabilities)

## Common Workflows

### Making Changes

1. **Setup:** `uv sync --all-groups`
2. **Before changes:** Run tests to establish baseline
3. **Development:**
    - Make minimal, focused changes
    - Follow existing patterns and conventions (see [Architecture & Conventions](.github/CONTRIBUTING.md#architecture--conventions))
    - Add type hints and docstrings
4. **Testing:**
    - Bug fixes: Write test first, then fix
    - Features: Test all major use cases
    - Run: `uv run --no-sync pytest src/ tests/ -m "not gpu"`
5. **Quality checks:** `pre-commit run --all-files`
6. **Build (if needed):** `uv build`
7. **Commit:** Pre-commit hooks run automatically

## CI/CD Workflows

GitHub Actions workflows in `.github/workflows/`:

- **Tests:** CPU and GPU test workflows across OS/Python versions
- **Build:** Package building and validation
- **Publishing:** Release automation

**Concurrency:** PRs cancel in-progress runs on new pushes

## Additional Resources

- **RF-DETR+ Repository:** https://github.com/roboflow/rf-detr-plus
- **Base RF-DETR Repository:** https://github.com/roboflow/rf-detr
- **Base RF-DETR Documentation:** https://rfdetr.roboflow.com
- **Issues:** https://github.com/roboflow/rf-detr-plus/issues
- **Discord:** https://discord.gg/GbfgXGJ8Bk
- **Contributing:** [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md)
- **Copilot Instructions:** [.github/copilot-instructions.md](.github/copilot-instructions.md)

---

**Note:** This file is designed for AI coding agents. For human-readable project information, see README.md. For contribution guidelines, see .github/CONTRIBUTING.md.
