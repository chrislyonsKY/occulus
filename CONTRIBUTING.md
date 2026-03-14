# Contributing to Occulus

Thanks for your interest in contributing. Please read this guide before opening a PR.

## Engineering Standards

These are non-negotiable. PRs that violate them will not be merged regardless of feature value.

1. **Documentation first** — every public function gets a NumPy-style docstring before code review
2. **Tests alongside code** — every new function ships with at least one unit test
3. **No happy-path-only code** — error handling is required, not optional
4. **No empty commits** — every commit in the PR has a clear, purposeful change
5. **Conventional Commits** — `feat(module): description` / `fix(module): description`

## Development Setup

```bash
git clone https://github.com/chrislyonsKY/occulus.git
cd occulus
pip install -e ".[dev]"
```

## Running Checks

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/

# Unit tests (no network required)
pytest -m "not integration"

# All tests including integration (requires network)
pytest
```

All four must pass before opening a PR.

## Pull Request Process

1. Fork the repo and create a branch: `feat/your-feature` or `fix/your-bug`
2. Make your changes, keeping commits atomic and well-described
3. Ensure all checks pass locally
4. Open a PR with a description of what changed and why
5. Link any related issues

## Commit Message Format

```
feat(module): add windowed COG read
fix(stac): handle empty results from pagination
test(search): add edge case for antimeridian bbox
docs(readme): update quick start example
```

Scope is the module name (e.g., `search`, `download`, `config`).

## Scope Boundaries

Occulus is a **point cloud analysis library**. It is not:

- A GIS application or full raster processing stack (use GDAL/rasterio for raster workflows)
- A real-time streaming engine or interactive 3D editor
- A data acquisition or sensor control toolkit

Features outside this scope will not be accepted regardless of quality.
Open a discussion first if you're unsure whether something fits.

## Code of Conduct

Be direct, be kind, be useful. Harassment of any kind is not tolerated.
