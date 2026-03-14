# Coding Standards Guardrails — Occulus

> These rules apply to ALL code generated for this project, regardless of which agent is active.
> Violations are **Critical** findings in any code review.
> When a guardrail conflicts with any other instruction, **the guardrail wins.**

---

## Python

### Documentation
- Every public function **must** have a NumPy-style docstring (summary, Parameters, Returns, Raises)
- Every module **must** have a module-level docstring
- Private helpers (`_name`) should have at minimum a one-line docstring

### Error Handling
- Every function that performs I/O or calls external code **must** have a `try/except`
- Never catch bare `Exception` unless immediately reraising
- External exceptions must be caught and reraised as `OcculusError` (chain with `from exc`)
- Never use bare `pass` in an `except` block

### Logging
- Use `logging.getLogger(__name__)` at module level
- **Never use `print()` in library code** — ruff T20 rule enforces this
- Use appropriate levels: `debug` for diagnostic, `info` for milestones, `warning` for recoveries, `error` for failures

### Type Hints
- All public function parameters and return values **must** be typed
- Use `from __future__ import annotations` for forward references
- Use `X | Y` union syntax (Python 3.10+) not `Optional[X]`
- Never use `Any` without a justifying comment

### Imports
- No wildcard imports (`from module import *`)
- No circular imports between project modules
- Optional third-party deps **must** be lazy-imported inside functions with a helpful `ImportError` message
- Standard library → third party → project-local, separated by blank lines (ruff isort enforces)

### Testing
- Every public function ships with at least one unit test
- Unit tests **must** mock all I/O — no real network calls
- Integration tests (real I/O) **must** be tagged `@pytest.mark.integration`
- Never commit a test that is skipped by default and never run

### Credentials and Configuration
- **Never hardcode** credentials, API keys, URLs, or connection strings in source files
- Use `occulus.config` or environment variables
- Connection strings and secrets go in `.env` (gitignored) — never in `pyproject.toml`

### Miscellaneous
- No bare `assert` in production code — use proper validation with `ValueError`/`OcculusError`
- `pathlib.Path` over `os.path` for all file path operations
- f-strings over `.format()` or `%` formatting
- Line length: 100 characters (ruff enforces)

---

## Git

- Commit messages use Conventional Commits: `feat(module): description`
- One logical change per commit
- No empty files committed — every file has working content
- No AI-generated code committed without understanding every line
- No `TODO` committed without a linked issue or decision record

---

## Toolchain (non-negotiable versions)

| Tool | Minimum | Enforced By |
|---|---|---|
| ruff | 0.4 | CI |
| mypy | 1.10 (strict) | CI |
| pytest | 8.0 | CI |
| Python | >=3.11 | CI matrix |
