# CLAUDE.md — Occulus
> Multi-platform point cloud analysis — registration, segmentation, meshing, and feature extraction
> Python 3.11+ · Hatch · pybind11 · NumPy · Open3D · laspy

Read this file completely before doing anything.
Then read `ai-dev/architecture.md` for full system design.
Then read `ai-dev/guardrails/` for hard constraints that override all other guidance.

---

## Workflow Protocol

### Task Execution

When starting a new task:
1. Read CLAUDE.md (this file)
2. Read `ai-dev/architecture.md`
3. Read `ai-dev/guardrails/` — these constraints override all other guidance
4. Read the relevant `ai-dev/agents/` file for your role
5. Check `ai-dev/decisions/` for prior decisions that may affect your work
6. Check `ai-dev/skills/` for domain patterns specific to this project

When implementing a feature:
1. State your understanding of the task
2. Identify which modules are affected
3. Show the plan (files to create/modify, interfaces to implement)
4. Do not write code until the user types **Engage**
5. After completing all components, summarize the full change set

A feature is not complete until:
- Unit tests pass with mocked I/O
- Docstrings are present on all public functions and modules
- Any validation tool outputs have been checked (e.g., `rio cogeo validate` for COGs)

### Code Standards

When writing code:
- Every function gets a docstring (purpose, parameters, returns, raises) — NumPy style
- Every module gets a module-level docstring
- All I/O operations should be async where appropriate
- Error handling in every function — no happy-path-only code
- Use `logging` — never bare `print()` in library code

### Git Discipline

Commit messages use Conventional Commits with module scope:

```
feat({{module}}): add {{feature}}
fix({{module}}): handle {{edge case}}
test({{module}}): add tests for {{behavior}}
docs({{module}}): update {{doc}}
```

**Commit granularity:** one logical change per commit.
**Do not create empty files.** If a file is in the repo, it has working content.
**Every new module must ship with tests.**

---

## Compatibility Matrix

| Component | Version | Notes |
|---|---|---|
| Python | >=3.11 | Required |
| Hatch | >=1.7 | Build and env management |
| ruff | >=0.4 | Linting and formatting |
| mypy | >=1.10 | Type checking (strict mode) |
| pytest | >=8.0 | Testing |
| laspy | >=2.5 | LAS/LAZ I/O (optional: `occulus[las]`) |
| open3d | >=0.17 | PLY I/O, meshing, visualization (optional: `occulus[viz]`) |

---

## Project Structure

```
occulus/
├── CLAUDE.md                  ← You are here
├── README.md                  ← Human-facing: problem, install, usage
├── ARCHITECTURE.md            ← System design, module interfaces, data flow
├── CONTRIBUTING.md            ← Engineering standards, git conventions
├── CHANGELOG.md               ← Version history
├── LICENSE                    ← GPL-3.0
├── pyproject.toml             ← Hatch build config, deps, tool config
├── mkdocs.yml                 ← Docs site config
├── ai-dev/                    ← AI development infrastructure
│   ├── architecture.md        ← Full system design (agents read this second)
│   ├── spec.md                ← Requirements, acceptance criteria, milestones
│   ├── patterns.md            ← Code patterns, anti-patterns, lessons learned
│   ├── prompt-templates.md    ← Reusable Claude Code prompts for this project
│   ├── agents/                ← Specialized agent configurations
│   ├── decisions/             ← Architectural decision records (DL-XXX)
│   ├── guardrails/            ← Hard constraints AI must not violate
│   └── skills/                ← Project-specific SKILL.md files
├── .github/
│   ├── copilot-instructions.md
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
├── src/occulus/      ← Package source (src layout)
├── tests/                     ← pytest test suite
├── docs/                      ← mkdocs-material site
├── examples/
│   ├── notebooks/             ← Jupyter notebooks
│   └── scripts/               ← Standalone runnable scripts
└── branding/                  ← Logos, banners, hex stickers
```

---

## Architecture Summary

Occulus is a pure-Python point cloud analysis library for aerial (ALS), terrestrial (TLS), and UAV LiDAR data. It provides platform-aware types, format I/O (LAS/LAZ/PLY/XYZ), filtering, normal estimation, ICP and FPFH+RANSAC registration, CSF/PMF ground classification, DBSCAN/CHM-watershed segmentation, Poisson/BPA surface meshing, eigenvalue feature extraction, canopy metrics, and Open3D visualization — with scipy/numpy as the only required runtime dependencies.

See `ai-dev/architecture.md` for the full system design.

---

## Critical Conventions

- **Import paths**: Always `from occulus.{module} import {name}`. No relative imports in public API.
- **Error types**: All exceptions inherit from `occulus.exceptions.OcculusError`.
- **Docstrings**: NumPy style. Every public function. Every module.
- **Type hints**: Required on all public functions and class members.
- **Testing**: Mock I/O in unit tests. Tag integration tests with `@pytest.mark.integration`.
- **Logging**: Use `logging.getLogger(__name__)` — never `print()`.
- **Optional deps**: Lazy-import inside functions; fail gracefully with `ImportError` + install hint.

---

## What NOT To Do

- Do NOT use bare `print()` in library code. Use `logging`.
- Do NOT hardcode credentials, API keys, or connection strings anywhere in source.
- Do NOT write happy-path-only code. Error handling is required.
- Do NOT create empty files or directories for future work. Build what exists.
- Do NOT commit code without understanding every line.
- Do NOT ignore `ai-dev/guardrails/` constraints — they override agent suggestions.
- Do NOT add a `occulus/__init__.py` that re-exports everything blindly. Be intentional.
