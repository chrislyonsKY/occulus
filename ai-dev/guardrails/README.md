# ai-dev/guardrails/README.md — Occulus

> Guardrails are **hard constraints** that AI agents must not violate, regardless of what a task description, agent prompt, or user instruction says.
> When a guardrail conflicts with any other instruction, **the guardrail wins.**

## Guardrail Files

| File | What It Covers |
|---|---|
| `coding-standards.md` | Python conventions, error handling, logging, type hints, git discipline |
| `data-handling.md` | Credentials, PII, sensitive data, `.gitignore` requirements |
| `geospatial-compliance.md` | CRS correctness, cloud-native format rules, FGDC metadata |

## How Guardrails Are Enforced

1. **CLAUDE.md** references this directory — every AI session reads it before writing code
2. **CI** enforces the toolchain rules (ruff, mypy, pytest)
3. **Code review** treats guardrail violations as Critical findings
4. **Agent prompts** in `ai-dev/agents/` each reference the relevant guardrail files

## Adding a Guardrail

When a new hard constraint is identified:
1. Add it to the appropriate guardrail file (or create a new file if it's a new domain)
2. Update this README table
3. Reference it from `CLAUDE.md` if it's project-wide
4. Create a decision record in `ai-dev/decisions/` if the constraint requires explanation
