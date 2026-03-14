# ai-dev/prompt-templates.md — Occulus

> Reusable Claude Code prompts for common development tasks on this project.
> Copy-paste into Claude Code (or any AI tool). Customize the bracketed sections.

---

## New Feature

```
Read CLAUDE.md, ai-dev/architecture.md, ai-dev/guardrails/.

I need to implement: [FEATURE DESCRIPTION]

Affected modules: [list modules from CLAUDE.md project structure]

Before writing any code:
1. State your understanding of the task
2. Identify all files that will be created or modified
3. Show the complete plan

Do not proceed until I type Engage.
```

---

## Bug Fix

```
Read CLAUDE.md, ai-dev/patterns.md, ai-dev/guardrails/.

Bug: [DESCRIPTION]
Reproduces: [steps or code]
Expected: [what should happen]
Actual: [what actually happens]

Before writing any code:
1. State your diagnosis of the root cause
2. Identify which files need to change
3. Explain the fix approach and any edge cases

Do not proceed until I type Engage.
```

---

## Code Review

```
Read CLAUDE.md, ai-dev/patterns.md, ai-dev/guardrails/.

Review [FILE OR MODULE] for:
- Adherence to conventions in CLAUDE.md (docstrings, error handling, logging)
- Compliance with ai-dev/guardrails/ constraints
- Correct type hints
- Missing edge cases
- Anti-patterns listed in ai-dev/patterns.md

Produce a numbered list of findings with severity: Critical / Warning / Info.
Do not suggest improvements beyond what the guardrails and CLAUDE.md require.
```

---

## Write Tests

```
Read CLAUDE.md, ai-dev/architecture.md, ai-dev/guardrails/coding-standards.md.

Write tests for [MODULE/FUNCTION].

Requirements:
- Unit tests: mock all I/O (use respx for HTTP, pytest fixtures for files)
- No real network calls in unit tests
- Tag integration tests with @pytest.mark.integration
- Follow patterns in tests/conftest.py

Cover:
- Happy path
- Empty/None inputs
- Invalid inputs (should raise OcculusError or ValueError)
- Edge cases: [list any specific edge cases]
```

---

## Write Docstrings

```
Read CLAUDE.md section "Critical Conventions" → Docstrings rule.

Add NumPy-style docstrings to all public functions and classes in [FILE].

Each docstring must include:
- One-sentence summary
- Parameters section with types
- Returns section with type
- Raises section for all exceptions the function can raise

Do not change any logic. Only add or improve docstrings.
```

---

## End-of-Session Commit Summary

```
Read CLAUDE.md → Git Discipline section.

Summarize all changes made this session.
Group into logical commits using Conventional Commits format with module scope:
  feat({{module}}): description
  fix({{module}}): description
  test({{module}}): description
  docs({{module}}): description

Show proposed commits. Do not run git until I type Engage.
```

---

## Add a Decision Record

```
Read ai-dev/decisions/ to find the next available DL number.

Create a new decision record for: [DECISION TOPIC]

Context: [why this decision is needed]
Decision: [what we chose]
Alternatives: [what else was considered and why rejected]
Consequences: [what this enables and what it constrains]

Write the file to ai-dev/decisions/DL-XXX-[topic-slug].md.
```
