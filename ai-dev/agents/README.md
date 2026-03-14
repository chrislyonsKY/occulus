# ai-dev/agents/README.md — Occulus

> Agent inventory. Each file contains a specialized persona for a development task.
> All agents read `CLAUDE.md` first, then their specific file.

## Agent Roster

| Agent | File | Use When |
|---|---|---|
| Architect | `architect.md` | Designing new modules, reviewing structure, ADRs |
| Python Expert | `python_expert.md` | Implementing business logic, library code |
| QA Reviewer | `qa_reviewer.md` | Writing tests, reviewing for edge cases |
| Technical Writer | `technical_writer.md` | Docstrings, README, docs site content |
| GIS Domain Expert | `gis_domain_expert.md` | Geospatial logic, CRS handling, format questions |

## How to Use

Reference in a Claude Code prompt:

```
Read CLAUDE.md.
Then read ai-dev/agents/python_expert.md.
Task: [your task]
```

Or upload the relevant agent file alongside CLAUDE.md in a Claude chat session.
