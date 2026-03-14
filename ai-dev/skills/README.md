# ai-dev/skills/README.md — Occulus

> Project-specific SKILL.md files with deep domain knowledge for AI agents.
> These supplement (not replace) the generic skills in the shared library.

## Available Skills

| Skill | File | Apply When |
|---|---|---|
| Occulus Domain Patterns | `occulus-skill.md` | Writing any code for this package |

## How to Use

Reference from a Claude Code prompt:

```
Read CLAUDE.md.
Read ai-dev/skills/occulus-skill.md before writing any code.

Task: [your task]
```

Or reference from an agent file:

```markdown
Before implementing, read ai-dev/skills/occulus-skill.md for
project-specific patterns and field names.
```

## Forking from the Shared Library

If the chrislyonsKY shared skills library has a relevant skill (e.g., `arcpy-enterprise-automation`,
`arcgis-pro-sdk-addin`, `cloud-native-geospatial`), fork it here and customize with:
- Project-specific field names and feature class paths
- Concrete connection string patterns for this project
- Domain vocabulary specific to this dataset or workflow
