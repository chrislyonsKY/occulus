# Agent: Python Expert — Occulus

> Read `CLAUDE.md` before proceeding.
> Then read `ai-dev/architecture.md` and `ai-dev/patterns.md`.
> Then read `ai-dev/guardrails/coding-standards.md`.

## Role

Implement business logic, data processing functions, and library code following established project patterns.

## Responsibilities

- Implement functions to the interface defined by the Architect
- Apply patterns from `ai-dev/patterns.md`
- Write unit tests alongside every new function
- Does NOT design module structure — that's the Architect's domain

## Patterns

### Function Template

```python
import logging
from occulus.exceptions import OcculusError

logger = logging.getLogger(__name__)


def {{function_name}}(
    param1: {{Type1}},
    param2: {{Type2}} = {{default}},
) -> {{ReturnType}}:
    """{{One-sentence summary}}.

    Parameters
    ----------
    param1 : {{Type1}}
        {{description}}
    param2 : {{Type2}}, optional
        {{description}}, by default {{default}}

    Returns
    -------
    {{ReturnType}}
        {{description}}

    Raises
    ------
    ValueError
        If {{invalid_input_condition}}.
    OcculusError
        If {{domain_error_condition}}.
    """
    # Validate inputs early
    if not param1:
        raise ValueError("param1 must not be empty")

    logger.debug("{{function_name}} called with param1=%s", param1)

    try:
        result = _do_work(param1, param2)
    except SomeExternalError as exc:
        raise OcculusError(f"Failed to {{do thing}}: {exc}") from exc

    logger.info("{{function_name}} completed: %d items", len(result))
    return result
```

### Test Template

```python
import pytest
from occulus import {{function_name}}
from occulus.exceptions import OcculusError


def test_{{function_name}}_happy_path():
    result = {{function_name}}(valid_param)
    assert result is not None
    assert len(result) > 0


def test_{{function_name}}_empty_input_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        {{function_name}}("")


def test_{{function_name}}_domain_error():
    with pytest.raises(OcculusError):
        {{function_name}}(bad_param_that_triggers_external_failure)


@pytest.mark.integration
def test_{{function_name}}_live():
    """Requires network. Run with: pytest -m integration"""
    result = {{function_name}}(real_param)
    assert result is not None
```

## Review Checklist

- [ ] Docstring present (NumPy style, all sections)
- [ ] Type hints on all parameters and return value
- [ ] Input validation before any I/O
- [ ] External exceptions caught and reraised as `OcculusError`
- [ ] `logger.debug/info/warning/error` used — never `print()`
- [ ] Unit test written alongside implementation
- [ ] Optional imports are lazy with helpful error messages
