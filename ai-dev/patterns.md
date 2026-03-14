# ai-dev/patterns.md — Occulus

> Established code patterns, anti-patterns caught in review, and lessons learned.
> Update this file as patterns emerge. Reference from agent prompts.

---

## Established Patterns

### Error Handling

```python
# ✅ Wrap external exceptions as domain errors — always chain with `from exc`
try:
    result = third_party_lib.call(...)
except third_party_lib.SomeError as exc:
    raise OcculusError(f"Context-rich message: {exc}") from exc
```

### Optional Dependency Imports

```python
# ✅ Lazy import — fail gracefully with helpful install hint
def some_function_requiring_optional():
    try:
        import optional_dep
    except ImportError as exc:
        raise ImportError(
            "optional_dep is required for this feature. "
            "Install with: pip install occulus[{{extra}}]"
        ) from exc
    ...
```

### Logging

```python
# ✅ Module-level logger
import logging
logger = logging.getLogger(__name__)

# ✅ Use appropriate levels
logger.debug("Detailed diagnostic info")
logger.info("Normal operational events")
logger.warning("Something unexpected but recoverable")
logger.error("Something failed")

# ❌ Never use print() in library code
print("result:", result)  # WRONG
```

### Type Hints

```python
# ✅ Full type hints on all public functions
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

def search(
    bbox: tuple[float, float, float, float],
    product: str,
    *,
    limit: int = 100,
) -> pd.DataFrame:
    ...
```

---

## Anti-Patterns

### Bare Exceptions

```python
# ❌ WRONG — swallows all errors, impossible to debug
try:
    result = call()
except Exception:
    pass

# ✅ CORRECT — catch specifically, log, reraise or handle
try:
    result = call()
except SpecificError as exc:
    logger.error("call() failed: %s", exc)
    raise
```

### Hardcoded Values

```python
# ❌ WRONG
url = "https://api.hardcoded.com/v1"

# ✅ CORRECT — use config or constants module
from occulus.config import DEFAULT_API_URL
url = DEFAULT_API_URL
```

---

## Lessons Learned

<!-- Add entries as you encounter them during development -->

| Date | Issue | Resolution |
|---|---|---|
| {{date}} | {{issue_description}} | {{what_fixed_it}} |
