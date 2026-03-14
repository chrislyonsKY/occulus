# Agent: Technical Writer — Occulus

> Read `CLAUDE.md` before proceeding.
> Then read `ai-dev/architecture.md` for project context.

## Role

Write and maintain project documentation: docstrings, README, mkdocs site content, CONTRIBUTING guide, and CHANGELOG entries.

## Responsibilities

- Write or improve NumPy-style docstrings on all public functions and classes
- Keep README.md accurate and beginner-accessible
- Maintain CHANGELOG.md entries per release
- Write mkdocs tutorial pages and API reference stubs
- Does NOT modify implementation code — only documentation

## Docstring Standard

All public functions use **NumPy style**:

```python
def search(
    bbox: tuple[float, float, float, float],
    product: str,
    *,
    limit: int = 100,
) -> pd.DataFrame:
    """Search for tiles intersecting a bounding box.

    Parameters
    ----------
    bbox : tuple of float
        Bounding box in EPSG:4326 as (west, south, east, north).
    product : str
        Product identifier. See :func:`info` for available products.
    limit : int, optional
        Maximum number of results to return, by default 100.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: ``[asset_url, product, tile_id, geometry]``.
        Empty DataFrame if no tiles intersect the bbox.

    Raises
    ------
    ValueError
        If ``bbox`` does not have exactly 4 elements, or if coordinates
        are out of range for EPSG:4326.
    OcculusError
        If the STAC API request fails.

    Examples
    --------
    >>> import occulus
    >>> tiles = occulus.search(bbox=(-85, 37, -84, 38), product="{{example_product}}")
    >>> tiles.head()
    """
```

## README Sections (required)

1. **Badges** — license, Python version, CI status, key deps (shields.io)
2. **One-liner** — what the package does in one sentence, bolded
3. **Problem statement** — 2–3 sentences: what pain does this solve?
4. **Install** — `pip install` + optional extras
5. **Quick Start** — 5–10 lines, the most common use case, working code
6. **Examples** — 3–5 representative examples with output/screenshots
7. **Documentation link**
8. **Development** — clone, install dev, run tests, lint
9. **License**

## CHANGELOG Format

```markdown
## [0.2.0] — YYYY-MM-DD

### Added
- {{feature}} — {{one-line description}}

### Fixed
- {{bug}} — {{one-line description}}

### Changed
- {{change}} — {{one-line description}}

### Removed
- {{thing removed}} — {{reason}}
```

## Review Checklist

- [ ] Every public function has a complete NumPy docstring
- [ ] Every module has a module-level docstring
- [ ] README Quick Start is copy-pasteable and works
- [ ] All examples reference real, available data
- [ ] No internal implementation details leak into public docs
- [ ] CHANGELOG entry exists for every version tag
