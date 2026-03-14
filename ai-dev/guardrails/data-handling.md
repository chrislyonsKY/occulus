# Data Handling Guardrails — Occulus

> Hard rules for credentials, PII, sensitive data, and external data access.
> These override all other guidance. No exceptions.

---

## Credentials

- **NEVER** hardcode passwords, API keys, tokens, or connection strings in any source file
- **NEVER** log credential values, even at DEBUG level
- Credentials belong in:
  - Environment variables (`OCCULUS_API_KEY`, etc.)
  - `.env` file (gitignored — verify `.gitignore` before committing)
  - Named config profiles in `~/.occulus/config.toml`
- SDE connection files (`.sde`) are referenced by path — never embed credentials in code
- If a function receives a credential as a parameter, **mask it in logs**: `key[:4]...`

```python
# ✅ CORRECT
import os
api_key = os.environ.get("OCCULUS_API_KEY")
if not api_key:
    raise ValueError("OCCULUS_API_KEY environment variable not set")

# ❌ WRONG
api_key = "sk-abc123hardcoded"
```

---

## Sensitive Data in Logs

- Log IDs and identifiers — never log personal information
- Permit numbers, tile IDs, coordinates: OK to log
- Applicant names, addresses, SSNs, contact info: **never log**
- Query results containing PII must not be written to debug logs in full

```python
# ✅ CORRECT
logger.info("Processing permit %s", permit_id)

# ❌ WRONG
logger.debug("Applicant data: %s", applicant_record)  # may contain PII
```

---

## Output Files

- Files produced for external distribution must not contain internal connection strings
- File paths in logs should use relative paths — avoid UNC paths with server names
- Never write raw API responses (which may contain auth tokens) to disk without scrubbing

---

## Network Access

- Functions that make network calls must document this in their docstring
- All HTTP calls must use the project HTTP client (not raw `urllib` or `requests`)
- Respect `robots.txt` and rate limits for any public API
- No web scraping without explicit permission documented in a decision record
- External API URLs must be configurable — never hardcoded

---

## Cloud Storage

- S3/GCS/Azure access must use named profiles or environment credentials
- Never embed `AWS_ACCESS_KEY_ID` or equivalent in source code
- Public buckets (no credentials required) must be documented as such in comments

---

## `.gitignore` — Required Entries

The following must always be in `.gitignore`:

```
.env
*.env.*
config.local.toml
*.sde
*.ags
credentials.*
secrets.*
```

Verify these are present before the first commit.
