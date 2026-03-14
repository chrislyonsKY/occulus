# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report security issues by emailing: chris.lyons@ky.gov

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes

You will receive a response within 5 business days.

## Supported Versions

| Version | Supported |
|---|---|
| Latest release | ✅ |
| Older releases | ❌ |

## Scope

Security reports are in scope for:
- Credential or secret exposure via any code path
- Arbitrary code execution via parsing of untrusted input
- Path traversal vulnerabilities in file operations

Out of scope:
- Vulnerabilities in optional dependencies (report to their maintainers)
- Issues requiring physical access to the user's machine
