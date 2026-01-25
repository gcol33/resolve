# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in RESOLVE, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly at gilles.colling@uliege.be
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix timeline**: Depends on severity, typically 2-4 weeks

## Security Best Practices

When using RESOLVE:

- Keep dependencies updated (`pip install --upgrade resolve`)
- Validate input data before processing
- Use the latest stable version
- Review model files before loading (don't load untrusted `.pt` files)

## Scope

This security policy covers:
- The RESOLVE Python package (`resolve`)
- The RESOLVE C++ core (`resolve-core`)
- The RESOLVE R package

Third-party dependencies are outside our direct control but we monitor for known vulnerabilities.
