---
name: Bug report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**Which interface**
Python (`resolve_core`), R (`resolve`), or the `resolve` command line.

**To Reproduce**
Minimal code to reproduce the behavior:

```python
import resolve_core as rc
# Your code here
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened. Include error messages if any.

**Environment**
- OS: [e.g., Windows 11, Ubuntu 22.04, macOS 14]
- RESOLVE version: `python -c "import resolve_core; print(resolve_core.__version__)"`,
  or `resolve version`, or `resolve.version()` in R
- Python version: [e.g., 3.12.0]
- PyTorch version: [e.g., 2.6.0]
- R version, if reporting through the R package: [e.g., 4.6.0]
- CUDA version (if applicable): [e.g., 12.4]

**Additional context**
Add any other context about the problem here.
