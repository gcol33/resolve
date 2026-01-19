# Contribution Guidelines

First of all, thank you very much for taking the time to contribute
to the **RESOLVE** project!

This document provides guidelines for contributing to RESOLVE—its codebase and documentation.
These guidelines are meant to guide you, not to restrict you.
If in doubt, use your best judgment and feel free to propose improvements through an issue or pull request.

#### Table Of Contents

- [Code of Conduct](#code-of-conduct)
- [Installation](#installation)
  - [Obtaining the source](#obtaining-the-source)
  - [Setting up your Python environment](#setting-up-your-python-environment)
  - [Installing from source](#installing-from-source)
- [Testing](#testing)
- [Documentation](#documentation)
  - [Building the documentation](#building-the-documentation)
  - [Design of the docs](#design-of-the-docs)
- [Project organization](#project-organization)
- [Contributing workflow](#contributing-workflow)
- [Style guidelines](#style-guidelines)
- [Pull request checklist](#pull-request-checklist)
- [Reporting bugs](#reporting-bugs)

## Code of Conduct

This project and everyone participating in it is governed by our **Code of Conduct** (`CODE_OF_CONDUCT.md`).
By participating, you are expected to uphold this code and maintain a respectful, inclusive environment.

## Installation

This installation guide is focused on development.
For regular installation, please see the [README](./README.md).

### Obtaining the source

Clone the RESOLVE repository:

```bash
git clone https://github.com/gcol33/resolve.git
cd resolve
```

If you work on a development branch:

```bash
git checkout dev
git pull origin dev
```

### Setting up your Python environment

RESOLVE is a Python package built on PyTorch.

1. **Install required tools**
   - Python (>= 3.10)
   - Git
   - An editor or IDE (VS Code, PyCharm, etc.)

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

### Installing from source

Build and install the package locally:

```bash
pip install -e .
```

For a full installation with all optional dependencies:

```bash
pip install -e ".[dev,docs]"
```

## Testing

RESOLVE uses **pytest** for testing.
All tests are located in `tests/`.

Run the full test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=resolve --cov-report=html
```

Run a subset of tests during development:

```bash
pytest tests/test_dataset.py
pytest tests/test_encoder.py -k "test_hash"
```

Guidelines:
- Keep tests fast and reproducible.
- Use fixed random seeds for reproducibility.
- Include edge cases and expected failures.
- Prefer small synthetic examples to large datasets.

## Documentation

### Building the documentation

Build the documentation site locally:

```bash
cd docs
# If using mkdocs:
mkdocs build
mkdocs serve  # Local preview at http://localhost:8000
```

The generated site is saved in the `docs/site/` directory.

### Design of the docs

- API documentation: Generated from docstrings
- Tutorials and examples: `docs/tutorials/`
- Website configuration: `mkdocs.yml`
- Package overview: `README.md`
- Changelog: `NEWS.md`

## Project organization

```
resolve/
├── .github/                <- Continuous integration workflows
├── .gitignore
├── pyproject.toml          <- Package metadata and dependencies
├── LICENSE.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── NEWS.md
├── README.md
├── src/
│   └── resolve/            <- Main package source
│       ├── __init__.py
│       ├── data/           <- Dataset and role handling
│       ├── encode/         <- Species encoding
│       ├── model/          <- Neural network architecture
│       ├── train/          <- Training loop and loss functions
│       └── inference/      <- Prediction interface
├── tests/                  <- Unit tests
├── docs/                   <- Documentation website
│   ├── index.md
│   ├── tutorials/          <- User guides and examples
│   └── api/                <- API reference
├── examples/               <- Example scripts
└── data/                   <- Sample datasets
```

## Contributing workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```
2. **Make focused commits** with clear messages.
3. **Run tests and checks** before committing:
   ```bash
   pytest
   ruff check src/
   ```
4. **Update documentation** and `NEWS.md`.
5. **Update examples** if user-facing behavior changes.
6. **Open a pull request** with a short description of your change.
7. **Respond to review feedback** constructively.

## Style guidelines

### Python code

- Follow PEP 8 conventions.
- Use type hints for function signatures.
- Use descriptive names and consistent indentation.
- Prefer vectorized operations (NumPy/PyTorch) over loops.
- Validate inputs early with clear error messages.
- Document all public functions with docstrings.

### Tests

- Add or update tests when functionality changes.
- Keep tests minimal and reproducible.
- Avoid external dependencies unless essential.
- Use pytest fixtures for shared setup.

## Pull request checklist

- [ ] Tests pass (`pytest`)
- [ ] Code passes linting (`ruff check src/`)
- [ ] Documentation updated (`NEWS.md`)
- [ ] Examples updated if needed
- [ ] No unrelated formatting changes
- [ ] PR description clearly explains the change

## Reporting bugs

When reporting an issue, please include:
- A minimal reproducible example
- Output of `pip show resolve` and `python --version`
- Expected vs. actual results
- Operating system and PyTorch version
- GPU info if relevant (`torch.cuda.get_device_name()`)

---

By contributing to RESOLVE, you agree that your code is released under the same license as the package.
