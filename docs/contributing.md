# Contributing to BMTool

Thank you for your interest in contributing to BMTool. This document provides guidelines and instructions for contributing to the project.

## Development Installation

For development, install BMTool in development mode:

```bash
git clone https://github.com/cyneuro/bmtool.git
cd bmtool
python setup.py develop
```

## Package Management

### Uploading to PyPI

To upload a new version to PyPI, follow these steps:

1. Install required tools:

```bash
# Install setuptools and wheel
python -m pip install --user --upgrade setuptools wheel
```

2. Build the distribution packages:

```bash
# Run from setup.py directory
python setup.py sdist bdist_wheel
```

This will generate files in the `dist` directory:
```
dist/
  bmtool-X.Y.Z-py3-none-any.whl
  bmtool-X.Y.Z.tar.gz
```

3. Upload to PyPI:

```bash
# Install Twine
python -m pip install --user --upgrade twine

# Upload to Test PyPI (optional)
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to PyPI
python -m twine upload dist/*
```

### Combined commands

For convenience, here are all the commands together:

```bash
python -m pip install --upgrade setuptools wheel
python setup.py sdist bdist_wheel
python -m pip install --upgrade twine
python -m twine upload dist/*
```

## Code Contributions

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Update documentation if needed
5. Ensure all tests pass
6. Submit a pull request

## Documentation Contributions

To contribute to the documentation:

1. Install MkDocs and required extensions:

```bash
pip install mkdocs mkdocs-material pymdown-extensions mkdocstrings mkdocstrings-python
```

2. Make changes to the Markdown files in the `docs/` directory

3. Preview locally:

```bash
mkdocs serve
```

4. Build the documentation:

```bash
mkdocs build
```

## Testing

Tests for BMTool are a work in progress. When contributing, please ensure your changes don't break existing functionality.

## Code Style

Please follow these guidelines for code style:
- Use 4 spaces for indentation (not tabs)
- Follow PEP 8 style guidelines where possible
- Use meaningful variable and function names
- Add docstrings for functions and classes
