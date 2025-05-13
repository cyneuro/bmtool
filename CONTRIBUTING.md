## Package Management

Uploading to PyPI
```python
# Install setuptools and wheel
python -m pip install --user --upgrade setuptools wheel

# Run from setup.py directory
python setup.py sdist bdist_wheel

# Files will be generated in the dist directory
dist/
  example_pkg_your_username-0.0.1-py3-none-any.whl
  example_pkg_your_username-0.0.1.tar.gz

# Install Twine
python -m pip install --user --upgrade twine

# Upload to Test
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to PyPI
python -m twine upload dist/*

# Install from Test
python -m pip install --index-url https://test.pypi.org/simple/ example-pkg-your-username

# Install from PyPI
python -m pip install example-pkg-your-username

```

## All together
```
python -m pip install --upgrade setuptools wheel
python setup.py sdist bdist_wheel
python -m pip install --upgrade twine
python -m twine upload dist/*

```

## Code Quality Tools

BMTool uses pre-commit hooks to maintain code quality and consistency. These hooks automatically check and format your code before each commit.

### Setting up pre-commit

```bash
# Install pre-commit in your environment
pip install pre-commit

# Install the git hooks
pre-commit install
```

After installation, pre-commit will automatically run on `git commit`. The hooks will:

1. Format your code with Ruff
2. Check for common issues and errors
3. Ensure consistent file formatting

If any checks fail, the commit will be aborted, and you'll need to fix the issues before committing again.

### Running pre-commit manually

You can run the pre-commit checks manually on all files:

```bash
pre-commit run --all-files
```

Or on specific files:

```bash
pre-commit run --files path/to/file1.py path/to/file2.py
```

### Ruff and Pyright

BMTool uses two main code quality tools:

1. **Ruff**: A fast Python linter and formatter
2. **Pyright**: A static type checker for Python

To install these tools in your development environment:

```bash
pip install -e ".[dev]"
```

## Travis Testing
To be implemented.
