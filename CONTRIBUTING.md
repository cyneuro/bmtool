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
python -m pip install --user --upgrade setuptools wheel
python setup.py sdist bdist_wheel
python -m pip install --user --upgrade twine
python -m twine upload dist/*

```

## Travis Testing
To be implemented.
