# Contributing to BMTool

Thank you for your interest in contributing to BMTool!

## Getting Started

1. **Clone the repository**:
```bash
git clone https://github.com/cyneuro/bmtool.git
cd bmtool
```

2. **Create a development environment**:
```bash
conda create -n bmtool-dev python=3.10
conda activate bmtool-dev
```

3. **Install in development mode with dev dependencies**:
```bash
pip install -e ".[dev]"
```

4. **Set up pre-commit hooks**:
```bash
pre-commit install
```

This will automatically run code formatting and linting checks before each commit.

## Making Contributions

1. Create a new branch for your changes
2. Make your changes and commit them
3. Push to your branch and open a Pull Request
4. Fill out the PR template

Pre-commit hooks will ensure your code meets our style guidelines. If any checks fail, fix the issues and commit again.

## Testing

Unit tests live in `tests/` and run with pytest. They cover helpers that do **not** require NEURON or BMTK (connector math, stimulus generators, and package imports).

Install test dependencies and run the suite:

```bash
pip install -e ".[test]"
pip install numpy pandas scipy h5py
pytest
```

To run the same lightweight install used in CI (skips NEURON/BMTK):

```bash
pip install pytest pytest-cov numpy pandas scipy h5py
pip install -e . --no-deps
pytest tests -m unit
```

When contributing, add tests for new pure-Python helpers when practical. Simulation-level tests that need NEURON are not part of this first suite.

## Questions?

- Open an [issue](https://github.com/cyneuro/bmtool/issues)
- Email: gregglickert@mail.missouri.edu

---

