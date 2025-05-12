# bmtool
A collection of modules to make developing [Neuron](https://www.neuron.yale.edu/neuron/) and [BMTK](https://alleninstitute.github.io/bmtk/) models easier.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/cyneuro/bmtool/blob/master/LICENSE)

## In depth documentation and examples can be found [here](https://cyneuro.github.io/bmtool/)

## Getting Started

**Installation**
```bash
pip install bmtool
```
For developers who will be pulling down additional updates to this repository regularly use the following instead.
```bash
git clone https://github.com/cyneuro/bmtool.git
cd bmtool
python setup.py develop
```
Then download updates (from this directory) with
```bash
git pull
```

BMTool provides several modules to simplify the development of computational neuroscience models with NEURON and the Brain Modeling Toolkit (BMTK). It offers functionality for:

- **Single Cell Modeling**: Analyze passive properties, current injection, FI curves, and impedance profiles
- **Synapse Development**: Tools for tuning synaptic properties and gap junctions
- **Network Construction**: Connectors for building complex network structures
- **Visualization**: Plot connection matrices, network positions, and more
- **Simulation Management**: Run simulations on SLURM clusters with parameter sweeps
- **Analysis**: Process simulation results efficiently

## Development

### Code Quality Tools

BMTool now uses modern Python code quality tools to maintain high standards:

- **Ruff**: A fast Python linter for code style and error checking
- **Pyright**: Static type checker to catch type-related bugs early

To install development dependencies in your conda environment:

```bash
# Activate your conda environment
conda activate bmtk

# Install development dependencies
pip install -e ".[dev]"
```

To run the tools:

```bash
# Run Ruff linter
ruff check bmtool/

# Run Ruff with auto-fix
ruff check --fix bmtool/

# Run Pyright type checker
pyright
```
