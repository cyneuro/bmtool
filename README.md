# bmtool

<div align="center">

**A comprehensive toolkit for developing computational neuroscience models with NEURON and BMTK**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cyneuro/bmtool/blob/master/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/bmtool.svg)](https://badge.fury.io/py/bmtool)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://cyneuro.github.io/bmtool/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

[Documentation](https://cyneuro.github.io/bmtool/) | [Installation](#installation) | [Features](#features) | [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

BMTool is a collection of utilities designed to streamline the development, analysis, and execution of large-scale neural network models using [NEURON](https://www.neuron.yale.edu/neuron/) and the [Brain Modeling Toolkit (BMTK)](https://alleninstitute.github.io/bmtk/). Whether you're building single-cell models, developing synaptic mechanisms, or running parameter sweeps on HPC clusters, BMTool provides the tools you need.

## Features

### Single Cell Modeling
- Analyze passive membrane properties
- Current injection protocols and voltage responses
- F-I curve generation and analysis
- Impedance profile calculations

### Synapse Development
- Synaptic property tuning and validation
- Gap junction modeling and analysis
- Visualization of synaptic responses
- Parameter optimization tools

### Network Construction
- Custom connectors for complex network models
- Distance-dependent connection probabilities
- Connection matrix visualization
- Network statistics and validation

### Visualization
- Network position plotting (2D/3D)
- Connection matrices and weight distributions
- Raster plots and spike train analysis
- LFP and ECP visualization
- Power spectral density analysis

### SLURM Cluster Management
- YAML-based simulation configuration
- Automated parameter sweeps (value-based and percentage-based)
- Multi-environment support for different HPC devices
- Job monitoring and status tracking
- Microsoft Teams webhook notifications

### Analysis Tools
- Spike rate and population activity analysis
- Phase locking and spike-phase timing
- Oscillation detection with FOOOF
- Power spectral analysis
- Batch processing capabilities

## Installation

Install the latest stable release from PyPI:

```bash
pip install bmtool
```

For development installation, see the [Contributing Guide](CONTRIBUTING.md).

## Documentation

Comprehensive documentation with examples and tutorials is available at:

**[https://cyneuro.github.io/bmtool/](https://cyneuro.github.io/bmtool/)**

### Key Documentation Sections

- [SLURM Module](https://cyneuro.github.io/bmtool/modules/slurm/) - Run simulations on HPC clusters
- [Analysis Workflows](https://cyneuro.github.io/bmtool/modules/analysis/) - Process simulation results
- [Network Building](https://cyneuro.github.io/bmtool/modules/connectors/) - Construct neural networks
- [Single Cell Tools](https://cyneuro.github.io/bmtool/modules/singlecell/) - Analyze individual neurons
- [API Reference](https://cyneuro.github.io/bmtool/api/) - Complete API documentation

## Contributing

We welcome contributions from the community! To get started:

1. Read the [Contributing Guide](CONTRIBUTING.md) for setup instructions
2. Check out open [issues](https://github.com/cyneuro/bmtool/issues) or propose new features
3. Follow our code style guidelines using Ruff and pre-commit hooks

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed information on development setup, code standards, and the pull request process.

## Requirements

- Python 3.8+
- NEURON 8.2.4
- BMTK
- See [setup.py](setup.py) for complete dependency list

## License

BMTool is released under the [MIT License](LICENSE).

## Support

For questions, bug reports, or feature requests:

- 📖 Check the [documentation](https://cyneuro.github.io/bmtool/)
- 🐛 Open an [issue](https://github.com/cyneuro/bmtool/issues)
- 💬 Contact: gregglickert@mail.missouri.edu

## Acknowledgments

Developed by the Neural Engineering Laboratory at the University of Missouri.
