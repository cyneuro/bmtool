# BMTool

A collection of modules to make developing [Neuron](https://www.neuron.yale.edu/neuron/) and [BMTK](https://alleninstitute.github.io/bmtk/) models easier.

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/cyneuro/bmtool/blob/master/LICENSE) 

## Overview

BMTool provides several modules to simplify the development of computational neuroscience models with NEURON and the Brain Modeling Toolkit (BMTK). It offers functionality for:

- **Single Cell Modeling**: Analyze passive properties, current injection, FI curves, and impedance profiles
- **Synapse Development**: Tools for tuning synaptic properties and gap junctions
- **Network Construction**: Connectors for building complex network structures
- **Visualization**: Plot connection matrices, network positions, and more
- **Simulation Management**: Run simulations on SLURM clusters with parameter sweeps
- **Analysis**: Process simulation results efficiently

## Installation

```bash
# Basic installation
pip install bmtool

# For development installation
git clone https://github.com/cyneuro/bmtool.git
cd bmtool
python setup.py develop
```

## Key Features

BMTool provides multiple modules to assist with different aspects of neural modeling:

- **Single Cell Module**: Analyze and tune biophysical cell models
- **Synapses Module**: Configure and tune synaptic connections
- **Connectors Module**: Build complex network connectivity patterns
- **BMPlot Module**: Visualize network connectivity and simulation results
- **Analysis Module**: Process spike and report data from simulations
- **SLURM Module**: Manage simulation workflows on HPC clusters
- **Graphs Module**: Analyze network properties and connectivity patterns

## Command Line Interface

BMTool provides a CLI for accessing functionality directly from the command line:

```bash
bmtool --help
```

See the [CLI documentation](cli.md) for more details. 