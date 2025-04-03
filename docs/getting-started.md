# Getting Started with BMTool

## Installation

BMTool can be installed directly from PyPI:

```bash
pip install bmtool
```

### Development Installation

For developers who will be contributing to BMTool or need the latest features:

```bash
git clone https://github.com/cyneuro/bmtool.git
cd bmtool
python setup.py develop
```

Update the repository (from the bmtool directory) with:

```bash
git pull
```

## Prerequisites

BMTool requires:

- Python 3.6 or later
- NEURON 7.7 or later (for cell modeling functionality)
- BMTK (Brain Modeling Toolkit)

Additional dependencies are automatically installed with the package.

## Basic Usage

### Command Line Interface

BMTool provides a command-line interface for easy access to many features:

```bash
# View available commands
bmtool --help

# Access plotting functionality
bmtool plot --help

# Access utility functions
bmtool util --help
```

### Python Module Usage

BMTool can be imported as a Python module to access its functionality:

```python
# Import specific modules
from bmtool.singlecell import Profiler, Passive, CurrentClamp, FI, ZAP
from bmtool.bmplot import total_connection_matrix, plot_3d_positions
from bmtool.connectors import UnidirectionConnector, ReciprocalConnector
```

## Next Steps

- Check out the [module documentation](modules/singlecell.md) for details on specific modules
- Explore the [examples](examples/single-cell.md) to learn how to use BMTool features
- Read the [API reference](api/singlecell.md) for detailed function and class documentation 