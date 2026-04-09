# CLI Overview

BMTool provides a command-line interface (CLI) that makes many of its features accessible without writing Python code. 

## Basic Usage

To see all available commands:

```bash
bmtool --help
```

```
Usage: bmtool [OPTIONS] COMMAND [ARGS]...

Options:
  --verbose  Verbose printing
  --help     Show this message and exit.

Commands:
  debug
  plot
  util
```

## Structure

The CLI is organized into three main groups:

- **[plot](plot.md)**: Visualization tools for network simulations (raster plots, connections, positions).
- **[util](cell.md)**: Utility tools for cell tuning, characterization, and building.
- **debug**: Debugging utilities for simulation configurations.
