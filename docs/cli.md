# Command Line Interface

BMTool provides a command-line interface (CLI) that makes many of its features accessible without writing Python code. This page documents the available commands and their usage.

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

## Plot Commands

The `plot` command provides access to visualization features:

```bash
bmtool plot --help
```

```
Usage: bmtool plot [OPTIONS] COMMAND [ARGS]...

Options:
  --config PATH  Configuration file to use, default: "simulation_config.json"
  --no-display   When set there will be no plot displayed, useful for saving
                 plots
  --help         Show this message and exit.

Commands:
  connection  Display information related to neuron connections
  positions   Plot cell positions for a given set of populations
  raster      Plot the spike raster for a given population
  report      Plot the specified report using BMTK's default report plotter
```

### Examples

```bash
# Plot raster from simulation output
bmtool plot raster

# Plot cell positions
bmtool plot positions

# Plot connection information
bmtool plot connection
```

## Utility Commands

The `util` command provides access to various utilities:

```bash
bmtool util --help
```

### Cell Utilities

```bash
bmtool util cell --help
```

#### Cell Tuning

```bash
# For BMTK models with a simulation_config.json file
bmtool util cell tune --builder

# For non-BMTK cell tuning
bmtool util cell --template TemplateFile.hoc --mod-folder ./ tune --builder
```

#### VHalf Segregation

```bash
# Interactive wizard mode
bmtool util cell vhseg

# Command mode
bmtool util cell --template CA3PyramidalCell vhseg --othersec dend[0],dend[1] \
  --infvars inf_im --segvars gbar_im --gleak gl_ichan2CA3 --eleak el_ichan2CA3
```

```bash
# For building simple models
bmtool util cell --hoc cell_template.hoc vhsegbuild --build
bmtool util cell --hoc segmented_template.hoc vhsegbuild
```

## Debug Commands

The `debug` command provides debug utilities:

```bash
bmtool debug --help
``` 