# Plot Commands

The `plot` command provides access to visualization features for BMTK network simulations.

```bash
bmtool plot --help
```

## Available Commands

### Raster Plot
Plot the spike raster for a given population.

```bash
# Plot raster from simulation output
bmtool plot raster
```

### Cell Positions
Plot cell positions for a given set of populations.

```bash
# Plot cell positions
bmtool plot positions
```

### Connection Information
Display information related to neuron connections.

```bash
# Plot connection information
bmtool plot connection
```

### Report
Plot the specified report (e.g., membrane potential) using BMTK's default report plotter.

```bash
bmtool plot report
```

## Options

```
Options:
  --config PATH  Configuration file to use, default: "simulation_config.json"
  --no-display   When set there will be no plot displayed, useful for saving
                 plots
  --help         Show this message and exit.
```
