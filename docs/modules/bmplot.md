# BMPlot Module

The BMPlot module provides visualization tools for BMTK networks, allowing you to analyze and plot connectivity patterns, cell positions, and network properties.

## Features

- **Connection Matrices**: Generate matrices showing connectivity between populations
- **Position Plots**: Visualize 3D positions of cells in the network
- **Rotation Plots**: Visualize cell orientation in 3D space
- **Connection Analysis**: Analyze connection properties and distributions
- **Raster Plots**: Visualize spike data from simulations

## Connection Matrices

### Total Connection Matrix

Generate a table showing the total number of connections between neuron populations:

```python
import bmtool.bmplot as bmplot

# Default - all connections
bmplot.total_connection_matrix(
    config='config.json',
    title='Total Connection Matrix',
    sources=None,  # Use all sources (default)
    targets=None,  # Use all targets (default)
    save_file=None  # Optional path to save the figure
)

# Specific source/target populations
bmplot.total_connection_matrix(
    config='config.json', 
    sources='LA',
    targets='LA'
)
```

### Percent Connection Matrix

Generate a matrix showing the percent connectivity between neuron populations:

```python
# Default - all connections
bmplot.percent_connection_matrix(
    config='config.json',
    method='total'  # Default method
)

# Only unidirectional connections
bmplot.percent_connection_matrix(
    config='config.json', 
    method='unidirectional'
)

# Only bidirectional connections
bmplot.percent_connection_matrix(
    config='config.json', 
    method='bidirectional'
)
```

### Convergence Connection Matrix

Generate a matrix showing the convergence of connections between neuron populations:

```python
# Mean convergence (default)
bmplot.convergence_connection_matrix(
    config='config.json',
    method='mean+std'  # Default method
)

# Maximum convergence
bmplot.convergence_connection_matrix(
    config='config.json', 
    method='max'
)

# Minimum convergence
bmplot.convergence_connection_matrix(
    config='config.json', 
    method='min'
)

# Standard deviation of convergence
bmplot.convergence_connection_matrix(
    config='config.json', 
    method='std'
)
```

### Divergence Connection Matrix

Generate a matrix showing the divergence of connections between neuron populations:

```python
# Mean divergence (default)
bmplot.divergence_connection_matrix(
    config='config.json',
    method='mean+std'  # Default method
)

# Maximum divergence
bmplot.divergence_connection_matrix(
    config='config.json', 
    method='max'
)

# Minimum divergence
bmplot.divergence_connection_matrix(
    config='config.json', 
    method='min'
)

# Standard deviation of divergence
bmplot.divergence_connection_matrix(
    config='config.json', 
    method='std'
)
```

### Gap Junction Matrix

Generate a matrix specifically for gap junctions:

```python
# Convergence analysis (default)
bmplot.gap_junction_matrix(
    config='config.json', 
    method='convergence'
)

# Percent connections
bmplot.gap_junction_matrix(
    config='config.json', 
    method='percent'
)
```

### Connector Percent Matrix

Generate a percentage connectivity matrix from a CSV file produced by BMTool connectors:

```python
bmplot.connector_percent_matrix(
    csv_path='connections.csv',
    title='Percent Connection Matrix',
    exclude_strings=None  # Optional strings to exclude
)
```

## Spatial Analysis

### Connection Distance

Generate a 3D plot with source and target cell locations and connection distance analysis:

```python
bmplot.connection_distance(
    config='config.json',
    sources='PopA',
    targets='PopB',
    title='Connection Distance Analysis',
    save_file=None,  # Optional file to save plot
    num_bins=50      # Number of histogram bins
)
```

### Connection Histogram

Generate a histogram showing the distribution of connections:

```python
bmplot.connection_histogram(
    config='config.json',
    sources='PopA',
    targets='PopB',
    title='Connection Distribution',
    num_bins=50,
    xmax=None,  # Optional maximum x value
    ymax=None   # Optional maximum y value
)
```

## 3D Visualization

### Plot 3D Positions

Generate a plot of cell positions in 3D space:

```python
bmplot.plot_3d_positions(
    config='config.json',
    sources=['PopA', 'PopB'],
    title='3D Cell Positions',
    save_file=None,  # Optional file to save plot
    subset=None      # Optional subset of cells to plot
)
```

### Plot 3D Cell Rotation

Generate a plot showing cell locations and orientation in 3D space:

```python
bmplot.plot_3d_cell_rotation(
    config='config.json',
    sources=['PopA'],
    title='3D Cell Orientation',
    quiver_length=20,  # Length of orientation arrows
    arrow_length_ratio=0.3,
    subset=100  # Plot only a subset of cells for clarity
)
```

## Network Visualization

### Plot Network Graph

Plot a network connection diagram:

```python
bmplot.plot_network_graph(
    config='config.json',
    sources='LA',
    targets='LA', 
    sids='pop_name',
    tids='pop_name',
    no_prepend_pop=True,
    title='Network Graph',
    edge_property='model_template'
)
```

## Spike Data Visualization

### Raster Plot

Create a raster plot from spike data:

```python
import pandas as pd
import matplotlib.pyplot as plt
from bmtool.bmplot import raster

# Load spike data
spikes_df = pd.read_csv('spikes.csv')

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Generate raster plot
raster(
    spikes_df=spikes_df,
    groupby='pop_name',
    time_range=(0, 2000),
    ax=ax,
    markersize=2,
    marker='o'
)

plt.show()
```

### Firing Rate Plots

Plot population firing rate statistics:

```python
import pandas as pd
import matplotlib.pyplot as plt
from bmtool.bmplot import plot_firing_rate_pop_stats, plot_firing_rate_distribution

# Assuming you already have firing rate statistics dataframes
# from bmtool.analysis.spikes.compute_firing_rate_stats()

# Plot population firing rate statistics
fig, ax = plt.subplots(figsize=(10, 6))
plot_firing_rate_pop_stats(
    firing_stats=pop_stats,
    groupby='pop_name',
    ax=ax,
    sort_by_mean=True,
    bar_width=0.7
)
plt.show()

# Plot firing rate distributions
fig, ax = plt.subplots(figsize=(10, 6))
plot_firing_rate_distribution(
    individual_stats=individual_stats,
    groupby='pop_name',
    ax=ax,
    log_scale=False,
    num_bins=30
)
plt.show()
```

## Report Visualization

Plot data from simulation reports:

```python
bmplot.plot_report(
    config_file='config.json',
    report_file='report.h5',
    report_name='membrane_potential',
    variables=['v'],
    gids=[1, 2, 3, 4, 5]
)
``` 