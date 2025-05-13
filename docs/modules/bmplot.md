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

```python
import bmtool.bmplot.connections as connections

# Default - all connections
connections.total_connection_matrix(
    config='config.json',
    title='Total Connection Matrix',
    sources=None,  # Use all sources (default)
    targets=None,  # Use all targets (default)
    save_file=None  # Optional path to save the figure
)

# Specific source/target populations
connections.total_connection_matrix(
    config='config.json',
    sources='LA',
    targets='LA'
)
```

### Percent Connection Matrix

Generate a matrix showing the percent connectivity between neuron populations:

```python
# Default - all connections
connections.percent_connection_matrix(
    config='config.json',
    method='total'  # Default method
)

# Only unidirectional connections
connections.percent_connection_matrix(
    config='config.json',
    method='unidirectional'
)

# Only bidirectional connections
connections.percent_connection_matrix(
    config='config.json',
    method='bidirectional'
)
```

### Convergence Connection Matrix

Generate a matrix showing the convergence of connections between neuron populations:

```python
# Mean convergence (default)
connections.convergence_connection_matrix(
    config='config.json',
    method='mean+std'  # Default method
)

# Maximum convergence
connections.convergence_connection_matrix(
    config='config.json',
    method='max'
)

# Minimum convergence
connections.convergence_connection_matrix(
    config='config.json',
    method='min'
)

# Standard deviation of convergence
connections.convergence_connection_matrix(
    config='config.json',
    method='std'
)
```

### Divergence Connection Matrix

Generate a matrix showing the divergence of connections between neuron populations:

```python
# Mean divergence (default)
connections.divergence_connection_matrix(
    config='config.json',
    method='mean+std'  # Default method
)

# Maximum divergence
connections.divergence_connection_matrix(
    config='config.json',
    method='max'
)

# Minimum divergence
connections.divergence_connection_matrix(
    config='config.json',
    method='min'
)

# Standard deviation of divergence
connections.divergence_connection_matrix(
    config='config.json',
    method='std'
)
```

### Gap Junction Matrix

Generate a matrix specifically for gap junctions:

```python
connections.gap_junction_matrix(config='config.json', method='percent')
```

### Connector Percent Matrix

Generate a percentage connectivity matrix from a CSV file produced by BMTool connectors:

```python
import bmtool.bmplot.connections as connections

connections.connector_percent_matrix(
    csv_path='connections.csv',
    title='Percent Connection Matrix',
    exclude_strings=None  # Optional strings to exclude
)
```

### Connection Distance

Generate a 3D plot with source and target cell locations and connection distance analysis:

```python
import bmtool.bmplot.connections as connections

connections.connection_distance(
    config='config.json',
    sources='PopA',
    targets='PopB',
    source_cell_id=1,  # Node ID of source cell
    target_id_type='PopB',  # Target population to analyze
    ignore_z=False  # Whether to ignore z-axis in distance calculations
)
```

### Connection Histogram

Generate a histogram showing the distribution of connections:

```python
import bmtool.bmplot.connections as connections

connections.connection_histogram(
    config='config.json',
    sources='PopA',
    targets='PopB',
    source_cell='PopA',  # Source cell type
    target_cell='PopB'   # Target cell type
)
```

## 3D Visualization

### 3D Cell Positions

Generate a plot of cell positions in 3D space:

```python
import bmtool.bmplot.connections as connections

connections.plot_3d_positions(
    config='config.json',
    sources=['PopA', 'PopB'],
    title='3D Cell Positions',
    save_file=None  # Optional path to save the figure
)
```

### 3D Cell Orientation

Generate a plot showing cell locations and orientation in 3D space:

```python
import bmtool.bmplot.connections as connections

connections.plot_3d_cell_rotation(
    config='config.json',
    sources=['PopA'],
    title='3D Cell Orientation',
    save_file=None  # Optional path to save the figure
)
```

## Network Visualization

### Network Graph

Plot a network connection diagram:

```python
import bmtool.bmplot.connections as connections

connections.plot_network_graph(
    config='config.json',
    sources='LA',
    targets='LA',
    tids='pop_name',
    sids='pop_name',
    no_prepend_pop=True  # Whether to prepend population name to node labels
)
```

## Spike Analysis

### Raster Plot

Generate a raster plot of spike times:

```python
import pandas as pd
import matplotlib.pyplot as plt
from bmtool.bmplot.spikes import raster

# Load spike data
spikes_df = pd.read_csv('spikes.csv')

# Create raster plot
raster(
    spikes_df=spikes_df,
    config='config.json',  # Optional, to load node population data
    network_name='network',  # Optional, specific network to use
    groupby='pop_name'  # Column to group spikes by
)

plt.show()
```

### Firing Rate Statistics

Plot firing rate statistics for different populations:

```python
import pandas as pd
import matplotlib.pyplot as plt
from bmtool.bmplot.spikes import plot_firing_rate_pop_stats, plot_firing_rate_distribution

# Assuming you already have firing rate statistics dataframes
# from bmtool.analysis.spikes.compute_firing_rate_stats()

# Plot mean firing rates with error bars
plot_firing_rate_pop_stats(
    firing_stats=firing_stats_df,
    groupby='pop_name'
)

# Plot distribution of individual firing rates
plot_firing_rate_distribution(
    individual_stats=individual_stats_df,
    groupby='pop_name',
    plot_type=['box', 'swarm']  # Can use 'box', 'violin', 'swarm' or combinations
)

plt.show()
```

## Entrainment Analysis

### Spike-Power Correlation

Plot the correlation between population spike rates and LFP power:

```python
import bmtool.bmplot.entrainment as entrainment

# Assuming you have correlation results from bmtool.analysis.entrainment
entrainment.plot_spike_power_correlation(
    correlation_results=correlation_results,
    frequencies=frequencies,
    pop_names=pop_names
)
```

## LFP Analysis

### Spectrogram

Plot a spectrogram from LFP data:

```python
import bmtool.bmplot.lfp as lfp

# Assuming you have an xarray dataset with spectrogram data
lfp.plot_spectrogram(
    sxx_xarray=spectrogram_data,
    remove_aperiodic=None,  # Optional aperiodic component to remove
    log_power=True,  # Whether to use log scale for power
    plt_range=[0, 100]  # Frequency range to plot
)
