# BMPlot Examples

This page provides examples of how to use the BMPlot module for visualizing network connectivity and structure.

## Jupyter Notebook Tutorial

For a comprehensive guide with visualizations, check out our Jupyter notebook tutorial:

### Plotting Examples

The [BMPlot Tutorial](notebooks/bmplot/bmplot.ipynb) demonstrates the various plotting functions available in BMTool. In this notebook, you'll learn:

- How to create connection matrices to visualize network connectivity
- How to plot cell positions in 3D space
- How to analyze connection distances and distributions

```python
import bmtool.bmplot.connections as connections

# Generate a table showing the total number of connections
connections.total_connection_matrix(config='config.json')

# Generate a table showing the percent connectivity
connections.percent_connection_matrix(config='config.json')

# Generate a table showing the mean convergence
connections.convergence_connection_matrix(config='config.json')

# Generate a table showing the mean divergence
connections.divergence_connection_matrix(config='config.json')

# Generate a matrix specifically for gap junctions
connections.gap_junction_matrix(config='config.json', method='percent')
```

### Spatial Analysis

```python
import bmtool.bmplot.connections as connections

# Generate a 3D plot with the source and target cells location and connection distance histogram
connections.connection_distance(config='config.json', source='PopA', target='PopB')

# Generate a histogram of connection distributions
connections.connection_histogram(config='config.json', source='PopA', target='PopB')
```

### 3D Visualization

```python
import bmtool.bmplot.connections as connections

# Generate a plot of cell positions in 3D space
connections.plot_3d_positions(config='config.json', populations=['PopA', 'PopB'])

# Generate a plot showing cell locations and orientation in 3D
connections.plot_3d_cell_rotation(config='config.json', populations=['PopA'])
```

### Network Graph

```python
import bmtool.bmplot.connections as connections

# Plot a network connection diagram
connections.plot_network_graph(
    config='config.json',
    sources='LA',
    targets='LA',
    tids='pop_name',
    sids='pop_name',
    no_prepend_pop=True
)
```

### Spike Analysis

```python
import bmtool.bmplot.spikes as spikes

# Create a raster plot from spike data
spikes.raster(spikes_df=spikes_df, groupby='pop_name')

# Plot firing rate statistics
spikes.plot_firing_rate_pop_stats(firing_stats=firing_stats_df, groupby='pop_name')
```

### LFP Analysis

```python
import bmtool.bmplot.lfp as lfp

# Plot a spectrogram
lfp.plot_spectrogram(sxx_xarray=spectrogram_data, log_power=True)
```

For more advanced examples and detailed visualizations, please refer to the Jupyter notebook tutorial above.
