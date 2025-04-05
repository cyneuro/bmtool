# BMPlot Examples

This page provides examples of how to use the BMPlot module for visualizing network connectivity and structure.

## Jupyter Notebook Tutorial

For a comprehensive guide with visualizations, check out our Jupyter notebook tutorial:

### Plotting Examples

The [BMPlot Tutorial](notebooks/bmplot/bmplot.ipynb) demonstrates the various plotting functions available in BMTool. In this notebook, you'll learn:

- How to create connection matrices to visualize network connectivity
- How to plot cell positions in 3D space
- How to analyze connection distances and distributions
- How to create network graph visualizations

## Basic API Usage

Here are some basic examples of how to use the BMPlot module in your code:

### Connection Matrices

```python
import bmtool.bmplot as bmplot

# Generate a table showing the total number of connections
bmplot.total_connection_matrix(config='config.json')

# Generate a table showing the percent connectivity
bmplot.percent_connection_matrix(config='config.json')

# Generate a table showing the mean convergence 
bmplot.convergence_connection_matrix(config='config.json')

# Generate a table showing the mean divergence
bmplot.divergence_connection_matrix(config='config.json')

# Generate a matrix specifically for gap junctions
bmplot.gap_junction_matrix(config='config.json', method='percent')
```

### Spatial Analysis

```python
import bmtool.bmplot as bmplot

# Generate a 3D plot with the source and target cells location and connection distance histogram
bmplot.connection_distance(config='config.json', source='PopA', target='PopB')

# Generate a histogram of connection distributions
bmplot.connection_histogram(config='config.json', source='PopA', target='PopB')
```

### 3D Visualization

```python
import bmtool.bmplot as bmplot

# Generate a plot of cell positions in 3D space
bmplot.plot_3d_positions(config='config.json', populations=['PopA', 'PopB'])

# Generate a plot showing cell locations and orientation in 3D
bmplot.plot_3d_cell_rotation(config='config.json', populations=['PopA'])
```

### Network Graph

```python
import bmtool.bmplot as bmplot

# Plot a network connection diagram
bmplot.plot_network_graph(
    config='config.json',
    sources='LA',
    targets='LA',
    tids='pop_name',
    sids='pop_name',
    no_prepend_pop=True
)
```

For more advanced examples and detailed visualizations, please refer to the Jupyter notebook tutorial above. 