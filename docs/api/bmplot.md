# BMPlot API Reference

This page provides API reference documentation for the BMPlot module.

<!-- These sections will be uncommented once docstrings are added to the code
::: bmtool.bmplot

## Connection Matrices

::: bmtool.bmplot.total_connection_matrix
::: bmtool.bmplot.percent_connection_matrix
::: bmtool.bmplot.connector_percent_matrix
::: bmtool.bmplot.convergence_connection_matrix
::: bmtool.bmplot.divergence_connection_matrix
::: bmtool.bmplot.gap_junction_matrix

## Spatial Analysis

::: bmtool.bmplot.connection_distance
::: bmtool.bmplot.connection_histogram

## 3D Visualization

::: bmtool.bmplot.plot_3d_positions
::: bmtool.bmplot.plot_3d_cell_rotation

## Network Graph

::: bmtool.bmplot.plot_network_graph
-->

The BMPlot module provides visualization tools for BMTK networks to analyze and visualize connectivity patterns and cell distributions.

## Key Components

### Connection Matrices

Functions for visualizing network connectivity:

- `total_connection_matrix`: Show total number of connections between populations
- `percent_connection_matrix`: Show percentage of connected pairs between populations
- `connector_percent_matrix`: Connectivity percentages from connector outputs
- `convergence_connection_matrix`: Average number of inputs per cell
- `divergence_connection_matrix`: Average number of outputs per cell
- `gap_junction_matrix`: Specific matrices for gap junction connectivity

### Spatial Analysis

Functions for analyzing spatial aspects of connectivity:

- `connection_distance`: Analyze connections with respect to distance
- `connection_histogram`: Distribution of connections across populations

### 3D Visualization

Functions for visualizing network architecture:

- `plot_3d_positions`: Show cell positions in 3D space
- `plot_3d_cell_rotation`: Show cell orientations in 3D space

### Network Graph

Functions for network visualization:

- `plot_network_graph`: Create interactive network graph visualizations 