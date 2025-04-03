# Analysis API Reference

This page provides API reference documentation for the Analysis module.

<!-- These sections will be uncommented once docstrings are added to the code
::: bmtool.analysis

## Spikes

::: bmtool.analysis.Spikes

## Reports

::: bmtool.analysis.Reports

## Analysis Functions

::: bmtool.analysis.oscillation_power
-->

The Analysis module provides tools for processing and analyzing simulation results from BMTK models, including spike data and other output reports.

## Key Components

### Spikes

The `Spikes` class provides functions for loading and analyzing spike data from simulations:

- Load spike data from HDF5 files
- Filter spikes by population, time window, or node IDs
- Calculate basic statistics like firing rates and ISIs
- Create raster plots and other visualizations

### Reports

The `Reports` class allows you to work with continuous data reports:

- Load report data from HDF5 files
- Extract specific time ranges or nodes
- Calculate population statistics
- Visualize membrane potentials and other continuous variables

### Analysis Functions

The module includes various helper functions for advanced analysis:

- `oscillation_power`: Analyze frequency components of network activity
- Synchrony measures and correlation tools 