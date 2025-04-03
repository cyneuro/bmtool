# SingleCell API Reference

This page provides API reference documentation for the SingleCell module.

<!-- These sections will be uncommented once docstrings are added to the code
::: bmtool.singlecell

## Profiler

::: bmtool.singlecell.Profiler

## Passive

::: bmtool.singlecell.Passive

## CurrentClamp

::: bmtool.singlecell.CurrentClamp

## FI

::: bmtool.singlecell.FI

## ZAP

::: bmtool.singlecell.ZAP

## Helper Functions

::: bmtool.singlecell.run_and_plot
::: bmtool.singlecell.load_allen_database_cells
-->

The SingleCell module provides tools for analyzing and tuning biophysical cell models.

## Key Components

### Profiler

The `Profiler` class initializes the NEURON environment for single cell simulations:

- Load template files and mechanisms
- Configure simulation parameters
- Prepare for single cell analyses

### Passive

The `Passive` class calculates passive membrane properties:

- Determine resting membrane potential
- Calculate input resistance
- Estimate membrane time constant
- Apply exponential fitting methods

### CurrentClamp

The `CurrentClamp` class simulates current injection:

- Apply current steps to neurons
- Count and analyze spikes
- Visualize voltage responses

### FI

The `FI` class generates frequency-current curves:

- Apply increasing current steps
- Record resulting spike counts
- Generate FI curves for cell characterization

### ZAP

The `ZAP` class analyzes frequency response:

- Apply chirp current with increasing frequency
- Calculate impedance profiles
- Identify resonance properties

### Helper Functions

- `run_and_plot`: Convenience function for running simulations and plotting results
- `load_allen_database_cells`: Function to load Allen Institute cell models from SWC and JSON files 