# Synapses API Reference

This page provides API reference documentation for the Synapses module.

<!-- These sections will be uncommented once docstrings are added to the code
::: bmtool.synapses

## SynapticTuner

::: bmtool.synapses.SynapticTuner

## GapJunctionTuner

::: bmtool.synapses.GapJunctionTuner
-->

The Synapses module provides tools for configuring and tuning synaptic connections in NEURON models.

## Key Components

### SynapticTuner

The `SynapticTuner` class provides an interactive interface for tuning chemical synapse parameters:

- Interactive sliders for parameter adjustment
- Real-time visualization of synaptic responses
- Support for various synapse mechanisms
- Parameter export for network models

### GapJunctionTuner

The `GapJunctionTuner` class assists with electrical synapse tuning:

- Interactive adjustment of gap junction properties
- Measurement of coupling coefficients
- Optimization tools for target coupling values
- Visualization of coupled cell behavior 