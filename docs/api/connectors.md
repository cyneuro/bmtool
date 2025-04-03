# Connectors API Reference

This page provides API reference documentation for the Connectors module.

<!-- These sections will be uncommented once docstrings are added to the code
::: bmtool.connectors

## UnidirectionConnector

::: bmtool.connectors.UnidirectionConnector

## ReciprocalConnector

::: bmtool.connectors.ReciprocalConnector

## CorrelatedGapJunction

::: bmtool.connectors.CorrelatedGapJunction

## OneToOneSequentialConnector

::: bmtool.connectors.OneToOneSequentialConnector
-->

The Connectors module provides helper functions and classes to work with BMTK's NetworkBuilder for building complex network connectivity patterns.

## Key Components

### UnidirectionConnector

The `UnidirectionConnector` class builds unidirectional connections:

- Create connections with specified probability
- Set number of synapses per connection
- Apply to connections within or between populations

### ReciprocalConnector

The `ReciprocalConnector` class creates reciprocal connections:

- Set base and reciprocal connection probabilities
- Control synapse counts for each direction
- Estimate reciprocal connectivity parameters from target statistics

### CorrelatedGapJunction

The `CorrelatedGapJunction` class creates gap junction connections:

- Create gap junctions correlated with chemical synapses
- Control connection probabilities based on chemical synapse status
- Different probabilities for non-connected, unidirectionally connected, and reciprocally connected pairs

### OneToOneSequentialConnector

The `OneToOneSequentialConnector` class creates one-to-one mappings:

- Connect elements sequentially between populations
- Useful for input-to-network connections
- Handle populations with different sizes 