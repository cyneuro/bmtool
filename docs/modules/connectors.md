# Connectors Module

The Connectors module provides helper functions and classes that work with BMTK's NetworkBuilder module to facilitate building complex network connectivity patterns. It supports creating reciprocal connections, distance-dependent connections, gap junctions, and more.

## Features

- **Unidirectional Connector**: Build connections with given probability between populations
- **Reciprocal Connector**: Build connections with reciprocal probability between populations
- **Correlated Gap Junction**: Create gap junctions correlated with chemical synapses
- **One-to-One Sequential Connector**: Create one-to-one mappings between populations

## Basic Setup

All connector examples use the following network node structure:

```python
from bmtk.builder import NetworkBuilder

# Create main network
net = NetworkBuilder('example_net')
net.add_nodes(N=100, pop_name='PopA', model_type='biophysical')
net.add_nodes(N=100, pop_name='PopB', model_type='biophysical')

# Create background inputs
background = NetworkBuilder('background')
background.add_nodes(N=300, pop_name='tON', potential='exc', model_type='virtual')
```

## Unidirectional Connector

Build unidirectional connections in a BMTK network model with a given probability within a single population or between two populations.

```python
from bmtool.connectors import UnidirectionConnector

# Create connector with 15% connection probability and 1 synapse per connection
connector = UnidirectionConnector(p=0.15, n_syn=1)

# Set up source and target nodes
connector.setup_nodes(source=net.nodes(pop_name='PopA'), target=net.nodes(pop_name='PopB'))

# Add the edges to the network
net.add_edges(**connector.edge_params())
```

## Reciprocal Connector

Build connections with reciprocal probability within a single population or between two populations.

```python
from bmtool.connectors import ReciprocalConnector

# Create connector with 15% base probability and 6.7% reciprocal probability
connector = ReciprocalConnector(p0=0.15, pr=0.06767705087, n_syn0=1, n_syn1=1, estimate_rho=False)

# Setup for recurrent connections within PopA
connector.setup_nodes(source=net.nodes(pop_name='PopA'), target=net.nodes(pop_name='PopA'))

# Add the edges to the network
net.add_edges(**connector.edge_params())
```

## Correlated Gap Junction

Build gap junction connections that can be correlated with chemical synapses.

```python
from bmtool.connectors import ReciprocalConnector, CorrelatedGapJunction

# First create a chemical synapse connectivity pattern
connector = ReciprocalConnector(p0=0.15, pr=0.06, n_syn0=1, n_syn1=1, estimate_rho=False)
connector.setup_nodes(source=net.nodes(pop_name='PopA'), target=net.nodes(pop_name='PopA'))
net.add_edges(**connector.edge_params())

# Then create gap junctions that are correlated with chemical synapses
gap_junc = CorrelatedGapJunction(p_non=0.1228, p_uni=0.56, p_rec=1, connector=connector)
gap_junc.setup_nodes(source=net.nodes(pop_name='PopA'), target=net.nodes(pop_name='PopA'))

# Add gap junction edges
conn = net.add_edges(
    is_gap_junction=True, 
    syn_weight=0.0000495, 
    target_sections=None,
    afferent_section_id=0, 
    afferent_section_pos=0.5,
    **gap_junc.edge_params()
)
```

## One-to-One Sequential Connector

Build one-to-one correspondence connections between two populations.

```python
from bmtool.connectors import OneToOneSequentialConnector

# Create the connector
connector = OneToOneSequentialConnector()

# Connect background to PopA
connector.setup_nodes(source=background.nodes(), target=net.nodes(pop_name='PopA'))
net.add_edges(**connector.edge_params())

# Connect background to PopB
connector.setup_nodes(target=net.nodes(pop_name='PopB'))
net.add_edges(**connector.edge_params())
``` 