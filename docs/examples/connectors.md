# Connectors Examples

This page provides examples of how to use the Connectors module for building complex network connectivity patterns in BMTK models.

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

The `UnidirectionConnector` allows you to build unidirectional connections with a given probability:

```python
from bmtool.connectors import UnidirectionConnector

# Create connector with 15% connection probability and 1 synapse per connection
connector = UnidirectionConnector(p=0.15, n_syn=1)

# Set up source and target nodes
connector.setup_nodes(source=net.nodes(pop_name='PopA'), target=net.nodes(pop_name='PopB'))

# Add the edges to the network
net.add_edges(**connector.edge_params())
```

This creates connections from neurons in population 'PopA' to neurons in population 'PopB' with a 15% probability of connection.

## Reciprocal Connector

The `ReciprocalConnector` enables you to build connections with a specific reciprocal probability:

```python
from bmtool.connectors import ReciprocalConnector

# Create connector with 15% base probability and 6.7% reciprocal probability
connector = ReciprocalConnector(
    p0=0.15,           # Base connection probability 
    pr=0.06767705087,  # Reciprocal connection probability
    n_syn0=1,          # Number of synapses for base connection
    n_syn1=1,          # Number of synapses for reciprocal connection
    estimate_rho=False # Whether to estimate rho value
)

# Setup for recurrent connections within PopA
connector.setup_nodes(source=net.nodes(pop_name='PopA'), target=net.nodes(pop_name='PopA'))

# Add the edges to the network
net.add_edges(**connector.edge_params())
```

This creates connections within population 'PopA' where:
- Any two neurons have a 15% probability of having a unidirectional connection
- If a unidirectional connection exists from neuron A to neuron B, there's a 6.7% probability of also having a connection from B to A

## Correlated Gap Junction

The `CorrelatedGapJunction` connector creates gap junction connections that can be correlated with chemical synapses:

```python
from bmtool.connectors import ReciprocalConnector, CorrelatedGapJunction

# First create a chemical synapse connectivity pattern
connector = ReciprocalConnector(p0=0.15, pr=0.06, n_syn0=1, n_syn1=1, estimate_rho=False)
connector.setup_nodes(source=net.nodes(pop_name='PopA'), target=net.nodes(pop_name='PopA'))
net.add_edges(**connector.edge_params())

# Then create gap junctions that are correlated with chemical synapses
gap_junc = CorrelatedGapJunction(
    p_non=0.1228,  # Probability for pairs with no chemical synapse
    p_uni=0.56,    # Probability for pairs with unidirectional chemical synapse
    p_rec=1,       # Probability for pairs with reciprocal chemical synapses
    connector=connector  # Use the chemical synapse connector for correlation
)
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

This creates gap junctions with probabilities that depend on the existing chemical synapse connections.

## One-to-One Sequential Connector

The `OneToOneSequentialConnector` creates one-to-one mappings between populations:

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

This creates one-to-one connections where each neuron in the background population connects to exactly one neuron in PopA and one in PopB.

## Advanced Usage

### Distance-Dependent Connectivity

You can create distance-dependent connections by providing a connection probability function:

```python
import numpy as np
from bmtool.connectors import UnidirectionConnector

# Define a distance-dependent probability function
def connection_probability(source, target, distance_range=500.0, p_max=0.15):
    """Probability decreases with distance"""
    dist = np.sqrt(np.sum((source['positions'] - target['positions'])**2, axis=1))
    return p_max * np.exp(-dist/distance_range)

# Create connector with the probability function
connector = UnidirectionConnector(p=connection_probability, n_syn=1)
connector.setup_nodes(source=net.nodes(pop_name='PopA'), target=net.nodes(pop_name='PopB'))
net.add_edges(**connector.edge_params())
```

This creates connections where the probability decreases exponentially with distance between neurons. 