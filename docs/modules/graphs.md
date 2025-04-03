# Graphs Module

The Graphs module provides tools for analyzing network connectivity as graph structures using the NetworkX library. It allows you to convert BMTK networks into graph representations for analysis and visualization.

## Features

- **Graph Generation**: Convert BMTK network connectivity to NetworkX graphs
- **Interactive Visualization**: Create interactive plots of network graphs
- **Connectivity Analysis**: Analyze network topology and connection patterns
- **Data Export**: Export connection data for further analysis

## Generate Graph

Convert a BMTK network into a NetworkX graph for analysis:

```python
from bmtool import graphs
import networkx as nx

# Generate a graph from a BMTK model
graph = graphs.generate_graph(config='config.json', source='LA', target='LA')

# Get basic graph statistics
print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())
print("Node labels:", set(nx.get_node_attributes(graph, 'label').values()))
```

## Plot Graph

Create an interactive visualization of a network graph:

```python
from bmtool import graphs

# Generate a graph from a BMTK model
graph = graphs.generate_graph(config='config.json', source='LA', target='LA')

# Create an interactive plot
graphs.plot_graph(graph)
```

This generates an interactive visualization showing nodes, edges, and the number of connections between different cell types.

## Generate Connection Table

Create a CSV file containing detailed connection information for each cell:

```python
from bmtool import graphs
import pandas as pd

# Generate a graph from a BMTK model
graph = graphs.generate_graph(config='config.json', source='LA', target='LA')

# Export connection data to CSV
graphs.export_node_connections_to_csv(graph, 'node_connections.csv')

# Load and view the connection data
df = pd.read_csv('node_connections.csv')
df.head()
```

The resulting CSV file contains information about each cell and the number of connections it receives from different cell types.

## Advanced Graph Analysis

Perform advanced graph analysis using NetworkX functions:

```python
from bmtool import graphs
import networkx as nx
import matplotlib.pyplot as plt

# Generate a graph from a BMTK model
graph = graphs.generate_graph(config='config.json', source='LA', target='LA')

# Calculate node degree distribution
degrees = [d for n, d in graph.degree()]
plt.hist(degrees, bins=20)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Node Degree Distribution')
plt.show()

# Calculate clustering coefficient
clustering = nx.average_clustering(graph)
print(f"Average clustering coefficient: {clustering}")

# Find strongly connected components
components = list(nx.strongly_connected_components(graph))
print(f"Number of strongly connected components: {len(components)}")
```

## Custom Graph Properties

Add custom properties to graph nodes and edges:

```python
from bmtool import graphs

# Generate a graph from a BMTK model
graph = graphs.generate_graph(config='config.json', source='LA', target='LA')

# Add custom node properties
for node in graph.nodes():
    cell_type = graph.nodes[node].get('label', '')
    if cell_type == 'PNc':
        graph.nodes[node]['is_principal'] = True
    else:
        graph.nodes[node]['is_principal'] = False

# Filter nodes by property
principal_cells = [n for n, d in graph.nodes(data=True) if d.get('is_principal', False)]
print(f"Number of principal cells: {len(principal_cells)}")
``` 