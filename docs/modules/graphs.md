# Graphs Module

The Graphs module provides functions for analyzing BMTK network connectivity as graph structures using the NetworkX library. It allows you to convert BMTK networks into graph representations for analysis and visualization.

## Features

- **Graph Generation**: Convert BMTK network connectivity to NetworkX graphs
- **Data Export**: Export connection data for further analysis with other tools

## Generate Graph

Convert a BMTK network into a NetworkX graph for analysis:

```python
import networkx as nx
from bmtool.graphs import generate_graph

# Generate a graph from a BMTK network model
graph = generate_graph(config='config.json', source='LA', target='LA')

# Get basic graph statistics
print("Number of nodes:", graph.number_of_nodes())
print("Number of edges:", graph.number_of_edges())

# Examine node attributes
node_attrs = graph.nodes(data=True)
print("Sample node:", list(node_attrs)[0])

# Examine edge attributes
edge_attrs = graph.edges(data=True)
print("Sample edge:", list(edge_attrs)[0])
```

## Export Node Connections

Export connection data to CSV format for analysis in other tools:

```python
from bmtool.graphs import generate_graph, export_node_connections_to_csv
import pandas as pd

# Generate a graph from a BMTK network model
graph = generate_graph(config='config.json', source='LA', target='LA')

# Export connection data to CSV
export_node_connections_to_csv(graph, 'node_connections.csv')

# Load and view the connection data
df = pd.read_csv('node_connections.csv')
print(df.head())
```

## Advanced Analysis with NetworkX

Use NetworkX's built-in functions for graph analysis:

```python
import networkx as nx
import matplotlib.pyplot as plt
from bmtool.graphs import generate_graph

# Generate a graph from a BMTK network model
graph = generate_graph(config='config.json', source='LA', target='LA')

# Calculate node degree distribution
degrees = [d for n, d in graph.degree()]
plt.figure(figsize=(8, 6))
plt.hist(degrees, bins=20)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.title('Node Degree Distribution')
plt.show()

# Find connected components
if nx.is_directed(graph):
    components = list(nx.weakly_connected_components(graph))
else:
    components = list(nx.connected_components(graph))
print(f"Number of connected components: {len(components)}")
print(f"Size of largest component: {len(max(components, key=len))}")

# Calculate centrality measures
centrality = nx.degree_centrality(graph)
sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
print("Top 5 nodes by degree centrality:")
for node, cent in sorted_centrality[:5]:
    print(f"Node {node}: {cent:.4f}")
```

## Working with NetworkX Attributes

Access and manipulate node and edge attributes:

```python
import networkx as nx
from bmtool.graphs import generate_graph

# Generate a graph from a BMTK network model
graph = generate_graph(config='config.json', source='LA', target='LA')

# Get all unique node labels (e.g., cell types)
node_labels = set(nx.get_node_attributes(graph, 'label').values())
print("Node labels:", node_labels)

# Count nodes by label
label_counts = {}
for node, attrs in graph.nodes(data=True):
    label = attrs.get('label', 'unknown')
    label_counts[label] = label_counts.get(label, 0) + 1
print("Nodes per label:", label_counts)

# Find all edges with a specific property
edge_types = {}
for u, v, attrs in graph.edges(data=True):
    edge_type = attrs.get('edge_type', 'unknown')
    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
print("Edge types:", edge_types)
``` 