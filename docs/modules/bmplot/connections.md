# Network Connections Plotting

The `bmplot.connections` module provides functions for visualizing network connectivity patterns and connection statistics.

## Connection Matrix

```python
from bmtool.bmplot.connections import plot_connection_matrix
import matplotlib.pyplot as plt

# Plot connection matrix
fig, ax = plt.subplots(figsize=(8, 8))
plot_connection_matrix(
    connection_data=connection_matrix,
    source_pops=source_populations,
    target_pops=target_populations,
    ax=ax
)
plt.show()
```

## Connection Statistics

```python
from bmtool.bmplot.connections import plot_connection_statistics

# Plot connection statistics
fig, ax = plt.subplots(figsize=(10, 6))
plot_connection_statistics(
    stats_df=connection_stats,
    metric='convergence',  # or 'divergence', 'probability'
    ax=ax
)
plt.show()
```

## Weight Distributions

```python
from bmtool.bmplot.connections import plot_weight_distribution

# Plot synaptic weight distributions
fig, ax = plt.subplots(figsize=(10, 6))
plot_weight_distribution(
    weights=synapse_weights,
    by_population=True,
    ax=ax
)
plt.show()
```
