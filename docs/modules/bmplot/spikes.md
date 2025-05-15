# Spike Plotting

The `bmplot.spikes` module provides functions for visualizing spike data and firing rate statistics.

## Raster Plots

```python
from bmtool.bmplot import raster
import matplotlib.pyplot as plt

# Create a raster plot
fig, ax = plt.subplots(figsize=(10, 6))
raster(
    spikes_df=spikes_df,
    groupby='pop_name',
    time_range=(0, 2000),
    ax=ax
)
plt.show()
```

## Firing Rate Statistics

```python
from bmtool.bmplot import plot_firing_rate_pop_stats, plot_firing_rate_distribution

# Plot population firing rate statistics
fig, ax = plt.subplots(figsize=(10, 6))
plot_firing_rate_pop_stats(
    firing_stats=pop_stats,
    groupby='pop_name',
    ax=ax
)
plt.show()

# Plot firing rate distributions
fig, ax = plt.subplots(figsize=(10, 6))
plot_firing_rate_distribution(
    individual_stats=individual_stats,
    groupby='pop_name',
    ax=ax
)
plt.show()
```
