# Spike Analysis

This module provides tools for analyzing spike data from BMTK simulations.

## Loading and Processing Spike Data

```python
import pandas as pd
from bmtool.analysis.spikes import load_spikes_to_df, compute_firing_rate_stats

# Load spike data from a simulation
spikes_df = load_spikes_to_df(
    spike_file='output/spikes.h5',
    network_name='network',
    config='config.json',  # Optional: to label cell types
    groupby='pop_name'
)

# Get basic spike statistics by population
pop_stats, individual_stats = compute_firing_rate_stats(
    df=spikes_df,
    groupby='pop_name',
    start_time=500,
    stop_time=1500
)

print("Population firing rate statistics:")
print(pop_stats)
```

## Population Spike Rates

Calculate and visualize population spike rates:

```python
from bmtool.analysis.spikes import get_population_spike_rate
import matplotlib.pyplot as plt

# Calculate population spike rates over time
population_rates = get_population_spike_rate(
    spikes=spikes_df,
    fs=400.0,               # Sampling frequency in Hz
    t_start=0,
    t_stop=2000,
    config='config.json',   # Optional
    network_name='network'  # Optional
)

# Plot population rates
for pop_name, rates in population_rates.items():
    plt.plot(rates, label=pop_name)
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
plt.title('Population Firing Rates')
plt.show()
```

## Visualizing Spike Data

Use BMPlot module for spike data visualization:

```python
from bmtool.bmplot import raster, plot_firing_rate_pop_stats, plot_firing_rate_distribution

# Create a raster plot
fig, ax = plt.subplots(figsize=(10, 6))
raster(
    spikes_df=spikes_df,
    groupby='pop_name',
    time_range=(0, 2000),
    ax=ax
)
plt.show()

# Plot firing rate statistics
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
