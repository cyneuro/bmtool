# Analysis Tutorials

The Analysis module provides tools for processing and analyzing simulation results from BMTK models, including spike data and other output reports.

## Features

- Load and analyze spike data from simulations
- Calculate population statistics and metrics
- Analyze LFP/ECP data with spectrograms and phase locking
- Visualize results with various plotting functions

The [Using Spikes](notebooks/analysis/spiking/using_spikes.ipynb) tutorial demonstrates how to work with spike data from simulations. In this notebook, you'll learn:

- How to load spike data from BMTK simulations
- How to calculate firing rate statistics
- How to visualize spike patterns using raster plots
- How to compute population metrics

## Other Tutorials

- [Plot Spectrogram](notebooks/analysis/spectrogram/spectrogram_with_bmtool.ipynb): Learn to create and visualize spectrograms from LFP/ECP data
- [Phase Locking](notebooks/analysis/phase_locking_value/spike_phase_entrainment.ipynb): Analyze the relationship between spike times and oscillatory phase

## Basic API Usage

Here are some basic examples of how to use the Analysis module in your code:

### Spike Analysis

```python
from bmtool.analysis.spikes import load_spikes_to_df, compute_firing_rate_stats
import pandas as pd
import matplotlib.pyplot as plt

# Load spike data from a simulation
spikes_df = load_spikes_to_df(
    spike_file='output/spikes.h5',
    network_name='network',
    config='config.json'  # Optional, for cell type labeling
)

# Get basic spike statistics
pop_stats, individual_stats = compute_firing_rate_stats(
    df=spikes_df,
    groupby='pop_name',
    start_time=500,
    stop_time=1500
)

print("Population firing rate statistics:")
print(pop_stats)
```

### Raster Plots

```python
from bmtool.analysis.spikes import load_spikes_to_df
from bmtool.bmplot import raster
import matplotlib.pyplot as plt

# Load spike data
spikes_df = load_spikes_to_df(
    spike_file='output/spikes.h5',
    network_name='network'
)

# Create a basic raster plot
fig, ax = plt.subplots(figsize=(10, 6))
raster(
    spikes_df=spikes_df,
    groupby='pop_name',
    time_range=(0, 2000),
    ax=ax
)
plt.show()
```

### Population Statistics

```python
from bmtool.analysis.spikes import load_spikes_to_df, get_population_spike_rate
import matplotlib.pyplot as plt

# Load spike data
spikes_df = load_spikes_to_df(
    spike_file='output/spikes.h5',
    network_name='network'
)

# Calculate population firing rates over time
population_rates = get_population_spike_rate(
    spikes=spikes_df,
    fs=400.0,  # Sampling frequency in Hz
    t_start=0,
    t_stop=2000
)

# Plot population rates
for pop_name, rates in population_rates.items():
    plt.plot(rates, label=pop_name)
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Population Firing Rates')
plt.legend()
plt.show()
```

For more advanced examples and detailed usage, please refer to the Jupyter notebook tutorials above.
