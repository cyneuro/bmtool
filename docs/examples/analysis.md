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

## Jupyter Notebook Tutorial

For a comprehensive guide with detailed examples, check out our Jupyter notebook tutorial:

### Using Spikes

The [Using Spikes](notebooks/analysis/using_spikes.ipynb) tutorial demonstrates how to work with spike data from simulations. In this notebook, you'll learn:

- How to load and filter spike data from simulation output
- How to create raster plots and other visualizations
- How to calculate basic statistics like firing rates
- How to analyze population-level activity patterns

## Basic API Usage

Here are some basic examples of how to use the Analysis module in your code:

### Spike Analysis

```python
from bmtool.analysis import Spikes
import matplotlib.pyplot as plt

# Load spike data from a simulation
spikes = Spikes(data_file='output/spikes.h5')

# Get basic spike statistics
total_spikes = spikes.count()
print(f"Total number of spikes: {total_spikes}")

# Get spikes for a specific population
pop_spikes = spikes.filter(population='excitatory')
print(f"Spikes in excitatory population: {pop_spikes.count()}")

# Calculate firing rates
mean_rate = spikes.mean_firing_rate(time_window=(500, 1500))
print(f"Mean firing rate: {mean_rate} Hz")
```

### Raster Plots

```python
from bmtool.analysis import Spikes
import matplotlib.pyplot as plt

# Load spike data
spikes = Spikes(data_file='output/spikes.h5')

# Create a basic raster plot
spikes.raster_plot()
plt.show()

# Create a raster plot with specific populations
spikes.raster_plot(
    populations=['excitatory', 'inhibitory'],
    colors=['blue', 'red'],
    time_window=(500, 1500),
    title='Network Activity',
    xlabel='Time (ms)',
    ylabel='Node ID'
)
plt.show()
```

### Population Statistics

```python
from bmtool.analysis import Spikes
import matplotlib.pyplot as plt

# Load spike data
spikes = Spikes(data_file='output/spikes.h5')

# Calculate population firing rate over time
population_rate = spikes.population_rate(
    bin_size=10,  # 10ms bins
    time_window=(0, 2000)
)

# Plot population rate
plt.figure(figsize=(10, 6))
plt.plot(population_rate.times, population_rate.rates)
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Population Firing Rate')
plt.show()
```

For more advanced examples and detailed usage, please refer to the Jupyter notebook tutorial above. 