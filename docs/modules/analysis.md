# Analysis Module

The Analysis module provides tools for processing and analyzing simulation results from BMTK models, including spike data and other output reports.

## Features

- **Spike Data Processing**: Load, filter, and analyze spike data
- **Population Analysis**: Calculate population statistics and metrics
- **Visualization**: Create raster plots, histograms, and other visualizations
- **Report Processing**: Analyze membrane potential and other continuous reports

## Spike Analysis

Load and analyze spike data from simulation output:

```python
from bmtool.analysis import Spikes

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

# Calculate interspike intervals for specific nodes
isis = spikes.isi(node_ids=[1, 2, 3, 4, 5])
print(f"Mean ISI: {isis.mean()}")
```

## Raster Plots

Create raster plots to visualize spike patterns:

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

## Population Statistics

Calculate and visualize population-level statistics:

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

# Calculate cross-correlations between populations
xcorr = spikes.cross_correlation(
    pop1='excitatory',
    pop2='inhibitory',
    bin_size=5,
    window_size=100
)

# Plot cross-correlation
plt.figure(figsize=(8, 4))
plt.plot(xcorr.lags, xcorr.correlation)
plt.xlabel('Lag (ms)')
plt.ylabel('Correlation')
plt.title('Cross-correlation between excitatory and inhibitory populations')
plt.axvline(x=0, color='gray', linestyle='--')
plt.show()
```

## Continuous Data Analysis

Analyze continuous data (e.g., membrane potential) from reports:

```python
from bmtool.analysis import Reports
import matplotlib.pyplot as plt

# Load report data
report = Reports(data_file='output/v_report.h5')

# Get data for specific nodes
v_data = report.get_data(node_ids=[1, 2, 3], time_window=(500, 1000))

# Plot membrane potential traces
plt.figure(figsize=(10, 6))
for node_id, data in v_data.items():
    plt.plot(data.times, data.values, label=f"Node {node_id}")
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.title('Membrane Potential Traces')
plt.legend()
plt.show()

# Calculate and plot average membrane potential across a population
v_mean = report.population_mean(population='excitatory')
plt.figure(figsize=(10, 4))
plt.plot(v_mean.times, v_mean.values)
plt.xlabel('Time (ms)')
plt.ylabel('Mean Membrane Potential (mV)')
plt.title('Mean Membrane Potential of Excitatory Population')
plt.show()
```

## Advanced Analysis

Perform more complex analyses like oscillation detection and synchrony measures:

```python
from bmtool.analysis import Spikes, oscillation_power

# Load spike data
spikes = Spikes(data_file='output/spikes.h5')

# Calculate population rate for spectral analysis
population_rate = spikes.population_rate(bin_size=1)

# Calculate power spectrum
freqs, power = oscillation_power(
    population_rate.rates,
    fs=1000/population_rate.bin_size,
    nperseg=1024
)

# Plot power spectrum
plt.figure(figsize=(10, 6))
plt.semilogy(freqs, power)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum of Network Activity')
plt.xlim(0, 100)
plt.grid(True)
plt.show()

# Calculate synchrony index
sync_index = spikes.synchrony_index(bin_size=5, time_window=(500, 1500))
print(f"Synchrony Index: {sync_index}")
``` 