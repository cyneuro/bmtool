# Analysis Module

The Analysis module provides tools for processing and analyzing simulation results from BMTK models, including spike data and LFP/ECP data.

## Features

- **Spike Data Processing**: Load, filter, and analyze spike data
- **Population Analysis**: Calculate population statistics and metrics
- **LFP/ECP Analysis**: Process and analyze local field potentials and extracellular potentials
- **Signal Processing**: Filter, transform, and analyze time series data

## Spike Analysis

Load and analyze spike data from simulation output:

```python
import pandas as pd
import matplotlib.pyplot as plt
from bmtool.analysis.spikes import load_spikes_to_df, compute_firing_rate_stats, get_population_spike_rate
from bmtool.bmplot import raster, plot_firing_rate_pop_stats, plot_firing_rate_distribution

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

## Raster Plots

Create raster plots to visualize spike patterns using the BMPlot module:

```python
import matplotlib.pyplot as plt
from bmtool.analysis.spikes import load_spikes_to_df
from bmtool.bmplot import raster

# Load spike data
spikes_df = load_spikes_to_df(
    spike_file='output/spikes.h5',
    network_name='network',
    config='config.json'
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

## LFP/ECP Analysis

Analyze LFP (Local Field Potential) and ECP (Extracellular Potential) data:

```python
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from bmtool.analysis.lfp import (
    load_ecp_to_xarray, 
    ecp_to_lfp, 
    slice_time_series,
    cwt_spectrogram_xarray,
    plot_spectrogram,
    butter_bandpass_filter,
    fit_fooof
)

# Load ECP data
ecp_data = load_ecp_to_xarray('output/ecp.h5', demean=True)

# Convert ECP to LFP with filtering
lfp_data = ecp_to_lfp(
    ecp_data=ecp_data,
    cutoff=250,        # Cutoff frequency in Hz
    fs=10000           # Sampling frequency in Hz
)

# Slice data to specific time range
lfp_slice = slice_time_series(lfp_data, time_ranges=(500, 1500))

# Calculate spectrogram
spectrogram = cwt_spectrogram_xarray(
    x=lfp_slice.sel(channel=0).data,
    fs=10000,
    freq_range=(1, 100),
    nNotes=8
)

# Plot spectrogram
fig, ax = plt.subplots(figsize=(10, 6))
plot_spectrogram(
    sxx_xarray=spectrogram,
    log_power=True,
    ax=ax
)
plt.show()
```

## Frequency Analysis and Phase Locking

Analyze frequency content and phase locking between signals:

```python
import numpy as np
import matplotlib.pyplot as plt
from bmtool.analysis.lfp import (
    butter_bandpass_filter,
    calculate_spike_lfp_plv,
    calculate_signal_signal_plv,
    calculate_ppc
)
from bmtool.analysis.spikes import load_spikes_to_df

# Load spike data and LFP data
spikes_df = load_spikes_to_df('output/spikes.h5', network_name='network')
lfp_data = load_ecp_to_xarray('output/ecp.h5', demean=True)

# Filter LFP to specific frequency band (e.g., theta: 4-8 Hz)
lfp_signal = lfp_data.sel(channel=0).data
fs = 10000  # Hz
filtered_lfp = butter_bandpass_filter(
    data=lfp_signal,
    lowcut=4,
    highcut=8,
    fs=fs
)

# Extract spike times for a specific population
population_spikes = spikes_df[spikes_df['pop_name'] == 'Pyramidal']
spike_times = population_spikes['timestamps'].to_numpy()

# Calculate phase-locking value between spikes and LFP
plv = calculate_spike_lfp_plv(
    spike_times=spike_times,
    lfp_signal=filtered_lfp,
    spike_fs=1000,  # Spike time unit in milliseconds
    lfp_fs=fs,
    fmin=4,
    fmax=8
)
print(f"Phase-locking value: {plv}")

# Calculate pairwise phase consistency
ppc = calculate_ppc(
    spike_times=spike_times,
    lfp_signal=filtered_lfp,
    spike_fs=1000,
    lfp_fs=fs,
    fmin=4, 
    fmax=8
)
print(f"Pairwise phase consistency: {ppc}")
```

## Signal Processing

Apply filters and transformations to time series data:

```python
import numpy as np
import matplotlib.pyplot as plt
from bmtool.analysis.lfp import (
    butter_bandpass_filter,
    wavelet_filter,
    fit_fooof,
    generate_resd_from_fooof,
    calculate_SNR
)

# Apply band-pass filter
filtered_signal = butter_bandpass_filter(
    data=lfp_signal,
    lowcut=30,
    highcut=80,
    fs=10000
)

# Apply wavelet filter centered at a specific frequency
gamma_filtered = wavelet_filter(
    x=lfp_signal,
    freq=40,        # Center frequency in Hz
    fs=10000,       # Sampling rate
    bandwidth=10    # Bandwidth in Hz
)

# Calculate power spectrum and fit FOOOF model
from scipy import signal

# Calculate power spectrum
freqs, pxx = signal.welch(lfp_signal, fs=10000, nperseg=4096)

# Fit FOOOF model to extract oscillatory and aperiodic components
fooof_model = fit_fooof(
    f=freqs,
    pxx=pxx,
    freq_range=[1, 100],
    peak_width_limits=[1, 8],
    max_n_peaks=6
)

# Get the residuals between the original spectrum and the aperiodic fit
resid_spectra, idx_freqs = generate_resd_from_fooof(fooof_model)

# Calculate signal-to-noise ratio in a specific frequency band
snr = calculate_SNR(fooof_model, freq_band=(30, 80))
print(f"Signal-to-noise ratio in gamma band: {snr}")
``` 