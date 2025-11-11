# Entrainment Analysis

This module provides tools for analyzing the relationship between spike timing and LFP oscillations, including phase-locking and spike-field coherence metrics.

## Phase-Locking Analysis

Calculate phase-locking between spikes and LFP:

```python
import numpy as np
from bmtool.analysis.entrainment import (
    calculate_spike_lfp_plv,
    calculate_signal_signal_plv,
    calculate_ppc,
    calculate_ppc2
)
from bmtool.analysis.spikes import load_spikes_to_df
from bmtool.analysis.lfp import load_ecp_to_xarray, butter_bandpass_filter

# Load data
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

# Get spike times for a specific population
population_spikes = spikes_df[spikes_df['pop_name'] == 'Pyramidal']
spike_times = population_spikes['timestamps'].values

# Calculate phase-locking value (PLV)
plv = calculate_spike_lfp_plv(
    spike_times=spike_times,
    lfp_data=filtered_lfp,
    spike_fs=1000,  # Spike times in milliseconds
    lfp_fs=fs,
    filter_method='butter',
    lowcut=4,
    highcut=8
)
print(f"Phase-locking value: {plv}")

# Calculate pairwise phase consistency (PPC)
ppc = calculate_ppc(
    spike_times=spike_times,
    lfp_data=filtered_lfp,
    spike_fs=1000,
    lfp_fs=fs,
    filter_method='butter',
    lowcut=4,
    highcut=8
)
print(f"Pairwise phase consistency: {ppc}")

# Calculate PPC2 (alternate method)
ppc2 = calculate_ppc2(
    spike_times=spike_times,
    lfp_data=filtered_lfp,
    spike_fs=1000,
    lfp_fs=fs,
    filter_method='butter',
    lowcut=4,
    highcut=8
)
print(f"PPC2 value: {ppc2}")
```

## Population Entrainment Analysis

Analyze entrainment across multiple cells or populations:

```python
from bmtool.analysis.entrainment import calculate_entrainment_per_cell

# Calculate entrainment metrics for all cells
entrainment_dict = calculate_entrainment_per_cell(
    spike_df=spikes_df,
    lfp_data=lfp_signal,
    filter_method='wavelet',
    pop_names=['Pyramidal', 'Basket'],
    entrainment_method='plv',  # or 'ppc', 'ppc2'
    spike_fs=1000,
    lfp_fs=fs,
    freqs=[4, 8, 20, 40, 80]  # Frequencies of interest
)

# Print results for each population
for pop, cell_dict in entrainment_dict.items():
    print(f"\nPopulation: {pop}")
    for cell_id, freq_dict in cell_dict.items():
        print(f"Cell {cell_id}:")
        for freq, value in freq_dict.items():
            print(f"  {freq} Hz: {value:.3f}")
```

## Spike-LFP Power Correlation

Analyze correlation between spike rates and LFP power:

```python
from bmtool.analysis.entrainment import calculate_spike_rate_power_correlation

# Calculate correlations across frequency bands
correlation_results, frequencies = calculate_spike_rate_power_correlation(
    spike_rate=population_rates,
    lfp_data=lfp_signal,
    fs=fs,
    pop_names=['Pyramidal', 'Basket'],
    filter_method='wavelet',
    freq_range=(4, 100),
    freq_step=4
)

# Plot results
for pop in correlation_results:
    corr_values = [correlation_results[pop][f]['correlation'] for f in frequencies]
    plt.plot(frequencies, corr_values, label=pop)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Correlation Coefficient')
plt.legend()
plt.title('Spike Rate-LFP Power Correlation')
plt.show()
```
