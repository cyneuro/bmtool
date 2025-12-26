# Entrainment Plotting

The `bmplot.entrainment` module provides functions for visualizing phase-locking, coherence, and other entrainment metrics.

## Spike Power Correlation

Plot the correlation between population spike rates and LFP power across frequencies:

```python
from bmtool.bmplot.entrainment import plot_spike_power_correlation
import matplotlib.pyplot as plt

# Plot spike rate-LFP power correlation across frequencies
fig = plot_spike_power_correlation(
    spike_df=spike_df,
    lfp_data=lfp,
    fs=400,
    pop_names=['PV', 'SST'],
    freq_range=(10, 100),
    freq_step=5
)
plt.show()
```

## Trial-Averaged Spike Power Correlation

Plot trial-averaged spike rate-LFP power correlation:

```python
from bmtool.bmplot.entrainment import plot_trial_avg_spike_power_correlation

fig = plot_trial_avg_spike_power_correlation(
    spike_df=spike_df,
    lfp_data=lfp,
    fs=400,
    pop_names=['PV', 'SST'],
    time_range=[(0, 1000), (1000, 2000)],  # Trial time ranges
    freq_range=(10, 100),
    freq_step=5
)
plt.show()
```

## Population Entrainment

Plot entrainment (PPC) metrics by population:

```python
from bmtool.bmplot.entrainment import plot_entrainment_by_population

# Plot PPC entrainment metrics for each population
fig = plot_entrainment_by_population(
    ppc_dict=ppc_results,
    pop_names=['PV', 'SST'],
    freqs=[4, 8, 20, 40, 80]
)
plt.show()
```

## Entrainment Swarm Plots

Visualize individual cell entrainment values at a specific frequency:

```python
from bmtool.bmplot.entrainment import plot_entrainment_swarm_plot

# Create swarm plot of entrainment at specific frequency
plot_entrainment_swarm_plot(
    ppc_dict=ppc_results,
    pop_names=['PV', 'SST'],
    freq=40  # Frequency in Hz
)
plt.show()
```

## Phase Histograms with Spike Counts

Plot spike phase distributions across LFP cycles:

```python
from bmtool.bmplot.entrainment import plot_cycle_with_spike_histograms

# Plot spike phase distributions
plot_cycle_with_spike_histograms(
    phase_data=phase_data,
    pop_names=['PV', 'SST'],
    bins=36
)
plt.show()
```

## Trial-Averaged Entrainment

Plot trial-averaged entrainment metrics over time:

```python
from bmtool.bmplot.entrainment import plot_trial_avg_entrainment

fig = plot_trial_avg_entrainment(
    spike_df=spike_df,
    lfp_data=lfp,
    fs=400,
    pop_names=['PV', 'SST'],
    time_range=[(0, 1000), (1000, 2000)],
    freq_of_interest=40
)
plt.show()
```

## Firing Rate Phase-Amplitude Histogram

Plot firing rate as a function of LFP phase and amplitude:

```python
from bmtool.bmplot.entrainment import plot_fr_hist_phase_amplitude

fig = plot_fr_hist_phase_amplitude(
    spike_df=spike_df,
    lfp_data=lfp,
    fs=400,
    pop_names=['PV', 'SST'],
    freq_of_interest=40
)
plt.show()
```

## Trial-Averaged Spike Rate PLV

Plot trial-averaged phase-locking values for spike rates:

```python
from bmtool.bmplot.entrainment import plot_trial_avg_spike_rate_plv

fig = plot_trial_avg_spike_rate_plv(
    spike_df=spike_df,
    lfp_data=lfp,
    fs=400,
    pop_names=['PV', 'SST'],
    time_range=[(0, 1000), (1000, 2000)],
    freq_of_interest=40
)
plt.show()
```
