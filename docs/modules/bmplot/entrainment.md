# Entrainment Plotting

The `bmplot.entrainment` module provides functions for visualizing phase-locking, coherence, and other entrainment metrics.

## Phase-Locking Plots

```python
from bmtool.bmplot.entrainment import plot_phase_distribution
import matplotlib.pyplot as plt

# Plot spike phase distribution
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
plot_phase_distribution(
    phases=spike_phases,
    population='Pyramidal',
    ax=ax
)
plt.show()
```

## Population Entrainment

```python
from bmtool.bmplot.entrainment import plot_entrainment_by_frequency

# Plot entrainment metrics across frequencies
fig, ax = plt.subplots(figsize=(10, 6))
plot_entrainment_by_frequency(
    entrainment_dict=entrainment_results,
    metric='plv',  # or 'ppc', 'ppc2'
    populations=['Pyramidal', 'Basket'],
    ax=ax
)
plt.show()
```

## Spike-LFP Coherence

```python
from bmtool.bmplot.entrainment import plot_spike_lfp_coherence

# Plot spike-LFP coherence
fig, ax = plt.subplots(figsize=(10, 6))
plot_spike_lfp_coherence(
    coherence_data=coherence_results,
    populations=['Pyramidal', 'Basket'],
    frequencies=freq_range,
    ax=ax
)
plt.show()
```
