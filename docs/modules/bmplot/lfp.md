# LFP/ECP Plotting

The `bmplot.lfp` module provides functions for visualizing LFP/ECP data, spectrograms, and power spectra.

## Spectrograms

```python
from bmtool.bmplot.lfp import plot_spectrogram
import matplotlib.pyplot as plt

# Plot spectrogram
fig, ax = plt.subplots(figsize=(10, 6))
plot_spectrogram(
    sxx_xarray=spectrogram,
    log_power=True,
    ax=ax
)
plt.show()
```

## Power Spectra

```python
from bmtool.bmplot.lfp import plot_power_spectrum

# Plot power spectrum with FOOOF fit
fig, ax = plt.subplots(figsize=(10, 6))
plot_power_spectrum(
    freqs=freqs,
    pxx=pxx,
    fooof_model=fooof_model,
    ax=ax
)
plt.show()
```

## LFP Time Series

```python
from bmtool.bmplot.lfp import plot_lfp_timeseries

# Plot LFP time series
fig, ax = plt.subplots(figsize=(10, 6))
plot_lfp_timeseries(
    lfp_data=lfp_data,
    channels=[0, 1, 2],  # Optional: specify channels to plot
    time_range=(0, 1000),  # Optional: specify time range
    ax=ax
)
plt.show()
```
