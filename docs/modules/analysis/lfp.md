# LFP/ECP Analysis

This module provides tools for analyzing Local Field Potentials (LFP) and Extracellular Potentials (ECP) from BMTK simulations.

## Loading and Processing LFP/ECP Data

```python
import numpy as np
import xarray as xr
from bmtool.analysis.lfp import load_ecp_to_xarray, ecp_to_lfp, slice_time_series

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
```

## Spectral Analysis

Analyze frequency content using wavelets and FOOOF:

```python
from bmtool.analysis.lfp import (
    cwt_spectrogram_xarray,
    fit_fooof,
    generate_resd_from_fooof
)
import matplotlib.pyplot as plt

# Calculate wavelet spectrogram
spectrogram = cwt_spectrogram_xarray(
    x=lfp_slice.sel(channel=0).data,
    fs=10000,
    freq_range=(1, 100),
    nNotes=8
)

# Calculate power spectrum and fit FOOOF model
from scipy import signal

# Calculate power spectrum
freqs, pxx = signal.welch(lfp_data.sel(channel=0).data, fs=10000, nperseg=4096)

# Fit FOOOF model
fooof_model = fit_fooof(
    f=freqs,
    pxx=pxx,
    freq_range=[1, 100],
    peak_width_limits=[1, 8],
    max_n_peaks=6
)

# Get residuals between original spectrum and aperiodic fit
resid_spectra, idx_freqs = generate_resd_from_fooof(fooof_model)
```

## Filtering and Signal Processing

Apply various filters to LFP/ECP data:

```python
from bmtool.analysis.lfp import butter_bandpass_filter, wavelet_filter, calculate_SNR

# Band-pass filter
filtered_signal = butter_bandpass_filter(
    data=lfp_data.sel(channel=0).data,
    lowcut=30,
    highcut=80,
    fs=10000
)

# Wavelet filter centered at specific frequency
gamma_filtered = wavelet_filter(
    x=lfp_data.sel(channel=0).data,
    freq=40,        # Center frequency in Hz
    fs=10000,       # Sampling rate
    bandwidth=10    # Bandwidth in Hz
)

# Calculate signal-to-noise ratio
snr = calculate_SNR(fooof_model, freq_band=(30, 80))
print(f"Signal-to-noise ratio in gamma band: {snr}")
```
