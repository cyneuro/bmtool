# Analysis API Reference

This page provides API reference documentation for the Analysis module.

The Analysis module provides tools for processing and analyzing simulation results from BMTK models, including spike data and LFP/ECP data.

## Spikes

The `spikes` module provides functions for loading and analyzing spike data from simulations.

::: bmtool.analysis.spikes.load_spikes_to_df
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.spikes.compute_firing_rate_stats
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.spikes._pop_spike_rate
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.spikes.get_population_spike_rate
    options:
      show_root_heading: true
      heading_level: 3

## LFP/ECP Analysis

The `lfp` module provides tools for analyzing local field potentials (LFP) and extracellular potentials (ECP).

::: bmtool.analysis.lfp.load_ecp_to_xarray
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.ecp_to_lfp
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.slice_time_series
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.fit_fooof
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.generate_resd_from_fooof
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.calculate_SNR
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.wavelet_filter
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.butter_bandpass_filter
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.calculate_signal_signal_plv
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.calculate_spike_lfp_plv
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.calculate_ppc
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.calculate_ppc2
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.cwt_spectrogram
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.cwt_spectrogram_xarray
    options:
      show_root_heading: true
      heading_level: 3

::: bmtool.analysis.lfp.plot_spectrogram
    options:
      show_root_heading: true
      heading_level: 3 