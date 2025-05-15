# Analysis API Reference

This page provides an overview of the Analysis module, which contains several submodules for different types of neural data analysis.

The Analysis module provides tools for processing and analyzing simulation results from BMTK models, including:

## [Spike Analysis](analysis/spikes.md)

The `spikes` module provides functions for loading and analyzing spike data from simulations, including:
- Loading spike data into pandas DataFrames
- Computing firing rate statistics
- Calculating population spike rates

## [LFP/ECP Analysis](analysis/lfp.md)

The `lfp` module provides tools for analyzing local field potentials (LFP) and extracellular potentials (ECP), including:
- Loading and processing ECP/LFP data
- Time series analysis and filtering
- Spectral analysis and wavelet transforms
- Signal-to-noise ratio calculations

## [Entrainment Analysis](analysis/entrainment.md)

The `entrainment` module provides tools for analyzing the relationship between spikes and LFP signals, including:
- Phase-locking value (PLV) calculations
- Pairwise phase consistency (PPC) analysis
- Spike-LFP entrainment metrics
- Spike rate and LFP power correlations

## [Network Connectivity Analysis](analysis/netcon_reports.md)

The `netcon_reports` module provides tools for analyzing and reporting network connectivity statistics.
