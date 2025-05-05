"""
Module for entrainment analysis
"""

import numpy as np
from scipy import signal
import numba
from numba import cuda
import pandas as pd
import xarray as xr
from .lfp import wavelet_filter,butter_bandpass_filter
from typing import Dict, List
from tqdm.notebook import tqdm
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_signal_signal_plv(x1: np.ndarray, x2: np.ndarray, fs: float, freq_of_interest: float = None, 
                  method: str = 'wavelet', lowcut: float = None, highcut: float = None, 
                  bandwidth: float = 2.0) -> np.ndarray:
    """
    Calculate Phase Locking Value (PLV) between two signals using wavelet or Hilbert method.
    
    Parameters:
    - x1, x2: Input signals (1D arrays, same length)
    - fs: Sampling frequency
    - freq_of_interest: Desired frequency for wavelet PLV calculation
    - method: 'wavelet' or 'hilbert' to choose the PLV calculation method
    - lowcut, highcut: Cutoff frequencies for the Hilbert method
    - bandwidth: Bandwidth parameter for the wavelet
    
    Returns:
    - plv: Phase Locking Value (1D array)
    """
    if len(x1) != len(x2):
        raise ValueError("Input signals must have the same length.")
    
    if method == 'wavelet':
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")
        
        # Apply CWT to both signals
        theta1 = wavelet_filter(x=x1, freq=freq_of_interest, fs=fs, bandwidth=bandwidth)
        theta2 = wavelet_filter(x=x2, freq=freq_of_interest, fs=fs, bandwidth=bandwidth)
    
    elif method == 'hilbert':
        if lowcut is None or highcut is None:
            print("Lowcut and or highcut were not definded, signal will not be filter and just take hilbert transform for plv calc")
        
        if lowcut and highcut:
            # Bandpass filter and get the analytic signal using the Hilbert transform
            x1 = butter_bandpass_filter(x1, lowcut, highcut, fs)
            x2 = butter_bandpass_filter(x2, lowcut, highcut, fs)
        
        # Get phase using the Hilbert transform
        theta1 = signal.hilbert(x1)
        theta2 = signal.hilbert(x2)
    
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'hilbert'.")
    
    # Calculate phase difference
    phase_diff = np.angle(theta1) - np.angle(theta2)
    
    # Calculate PLV from standard equation from Measuring phase synchrony in brain signals(1999)
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))
    
    return plv


def calculate_spike_lfp_plv(spike_times: np.ndarray = None, lfp_signal: np.ndarray = None, spike_fs : float = None,
                   lfp_fs: float = None, method: str = 'hilbert', freq_of_interest: float = None,
                   lowcut: float = None, highcut: float = None,
                   bandwidth: float = 2.0) -> tuple:
    """
    Calculate spike-lfp phase locking value Based on https://www.sciencedirect.com/science/article/pii/S1053811910000959
    
    Parameters:
    - spike_times: Array of spike times 
    - lfp_signal: Local field potential time series
    - spike_fs: Sampling frequency in Hz of the spike times only needed if spikes times and lfp has different fs
    - lfp_fs : Sampling frequency in Hz of the LFP
    - method: 'wavelet' or 'hilbert' to choose the phase extraction method
    - freq_of_interest: Desired frequency for wavelet phase extraction
    - lowcut, highcut: Cutoff frequencies for bandpass filtering (Hilbert method)
    - bandwidth: Bandwidth parameter for the wavelet
    
    Returns:
    - ppc1: Phase-Phase Coupling value
    - spike_phases: Phases at spike times
    """
    
    if spike_fs == None:
        spike_fs = lfp_fs
    # Convert spike times to sample indices
    spike_times_seconds = spike_times / spike_fs

    # Then convert from seconds to samples at the new sampling rate
    spike_indices = np.round(spike_times_seconds * lfp_fs).astype(int)
    
    # Filter indices to ensure they're within bounds of the LFP signal
    valid_indices = [idx for idx in spike_indices if 0 <= idx < len(lfp_signal)]
    if len(valid_indices) <= 1:
        return 0, np.array([])
    
    # Extract phase using the specified method
    if method == 'wavelet':
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")
        
        # Apply CWT to extract phase at the frequency of interest
        lfp_complex = wavelet_filter(x=lfp_signal, freq=freq_of_interest, fs=lfp_fs, bandwidth=bandwidth)
        instantaneous_phase = np.angle(lfp_complex)
        
    elif method == 'hilbert':
        if lowcut is None or highcut is None:
            print("Lowcut and/or highcut were not defined, signal will not be filtered and will just take Hilbert transform for PPC1 calculation")
            filtered_lfp = lfp_signal
        else:
            # Bandpass filter the signal
            filtered_lfp = butter_bandpass_filter(lfp_signal, lowcut, highcut, lfp_fs)
        
        # Get phase using the Hilbert transform
        analytic_signal = signal.hilbert(filtered_lfp)
        instantaneous_phase = np.angle(analytic_signal)
        
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'hilbert'.")
    
    # Get phases at spike times
    spike_phases = instantaneous_phase[valid_indices]
    
    # Calculate PPC1
    n = len(spike_phases)
    
    # Convert phases to unit vectors in the complex plane
    unit_vectors = np.exp(1j * spike_phases)
    
    # Calculate the resultant vector
    resultant_vector = np.sum(unit_vectors)
    
    # Plv is the squared length of the resultant vector divided by n²
    plv = (np.abs(resultant_vector) ** 2) / (n ** 2)
    
    return plv


@numba.njit(parallel=True, fastmath=True)
def _ppc_parallel_numba(spike_phases):
    """Numba-optimized parallel PPC calculation"""
    n = len(spike_phases)
    sum_cos = 0.0
    for i in numba.prange(n):
        phase_i = spike_phases[i]
        for j in range(i + 1, n):
            sum_cos += np.cos(phase_i - spike_phases[j])
    return (2 / (n * (n - 1))) * sum_cos


@cuda.jit(fastmath=True)
def _ppc_cuda_kernel(spike_phases, out):
    i = cuda.grid(1)
    if i < len(spike_phases):
        local_sum = 0.0
        for j in range(i+1, len(spike_phases)):
            local_sum += np.cos(spike_phases[i] - spike_phases[j])
        out[i] = local_sum


def _ppc_gpu(spike_phases):
    """GPU-accelerated implementation"""
    d_phases = cuda.to_device(spike_phases)
    d_out = cuda.device_array(len(spike_phases), dtype=np.float64)
    
    threads = 256
    blocks = (len(spike_phases) + threads - 1) // threads
    
    _ppc_cuda_kernel[blocks, threads](d_phases, d_out)
    total = d_out.copy_to_host().sum()
    return (2/(len(spike_phases)*(len(spike_phases)-1))) * total


def calculate_ppc(spike_times: np.ndarray = None, lfp_signal: np.ndarray = None, spike_fs: float = None,
                  lfp_fs: float = None, method: str = 'hilbert', freq_of_interest: float = None,
                  lowcut: float = None, highcut: float = None,
                  bandwidth: float = 2.0,ppc_method: str = 'numpy') -> tuple:
    """
    Calculate Pairwise Phase Consistency (PPC) between spike times and LFP signal.
    Based on https://www.sciencedirect.com/science/article/pii/S1053811910000959
    
    Parameters:
    - spike_times: Array of spike times 
    - lfp_signal: Local field potential time series
    - spike_fs: Sampling frequency in Hz of the spike times only needed if spikes times and lfp has different fs
    - lfp_fs: Sampling frequency in Hz of the LFP
    - method: 'wavelet' or 'hilbert' to choose the phase extraction method
    - freq_of_interest: Desired frequency for wavelet phase extraction
    - lowcut, highcut: Cutoff frequencies for bandpass filtering (Hilbert method)
    - bandwidth: Bandwidth parameter for the wavelet
    - ppc_method: which algo to use for PPC calculate can be numpy, numba or gpu
    
    Returns:
    - ppc: Pairwise Phase Consistency value
    """
    if spike_fs is None:
        spike_fs = lfp_fs
    # Convert spike times to sample indices
    spike_times_seconds = spike_times / spike_fs

    # Then convert from seconds to samples at the new sampling rate
    spike_indices = np.round(spike_times_seconds * lfp_fs).astype(int)
    
    # Filter indices to ensure they're within bounds of the LFP signal
    valid_indices = [idx for idx in spike_indices if 0 <= idx < len(lfp_signal)]
    if len(valid_indices) <= 1:
        return 0, np.array([])
    
    # Extract phase using the specified method
    if method == 'wavelet':
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")
        
        # Apply CWT to extract phase at the frequency of interest
        lfp_complex = wavelet_filter(x=lfp_signal, freq=freq_of_interest, fs=lfp_fs, bandwidth=bandwidth)
        instantaneous_phase = np.angle(lfp_complex)
        
    elif method == 'hilbert':
        if lowcut is None or highcut is None:
            print("Lowcut and/or highcut were not defined, signal will not be filtered and will just take Hilbert transform for PPC calculation")
            filtered_lfp = lfp_signal
        else:
            # Bandpass filter the signal
            filtered_lfp = butter_bandpass_filter(lfp_signal, lowcut, highcut, lfp_fs)
        
        # Get phase using the Hilbert transform
        analytic_signal = signal.hilbert(filtered_lfp)
        instantaneous_phase = np.angle(analytic_signal)
        
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'hilbert'.")
    
    # Get phases at spike times
    spike_phases = instantaneous_phase[valid_indices]
    
    n_spikes = len(spike_phases)

    # Calculate PPC (Pairwise Phase Consistency)
    if n_spikes <= 1:
        return 0, spike_phases
    
    # Explicit calculation of pairwise phase consistency
    sum_cos_diff = 0.0
    
    # # Σᵢ Σⱼ₍ᵢ₊₁₎ f(θᵢ, θⱼ)
    # for i in range(n_spikes - 1):  # For each spike i
    #     for j in range(i + 1, n_spikes):  # For each spike j > i
    #         # Calculate the phase difference between spikes i and j
    #         phase_diff = spike_phases[i] - spike_phases[j]
            
    #         #f(θᵢ, θⱼ) = cos(θᵢ)cos(θⱼ) + sin(θᵢ)sin(θⱼ) can become #f(θᵢ, θⱼ) = cos(θᵢ - θⱼ)
    #         cos_diff = np.cos(phase_diff)
            
    #         # Add to the sum
    #         sum_cos_diff += cos_diff
    
    # # Calculate PPC according to the equation
    # # PPC = (2 / (N(N-1))) * Σᵢ Σⱼ₍ᵢ₊₁₎ f(θᵢ, θⱼ)
    # ppc = ((2 / (n_spikes * (n_spikes - 1))) * sum_cos_diff)
    
    # same as above (i think) but with vectorized computation and memory fixes so it wont take forever to run.
    if ppc_method == 'numpy':
        i, j = np.triu_indices(n_spikes, k=1)
        phase_diff = spike_phases[i] - spike_phases[j]
        sum_cos_diff = np.sum(np.cos(phase_diff))
        ppc = ((2 / (n_spikes * (n_spikes - 1))) * sum_cos_diff)
    elif ppc_method == 'numba':
        ppc = _ppc_parallel_numba(spike_phases)
    elif ppc_method == 'gpu':
        ppc = _ppc_gpu(spike_phases)
    else:
        raise ExceptionType("Please use a supported ppc method currently that is numpy, numba or gpu")
    return ppc

    
def calculate_ppc2(spike_times: np.ndarray = None, lfp_signal: np.ndarray = None, spike_fs: float = None,
                  lfp_fs: float = None, method: str = 'hilbert', freq_of_interest: float = None,
                  lowcut: float = None, highcut: float = None,
                  bandwidth: float = 2.0) -> tuple:
    """
    # -----------------------------------------------------------------------------
    # PPC2 Calculation (Vinck et al., 2010) 
    # -----------------------------------------------------------------------------
    # Equation(Original):
    #   PPC = (2 / (n * (n - 1))) * sum(cos(φ_i - φ_j) for all i < j)
    # Optimized Formula (Algebraically Equivalent):
    #   PPC = (|sum(e^(i*φ_j))|^2 - n) / (n * (n - 1))
    # -----------------------------------------------------------------------------
        
    Parameters:
    - spike_times: Array of spike times 
    - lfp_signal: Local field potential time series
    - spike_fs: Sampling frequency in Hz of the spike times only needed if spikes times and lfp has different fs
    - lfp_fs: Sampling frequency in Hz of the LFP
    - method: 'wavelet' or 'hilbert' to choose the phase extraction method
    - freq_of_interest: Desired frequency for wavelet phase extraction
    - lowcut, highcut: Cutoff frequencies for bandpass filtering (Hilbert method)
    - bandwidth: Bandwidth parameter for the wavelet
    
    Returns:
    - ppc2: Pairwise Phase Consistency 2 value
    """
    
    if spike_fs is None:
        spike_fs = lfp_fs
    # Convert spike times to sample indices
    spike_times_seconds = spike_times / spike_fs

    # Then convert from seconds to samples at the new sampling rate
    spike_indices = np.round(spike_times_seconds * lfp_fs).astype(int)
    
    # Filter indices to ensure they're within bounds of the LFP signal
    valid_indices = [idx for idx in spike_indices if 0 <= idx < len(lfp_signal)]
    if len(valid_indices) <= 1:
        return 0, np.array([])
    
    # Extract phase using the specified method
    if method == 'wavelet':
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")
        
        # Apply CWT to extract phase at the frequency of interest
        lfp_complex = wavelet_filter(x=lfp_signal, freq=freq_of_interest, fs=lfp_fs, bandwidth=bandwidth)
        instantaneous_phase = np.angle(lfp_complex)
        
    elif method == 'hilbert':
        if lowcut is None or highcut is None:
            print("Lowcut and/or highcut were not defined, signal will not be filtered and will just take Hilbert transform for PPC2 calculation")
            filtered_lfp = lfp_signal
        else:
            # Bandpass filter the signal
            filtered_lfp = butter_bandpass_filter(lfp_signal, lowcut, highcut, lfp_fs)
        
        # Get phase using the Hilbert transform
        analytic_signal = signal.hilbert(filtered_lfp)
        instantaneous_phase = np.angle(analytic_signal)
        
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'hilbert'.")
    
    # Get phases at spike times
    spike_phases = instantaneous_phase[valid_indices]
    
    # Calculate PPC2 according to Vinck et al. (2010), Equation 6
    n = len(spike_phases)
    
    if n <= 1:
        return 0, spike_phases
    
    # Convert phases to unit vectors in the complex plane
    unit_vectors = np.exp(1j * spike_phases)
    
    # Calculate the resultant vector
    resultant_vector = np.sum(unit_vectors)
    
    # PPC2 = (|∑(e^(i*φ_j))|² - n) / (n * (n - 1))
    ppc2 = (np.abs(resultant_vector)**2 - n) / (n * (n - 1))
    
    return ppc2


def calculate_ppc_per_cell(spike_df: pd.DataFrame, lfp_signal: np.ndarray,
                            spike_fs: float, lfp_fs:float,
                            pop_names: List[str],freqs: List[float]) -> Dict[str, Dict[int, Dict[float, float]]]:
    """
    Calculate pairwise phase consistency (PPC) per neuron (cell) for specified frequencies across different populations.

    This function computes the PPC for each neuron within the specified populations based on their spike times
    and a single-channel local field potential (LFP) signal.

    Args:
        spike_df (pd.DataFrame): Spike dataframe use bmtool.analysis.load_spikes_to_df
        lfp (xr.DataArray): xarray DataArray representing the LFP use bmtool.analysis.ecp_to_lfp
        spike_fs (float): sampling rate of spikes BMTK default is 1000
        lfp_fs (float): sampling rate of lfp 
        pop_names (List[str]): List of population names (as strings) to compute PPC for. pop_names should be in spike_df
        freqs (List[float]): List of frequencies (in Hz) at which to calculate PPC.

    Returns:
        Dict[str, Dict[int, Dict[float, float]]]: Nested dictionary where the structure is:
            {
                population_name: {
                    node_id: {
                        frequency: PPC value
                    }
                }
            }
            PPC values are floats representing the pairwise phase consistency at each frequency.
    """
    ppc_dict = {}
    for pop in pop_names:
        skip_count = 0
        pop_spikes = spike_df[spike_df['pop_name'] == pop]
        nodes = pop_spikes['node_ids'].unique()
        ppc_dict[pop] = {}
        print(f'Processing {pop} population')
        for node in tqdm(nodes):
            node_spikes = pop_spikes[pop_spikes['node_ids'] == node]
            
            # Skip nodes with less than or equal to 1 spike
            if len(node_spikes) <= 1:
                skip_count += 1
                continue

            ppc_dict[pop][node] = {}
            for freq in freqs:
                ppc = calculate_ppc2(
                    node_spikes['timestamps'].values,
                    lfp_signal,
                    spike_fs=spike_fs,
                    lfp_fs=lfp_fs,
                    freq_of_interest=freq,
                    method='wavelet'
                )
                ppc_dict[pop][node][freq] = ppc

        print(f'Calculated PPC for {pop} population with {len(nodes)-skip_count} valid cells, skipped {skip_count} cells for lack of spikes')

    return ppc_dict


def calculate_spike_rate_power_correlation(spike_rate, lfp, fs, pop_names, freq_range=(10, 100), freq_step=5):
    """
    Calculate correlation between population spike rates and LFP power across frequencies
    using wavelet filtering. This function assumes the fs of the spike_rate and lfp are the same.
    
    Parameters:
    -----------
    spike_rate : DataFrame
        Pre-calculated population spike rates at the same fs as lfp
    lfp : np.array
        LFP data
    fs : float
        Sampling frequency
    pop_names : list
        List of population names to analyze
    freq_range : tuple
        Min and max frequency to analyze
    freq_step : float
        Step size for frequency analysis
    
    Returns:
    --------
    correlation_results : dict
        Dictionary with correlation results for each population and frequency
    frequencies : array
        Array of frequencies analyzed
    """
    
    # Define frequency bands to analyze
    frequencies = np.arange(freq_range[0], freq_range[1] + 1, freq_step)
    
    # Dictionary to store results
    correlation_results = {pop: {} for pop in pop_names}
    
    # Calculate power at each frequency band using wavelet filter
    power_by_freq = {}
    for freq in frequencies:
        # Use the wavelet_filter function from bmlfp
        filtered_signal = wavelet_filter(lfp, freq, fs)
        # Calculate power (magnitude squared of complex wavelet transform)
        power = np.abs(filtered_signal)**2
        power_by_freq[freq] = power
    
    # Calculate correlation for each population
    for pop in pop_names:
        # Extract spike rate for this population
        pop_rate = spike_rate[pop]
        
        # Calculate correlation with power at each frequency
        for freq in frequencies:
            # Make sure the lengths match
            if len(pop_rate) != len(power_by_freq[freq]):
                raise Exception(f"Mismatched lengths for {pop} at {freq} Hz len(pop_rate): {len(pop_rate)}, len(power_by_freq): {len(power_by_freq[freq])}")
            # use spearman for non-parametric correlation
            corr, p_val = stats.spearmanr(pop_rate, power_by_freq[freq])
            correlation_results[pop][freq] = {'correlation': corr, 'p_value': p_val}
    
    return correlation_results, frequencies

