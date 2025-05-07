"""
Module for entrainment analysis
"""

import numpy as np
from scipy import signal
import numba
from numba import cuda
import pandas as pd
import xarray as xr
from .lfp import wavelet_filter,butter_bandpass_filter,get_lfp_power
from typing import Dict, List
from tqdm.notebook import tqdm
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_signal_signal_plv(signal1: np.ndarray, signal2: np.ndarray, fs: float, freq_of_interest: float = None, 
                  filter_method: str = 'wavelet', lowcut: float = None, highcut: float = None, 
                  bandwidth: float = 2.0) -> np.ndarray:
    """
    Calculate Phase Locking Value (PLV) between two signals using wavelet or Hilbert method.
    
    Parameters
    ----------
    signal1 : np.ndarray
        First input signal (1D array)
    signal2 : np.ndarray
        Second input signal (1D array, same length as signal1)
    fs : float
        Sampling frequency in Hz
    freq_of_interest : float, optional
        Desired frequency for wavelet PLV calculation, required if filter_method='wavelet'
    filter_method : str, optional
        Method to use for filtering, either 'wavelet' or 'butter' (default: 'wavelet')
    lowcut : float, optional
        Lower frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    highcut : float, optional
        Upper frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 2.0)
    
    Returns
    -------
    np.ndarray
        Phase Locking Value (1D array)
    """
    if len(signal1) != len(signal2):
        raise ValueError("Input signals must have the same length.")
    
    if filter_method == 'wavelet':
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")
        
        # Apply CWT to both signals
        theta1 = wavelet_filter(x=signal1, freq=freq_of_interest, fs=fs, bandwidth=bandwidth)
        theta2 = wavelet_filter(x=signal2, freq=freq_of_interest, fs=fs, bandwidth=bandwidth)
    
    elif filter_method == 'butter':
        if lowcut is None or highcut is None:
            print("Lowcut and/or highcut were not defined, signal will not be filtered and will just take Hilbert transform for PLV calculation")
        
        if lowcut and highcut:
            # Bandpass filter and get the analytic signal using the Hilbert transform
            filtered_signal1 = butter_bandpass_filter(data=signal1, lowcut=lowcut, highcut=highcut, fs=fs)
            filtered_signal2 = butter_bandpass_filter(data=signal2, lowcut=lowcut, highcut=highcut, fs=fs)
            # Get phase using the Hilbert transform
            theta1 = signal.hilbert(filtered_signal1)
            theta2 = signal.hilbert(filtered_signal2)
        else:
            # Get phase using the Hilbert transform without filtering
            theta1 = signal.hilbert(signal1)
            theta2 = signal.hilbert(signal2)
    
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'butter'.")
    
    # Calculate phase difference
    phase_diff = np.angle(theta1) - np.angle(theta2)
    
    # Calculate PLV from standard equation from Measuring phase synchrony in brain signals(1999)
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))
    
    return plv


def calculate_spike_lfp_plv(spike_times: np.ndarray = None, lfp_data: np.ndarray = None, spike_fs: float = None,
                   lfp_fs: float = None, filter_method: str = 'butter', freq_of_interest: float = None,
                   lowcut: float = None, highcut: float = None,
                   bandwidth: float = 2.0) -> tuple:
    """
    Calculate spike-lfp phase locking value Based on https://www.sciencedirect.com/science/article/pii/S1053811910000959
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times
    lfp_data : np.ndarray
        Local field potential time series data
    spike_fs : float, optional
        Sampling frequency in Hz of the spike times, only needed if spike times and LFP have different sampling rates
    lfp_fs : float
        Sampling frequency in Hz of the LFP data
    filter_method : str, optional
        Method to use for filtering, either 'wavelet' or 'butter' (default: 'butter')
    freq_of_interest : float, optional
        Desired frequency for wavelet phase extraction, required if filter_method='wavelet'
    lowcut : float, optional
        Lower frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    highcut : float, optional
        Upper frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 2.0)
    
    Returns
    -------
    tuple
        (plv, spike_phases) where:
        - plv: Phase Locking Value
        - spike_phases: Phases at spike times
    """
    
    if spike_fs is None:
        spike_fs = lfp_fs
    # Convert spike times to sample indices
    spike_times_seconds = spike_times / spike_fs

    # Then convert from seconds to samples at the new sampling rate
    spike_indices = np.round(spike_times_seconds * lfp_fs).astype(int)
    
    # Filter indices to ensure they're within bounds of the LFP signal
    valid_indices = [idx for idx in spike_indices if 0 <= idx < len(lfp_data)]
    if len(valid_indices) <= 1:
        return 0, np.array([])
    
    # Filter the LFP signal to extract the phase
    if filter_method == 'wavelet':
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")
        
        # Apply CWT to extract phase
        filtered_lfp = wavelet_filter(x=lfp_data, freq=freq_of_interest, fs=lfp_fs, bandwidth=bandwidth)
    
    elif filter_method == 'butter':
        if lowcut is None or highcut is None:
            raise ValueError("Both lowcut and highcut must be specified for the butter method.")
        
        # Bandpass filter the LFP signal
        filtered_lfp = butter_bandpass_filter(data=lfp_data, lowcut=lowcut, highcut=highcut, fs=lfp_fs)
        filtered_lfp = signal.hilbert(filtered_lfp)  # Get analytic signal
    
        
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'butter'.")
    
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


def calculate_ppc(spike_times: np.ndarray = None, lfp_data: np.ndarray = None, spike_fs: float = None,
                  lfp_fs: float = None, filter_method: str = 'wavelet', freq_of_interest: float = None,
                  lowcut: float = None, highcut: float = None,
                  bandwidth: float = 2.0, ppc_method: str = 'numpy') -> tuple:
    """
    Calculate Pairwise Phase Consistency (PPC) between spike times and LFP signal.
    Based on https://www.sciencedirect.com/science/article/pii/S1053811910000959
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times
    lfp_data : np.ndarray
        Local field potential time series data
    spike_fs : float, optional
        Sampling frequency in Hz of the spike times, only needed if spike times and LFP have different sampling rates
    lfp_fs : float
        Sampling frequency in Hz of the LFP data
    filter_method : str, optional
        Method to use for filtering, either 'wavelet' or 'butter' (default: 'wavelet')
    freq_of_interest : float, optional
        Desired frequency for wavelet phase extraction, required if filter_method='wavelet'
    lowcut : float, optional
        Lower frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    highcut : float, optional
        Upper frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 2.0)
    ppc_method : str, optional
        Algorithm to use for PPC calculation: 'numpy', 'numba', or 'gpu' (default: 'numpy')
    
    Returns
    -------
    tuple
        (ppc, spike_phases) where:
        - ppc: Pairwise Phase Consistency value
        - spike_phases: Phases at spike times
    """
    if spike_fs is None:
        spike_fs = lfp_fs
    # Convert spike times to sample indices
    spike_times_seconds = spike_times / spike_fs

    # Then convert from seconds to samples at the new sampling rate
    spike_indices = np.round(spike_times_seconds * lfp_fs).astype(int)
    
    # Filter indices to ensure they're within bounds of the LFP signal
    valid_indices = [idx for idx in spike_indices if 0 <= idx < len(lfp_data)]
    if len(valid_indices) <= 1:
        return 0, np.array([])
    
    # Extract phase using the specified method
    if filter_method == 'wavelet':
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")
        
        # Apply CWT to extract phase at the frequency of interest
        lfp_complex = wavelet_filter(x=lfp_data, freq=freq_of_interest, fs=lfp_fs, bandwidth=bandwidth)
        instantaneous_phase = np.angle(lfp_complex)
        
    elif filter_method == 'butter':
        if lowcut is None or highcut is None:
            raise ValueError("Both lowcut and highcut must be specified for the butter method.")
        
        # Bandpass filter the signal
        filtered_lfp = butter_bandpass_filter(data=lfp_data, lowcut=lowcut, highcut=highcut, fs=lfp_fs)
        
        # Get phase using the Hilbert transform
        analytic_signal = signal.hilbert(filtered_lfp)
        instantaneous_phase = np.angle(analytic_signal)
        
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'butter'.")
    
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

    
def calculate_ppc2(spike_times: np.ndarray = None, lfp_data: np.ndarray = None, spike_fs: float = None,
                  lfp_fs: float = None, filter_method: str = 'wavelet', freq_of_interest: float = None,
                  lowcut: float = None, highcut: float = None,
                  bandwidth: float = 2.0) -> float:
    """
    # -----------------------------------------------------------------------------
    # PPC2 Calculation (Vinck et al., 2010) 
    # -----------------------------------------------------------------------------
    # Equation(Original):
    #   PPC = (2 / (n * (n - 1))) * sum(cos(φ_i - φ_j) for all i < j)
    # Optimized Formula (Algebraically Equivalent):
    #   PPC = (|sum(e^(i*φ_j))|^2 - n) / (n * (n - 1))
    # -----------------------------------------------------------------------------
        
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times
    lfp_data : np.ndarray
        Local field potential time series data
    spike_fs : float, optional
        Sampling frequency in Hz of the spike times, only needed if spike times and LFP have different sampling rates
    lfp_fs : float
        Sampling frequency in Hz of the LFP data
    filter_method : str, optional
        Method to use for filtering, either 'wavelet' or 'butter' (default: 'wavelet')
    freq_of_interest : float, optional
        Desired frequency for wavelet phase extraction, required if filter_method='wavelet'
    lowcut : float, optional
        Lower frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    highcut : float, optional
        Upper frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 2.0)
    
    Returns
    -------
    float
        Pairwise Phase Consistency 2 (PPC2) value
    """
    
    if spike_fs is None:
        spike_fs = lfp_fs
    # Convert spike times to sample indices
    spike_times_seconds = spike_times / spike_fs

    # Then convert from seconds to samples at the new sampling rate
    spike_indices = np.round(spike_times_seconds * lfp_fs).astype(int)
    
    # Filter indices to ensure they're within bounds of the LFP signal
    valid_indices = [idx for idx in spike_indices if 0 <= idx < len(lfp_data)]
    if len(valid_indices) <= 1:
        return 0
    
    # Extract phase using the specified method
    if filter_method == 'wavelet':
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")
        
        # Apply CWT to extract phase at the frequency of interest
        lfp_complex = wavelet_filter(x=lfp_data, freq=freq_of_interest, fs=lfp_fs, bandwidth=bandwidth)
        instantaneous_phase = np.angle(lfp_complex)
        
    elif filter_method == 'butter':
        if lowcut is None or highcut is None:
            raise ValueError("Both lowcut and highcut must be specified for the butter method.")
        
        # Bandpass filter the signal
        filtered_lfp = butter_bandpass_filter(data=lfp_data, lowcut=lowcut, highcut=highcut, fs=lfp_fs)
        
        # Get phase using the Hilbert transform
        analytic_signal = signal.hilbert(filtered_lfp)
        instantaneous_phase = np.angle(analytic_signal)
        
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'butter'.")
    
    # Get phases at spike times
    spike_phases = instantaneous_phase[valid_indices]
    
    # Calculate PPC2 according to Vinck et al. (2010), Equation 6
    n = len(spike_phases)
    
    if n <= 1:
        return 0
    
    # Convert phases to unit vectors in the complex plane
    unit_vectors = np.exp(1j * spike_phases)
    
    # Calculate the resultant vector
    resultant_vector = np.sum(unit_vectors)
    
    # PPC2 = (|∑(e^(i*φ_j))|² - n) / (n * (n - 1))
    ppc2 = (np.abs(resultant_vector)**2 - n) / (n * (n - 1))
    
    return ppc2


def calculate_ppc_per_cell(spike_df: pd.DataFrame=None, lfp_data: np.ndarray=None,
                            spike_fs: float=None, lfp_fs: float=None, bandwidth: float=2,
                            pop_names: List[str]=None, freqs: List[float]=None) -> Dict[str, Dict[int, Dict[float, float]]]:
    """
    Calculate pairwise phase consistency (PPC) per neuron (cell) for specified frequencies across different populations.

    This function computes the PPC for each neuron within the specified populations based on their spike times
    and the provided LFP signal. It returns a nested dictionary structure containing the PPC values
    organized by population, node ID, and frequency.

    Parameters
    ----------
    spike_df : pd.DataFrame
        DataFrame containing spike data with columns 'pop_name', 'node_ids', and 'timestamps'
    lfp_data : np.ndarray
        Local field potential (LFP) time series data
    spike_fs : float
        Sampling frequency of the spike times in Hz
    lfp_fs : float
        Sampling frequency of the LFP signal in Hz
    pop_names : List[str]
        List of population names to analyze
    freqs : List[float]
        List of frequencies (in Hz) at which to calculate PPC

    Returns
    -------
    Dict[str, Dict[int, Dict[float, float]]]
        Nested dictionary where the structure is:
        {
            population_name: {
                node_id: {
                    frequency: PPC value
                }
            }
        }
        PPC values are floats representing the pairwise phase consistency at each frequency
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
                    lfp_data,
                    spike_fs=spike_fs,
                    lfp_fs=lfp_fs,
                    freq_of_interest=freq,
                    bandwidth=bandwidth,
                    filter_method='wavelet'
                )
                ppc_dict[pop][node][freq] = ppc

        print(f'Calculated PPC for {pop} population with {len(nodes)-skip_count} valid cells, skipped {skip_count} cells for lack of spikes')

    return ppc_dict


def calculate_spike_rate_power_correlation(spike_rate, lfp_data, fs, pop_names, filter_method='wavelet',
                                          bandwidth=2.0, lowcut=None, highcut=None,
                                          freq_range=(10, 100), freq_step=5):
    """
    Calculate correlation between population spike rates and LFP power across frequencies
    using wavelet filtering. This function assumes the fs of the spike_rate and lfp are the same.
    
    Parameters:
    -----------
    spike_rate : DataFrame
        Pre-calculated population spike rates at the same fs as lfp
    lfp_data : np.array
        LFP data
    fs : float
        Sampling frequency
    pop_names : list
        List of population names to analyze
    filter_method : str, optional
        Filtering method to use, either 'wavelet' or 'butter' (default: 'wavelet')
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 2.0)
    lowcut : float, optional
        Lower frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    highcut : float, optional
        Upper frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    freq_range : tuple, optional
        Min and max frequency to analyze (default: (10, 100))
    freq_step : float, optional
        Step size for frequency analysis (default: 5)
    
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
    
    # Calculate power at each frequency band using specified filter
    power_by_freq = {}
    for freq in frequencies:
        if filter_method == 'wavelet':
            power_by_freq[freq] = get_lfp_power(lfp_data, freq, fs, filter_method, 
                                               lowcut=None, highcut=None, bandwidth=bandwidth)
        elif filter_method == 'butter':
            power_by_freq[freq] = get_lfp_power(lfp_data, freq, fs, filter_method, 
                                               lowcut=lowcut, highcut=highcut)
    
    # Calculate correlation for each population
    for pop in pop_names:
        # Extract spike rate for this population
        pop_rate = spike_rate[pop]
        
        # Calculate correlation with power at each frequency
        for freq in frequencies:
            # Make sure the lengths match
            if len(pop_rate) != len(power_by_freq[freq]):
                raise ValueError(f"Mismatched lengths for {pop} at {freq} Hz len(pop_rate): {len(pop_rate)}, len(power_by_freq): {len(power_by_freq[freq])}")
            # use spearman for non-parametric correlation
            corr, p_val = stats.spearmanr(pop_rate, power_by_freq[freq])
            correlation_results[pop][freq] = {'correlation': corr, 'p_value': p_val}
    
    return correlation_results, frequencies

