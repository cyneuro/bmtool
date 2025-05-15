"""
Module for entrainment analysis
"""

from typing import Dict, List, Optional, Union

import numba
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from numba import cuda
from scipy import signal
from tqdm.notebook import tqdm

from .lfp import butter_bandpass_filter, get_lfp_phase, get_lfp_power, wavelet_filter


def align_spike_times_with_lfp(lfp: xr.DataArray, timestamps: np.ndarray) -> np.ndarray:
    """the lfp xarray should have a time axis. use that to align the spike times since the lfp can start at a
    non-zero time after sliced. Both need to be on same fs for this to be correct.

    Parameters
    ----------
    lfp : xarray.DataArray
        LFP data with time coordinates
    timestamps : np.ndarray
        Array of spike timestamps

    Returns
    -------
    np.ndarray
        Copy of timestamps with adjusted timestamps to align with lfp.
    """
    # print("Pairing LFP and Spike Times")
    # print(lfp.time.values)
    # print(f"LFP starts at {lfp.time.values[0]}ms")
    # need to make sure lfp and spikes have the same time axis
    # align spikes with lfp
    timestamps = timestamps[
        (timestamps >= lfp.time.values[0]) & (timestamps <= lfp.time.values[-1])
    ].copy()
    # set the time axis of the spikes to match the lfp
    # timestamps = timestamps - lfp.time.values[0]
    return timestamps


def calculate_signal_signal_plv(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float,
    freq_of_interest: float = None,
    filter_method: str = "wavelet",
    lowcut: float = None,
    highcut: float = None,
    bandwidth: float = 2.0,
) -> np.ndarray:
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

    if filter_method == "wavelet":
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")

        # Apply CWT to both signals
        theta1 = wavelet_filter(x=signal1, freq=freq_of_interest, fs=fs, bandwidth=bandwidth)
        theta2 = wavelet_filter(x=signal2, freq=freq_of_interest, fs=fs, bandwidth=bandwidth)

    elif filter_method == "butter":
        if lowcut is None or highcut is None:
            print(
                "Lowcut and/or highcut were not defined, signal will not be filtered and will just take Hilbert transform for PLV calculation"
            )

        if lowcut and highcut:
            # Bandpass filter and get the analytic signal using the Hilbert transform
            filtered_signal1 = butter_bandpass_filter(
                data=signal1, lowcut=lowcut, highcut=highcut, fs=fs
            )
            filtered_signal2 = butter_bandpass_filter(
                data=signal2, lowcut=lowcut, highcut=highcut, fs=fs
            )
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


def _get_spike_phases(
    spike_times: np.ndarray,
    lfp_data: Union[np.ndarray, xr.DataArray],
    spike_fs: float,
    lfp_fs: float,
    filter_method: str = "wavelet",
    freq_of_interest: Optional[float] = None,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    bandwidth: float = 2.0,
    filtered_lfp_phase: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> np.ndarray:
    """
    Helper function to get spike phases from LFP data.

    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times
    lfp_data : Union[np.ndarray, xr.DataArray]
        Local field potential time series data. Not required if filtered_lfp_phase is provided.
    spike_fs : float
        Sampling frequency in Hz of the spike times
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
    filtered_lfp_phase : np.ndarray, optional
        Pre-computed instantaneous phase of the filtered LFP. If provided, the function will skip the filtering step.

    Returns
    -------
    np.ndarray
        Array of phases at spike times
    """
    # Convert spike times to sample indices
    spike_times_seconds = spike_times / spike_fs

    # Then convert from seconds to samples at the new sampling rate
    spike_indices = np.round(spike_times_seconds * lfp_fs).astype(int)

    # Filter indices to ensure they're within bounds of the LFP signal
    if isinstance(lfp_data, xr.DataArray):
        if filtered_lfp_phase is not None:
            valid_indices = align_spike_times_with_lfp(
                lfp=filtered_lfp_phase, timestamps=spike_indices
            )
        else:
            valid_indices = align_spike_times_with_lfp(lfp=lfp_data, timestamps=spike_indices)
    elif isinstance(lfp_data, np.ndarray):
        if filtered_lfp_phase is not None:
            valid_indices = [idx for idx in spike_indices if 0 <= idx < len(filtered_lfp_phase)]
        else:
            valid_indices = [idx for idx in spike_indices if 0 <= idx < len(lfp_data)]

    if len(valid_indices) <= 1:
        return np.array([])

    # Get instantaneous phase
    if filtered_lfp_phase is None:
        instantaneous_phase = get_lfp_phase(
            lfp_data=lfp_data,
            filter_method=filter_method,
            freq_of_interest=freq_of_interest,
            lowcut=lowcut,
            highcut=highcut,
            bandwidth=bandwidth,
            fs=lfp_fs,
        )
    else:
        instantaneous_phase = filtered_lfp_phase

    # Get phases at spike times
    if isinstance(instantaneous_phase, xr.DataArray):
        spike_phases = instantaneous_phase.sel(time=valid_indices, method="nearest").values
    else:
        spike_phases = instantaneous_phase[valid_indices]

    return spike_phases


def calculate_spike_lfp_plv(
    spike_times: np.ndarray = None,
    lfp_data=None,
    spike_fs: float = None,
    lfp_fs: float = None,
    filter_method: str = "butter",
    freq_of_interest: float = None,
    lowcut: float = None,
    highcut: float = None,
    bandwidth: float = 2.0,
    filtered_lfp_phase: np.ndarray = None,
) -> float:
    """
    Calculate spike-lfp unbiased phase locking value

    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times
    lfp_data : np.ndarray
        Local field potential time series data. Not required if filtered_lfp_phase is provided.
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
    filtered_lfp_phase : np.ndarray, optional
        Pre-computed instantaneous phase of the filtered LFP. If provided, the function will skip the filtering step.

    Returns
    -------
    float
        Phase Locking Value (unbiased)
    """

    spike_phases = _get_spike_phases(
        spike_times=spike_times,
        lfp_data=lfp_data,
        spike_fs=spike_fs,
        lfp_fs=lfp_fs,
        filter_method=filter_method,
        freq_of_interest=freq_of_interest,
        lowcut=lowcut,
        highcut=highcut,
        bandwidth=bandwidth,
        filtered_lfp_phase=filtered_lfp_phase,
    )

    if len(spike_phases) <= 1:
        return 0

    # Number of spikes
    N = len(spike_phases)

    # Convert phases to unit vectors in the complex plane
    unit_vectors = np.exp(1j * spike_phases)

    # Sum of all unit vectors (resultant vector)
    resultant_vector = np.sum(unit_vectors)

    # Calculate plv^2 * N
    plv2n = (resultant_vector * resultant_vector.conjugate()).real / N  # plv^2 * N
    plv = (plv2n / N) ** 0.5
    ppc = (plv2n - 1) / (N - 1)  # ppc = (plv^2 * N - 1) / (N - 1)
    plv_unbiased = np.fmax(ppc, 0.0) ** 0.5  # ensure non-negative

    return plv_unbiased


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
        for j in range(i + 1, len(spike_phases)):
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
    return (2 / (len(spike_phases) * (len(spike_phases) - 1))) * total


def calculate_ppc(
    spike_times: np.ndarray = None,
    lfp_data=None,
    spike_fs: float = None,
    lfp_fs: float = None,
    filter_method: str = "wavelet",
    freq_of_interest: float = None,
    lowcut: float = None,
    highcut: float = None,
    bandwidth: float = 2.0,
    ppc_method: str = "numpy",
    filtered_lfp_phase: np.ndarray = None,
) -> float:
    """
    Calculate Pairwise Phase Consistency (PPC) between spike times and LFP signal.
    Based on https://www.sciencedirect.com/science/article/pii/S1053811910000959

    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times
    lfp_data : np.ndarray
        Local field potential time series data. Not required if filtered_lfp_phase is provided.
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
    filtered_lfp_phase : np.ndarray, optional
        Pre-computed instantaneous phase of the filtered LFP. If provided, the function will skip the filtering step.

    Returns
    -------
    float
        Pairwise Phase Consistency value
    """

    spike_phases = _get_spike_phases(
        spike_times=spike_times,
        lfp_data=lfp_data,
        spike_fs=spike_fs,
        lfp_fs=lfp_fs,
        filter_method=filter_method,
        freq_of_interest=freq_of_interest,
        lowcut=lowcut,
        highcut=highcut,
        bandwidth=bandwidth,
        filtered_lfp_phase=filtered_lfp_phase,
    )

    if len(spike_phases) <= 1:
        return 0

    n_spikes = len(spike_phases)

    # Calculate PPC (Pairwise Phase Consistency)
    # Explicit calculation of pairwise phase consistency
    # Vectorized computation for efficiency
    if ppc_method == "numpy":
        i, j = np.triu_indices(n_spikes, k=1)
        phase_diff = spike_phases[i] - spike_phases[j]
        sum_cos_diff = np.sum(np.cos(phase_diff))
        ppc = (2 / (n_spikes * (n_spikes - 1))) * sum_cos_diff
    elif ppc_method == "numba":
        ppc = _ppc_parallel_numba(spike_phases)
    elif ppc_method == "gpu":
        ppc = _ppc_gpu(spike_phases)
    else:
        raise ValueError("Please use a supported ppc method currently that is numpy, numba or gpu")
    return ppc


def calculate_ppc2(
    spike_times: np.ndarray = None,
    lfp_data=None,
    spike_fs: float = None,
    lfp_fs: float = None,
    filter_method: str = "wavelet",
    freq_of_interest: float = None,
    lowcut: float = None,
    highcut: float = None,
    bandwidth: float = 2.0,
    filtered_lfp_phase: np.ndarray = None,
) -> float:
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
        Local field potential time series data. Not required if filtered_lfp_phase is provided.
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
    filtered_lfp_phase : np.ndarray, optional
        Pre-computed instantaneous phase of the filtered LFP. If provided, the function will skip the filtering step.

    Returns
    -------
    float
        Pairwise Phase Consistency 2 (PPC2) value
    """

    spike_phases = _get_spike_phases(
        spike_times=spike_times,
        lfp_data=lfp_data,
        spike_fs=spike_fs,
        lfp_fs=lfp_fs,
        filter_method=filter_method,
        freq_of_interest=freq_of_interest,
        lowcut=lowcut,
        highcut=highcut,
        bandwidth=bandwidth,
        filtered_lfp_phase=filtered_lfp_phase,
    )

    if len(spike_phases) <= 1:
        return 0

    # Calculate PPC2 according to Vinck et al. (2010), Equation 6
    n = len(spike_phases)

    # Convert phases to unit vectors in the complex plane
    unit_vectors = np.exp(1j * spike_phases)

    # Calculate the resultant vector
    resultant_vector = np.sum(unit_vectors)

    # PPC2 = (|∑(e^(i*φ_j))|² - n) / (n * (n - 1))
    ppc2 = (np.abs(resultant_vector) ** 2 - n) / (n * (n - 1))

    return ppc2


def calculate_entrainment_per_cell(
    spike_df: pd.DataFrame = None,
    lfp_data: np.ndarray = None,
    filter_method: str = "wavelet",
    pop_names: List[str] = None,
    entrainment_method: str = "plv",
    lowcut: float = None,
    highcut: float = None,
    spike_fs: float = None,
    lfp_fs: float = None,
    bandwidth: float = 2,
    freqs: List[float] = None,
    ppc_method: str = "numpy",
) -> Dict[str, Dict[int, Dict[float, float]]]:
    """
    Calculate neural entrainment (PPC, PLV) per neuron (cell) for specified frequencies across different populations.

    This function computes the entrainment metrics for each neuron within the specified populations based on their spike times
    and the provided LFP signal. It returns a nested dictionary structure containing the entrainment values
    organized by population, node ID, and frequency.

    Parameters
    ----------
    spike_df : pd.DataFrame
        DataFrame containing spike data with columns 'pop_name', 'node_ids', and 'timestamps'
    lfp_data : np.ndarray
        Local field potential (LFP) time series data
    filter_method : str, optional
        Method to use for filtering, either 'wavelet' or 'butter' (default: 'wavelet')
    entrainment_method : str, optional
        Method to use for entrainment calculation, either 'plv', 'ppc', or 'ppc2' (default: 'plv')
    lowcut : float, optional
        Lower frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    highcut : float, optional
        Upper frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    spike_fs : float
        Sampling frequency of the spike times in Hz
    lfp_fs : float
        Sampling frequency of the LFP signal in Hz
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 2.0)
    ppc_method : str, optional
        Algorithm to use for PPC calculation: 'numpy', 'numba', or 'gpu' (default: 'numpy')
    pop_names : List[str]
        List of population names to analyze
    freqs : List[float]
        List of frequencies (in Hz) at which to calculate entrainment

    Returns
    -------
    Dict[str, Dict[int, Dict[float, float]]]
        Nested dictionary where the structure is:
        {
            population_name: {
                node_id: {
                    frequency: entrainment value
                }
            }
        }
        Entrainment values are floats representing the metric (PPC, PLV) at each frequency
    """
    # pre filter lfp to speed up calculate of entrainment
    filtered_lfp_phases = {}
    for freq in range(len(freqs)):
        phase = get_lfp_phase(
            lfp_data=lfp_data,
            freq_of_interest=freqs[freq],
            fs=lfp_fs,
            filter_method=filter_method,
            lowcut=lowcut,
            highcut=highcut,
            bandwidth=bandwidth,
        )
        filtered_lfp_phases[freqs[freq]] = phase

    entrainment_dict = {}
    for pop in pop_names:
        skip_count = 0
        pop_spikes = spike_df[spike_df["pop_name"] == pop]
        nodes = sorted(pop_spikes["node_ids"].unique())  # sort so all nodes are processed in order
        entrainment_dict[pop] = {}
        print(f"Processing {pop} population")
        for node in tqdm(nodes):
            node_spikes = pop_spikes[pop_spikes["node_ids"] == node]

            # Skip nodes with less than or equal to 1 spike
            if len(node_spikes) <= 1:
                skip_count += 1
                continue

            entrainment_dict[pop][node] = {}
            for freq in freqs:
                # Calculate entrainment based on the selected method using the pre-filtered phases
                if entrainment_method == "plv":
                    entrainment_dict[pop][node][freq] = calculate_spike_lfp_plv(
                        node_spikes["timestamps"].values,
                        lfp_data,
                        spike_fs=spike_fs,
                        lfp_fs=lfp_fs,
                        freq_of_interest=freq,
                        bandwidth=bandwidth,
                        lowcut=lowcut,
                        highcut=highcut,
                        filter_method=filter_method,
                        filtered_lfp_phase=filtered_lfp_phases[freq],
                    )
                elif entrainment_method == "ppc2":
                    entrainment_dict[pop][node][freq] = calculate_ppc2(
                        node_spikes["timestamps"].values,
                        lfp_data,
                        spike_fs=spike_fs,
                        lfp_fs=lfp_fs,
                        freq_of_interest=freq,
                        bandwidth=bandwidth,
                        lowcut=lowcut,
                        highcut=highcut,
                        filter_method=filter_method,
                        filtered_lfp_phase=filtered_lfp_phases[freq],
                    )
                elif entrainment_method == "ppc":
                    entrainment_dict[pop][node][freq] = calculate_ppc(
                        node_spikes["timestamps"].values,
                        lfp_data,
                        spike_fs=spike_fs,
                        lfp_fs=lfp_fs,
                        freq_of_interest=freq,
                        bandwidth=bandwidth,
                        lowcut=lowcut,
                        highcut=highcut,
                        filter_method=filter_method,
                        ppc_method=ppc_method,
                        filtered_lfp_phase=filtered_lfp_phases[freq],
                    )

        print(
            f"Calculated {entrainment_method.upper()} for {pop} population with {len(nodes)-skip_count} valid cells, skipped {skip_count} cells for lack of spikes"
        )

    return entrainment_dict


def calculate_spike_rate_power_correlation(
    spike_rate,
    lfp_data,
    fs,
    pop_names,
    filter_method="wavelet",
    bandwidth=2.0,
    lowcut=None,
    highcut=None,
    freq_range=(10, 100),
    freq_step=5,
):
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
        power_by_freq[freq] = get_lfp_power(
            lfp_data, freq, fs, filter_method, lowcut=lowcut, highcut=highcut, bandwidth=bandwidth
        )

    # Calculate correlation for each population
    for pop in pop_names:
        # Extract spike rate for this population
        pop_rate = spike_rate[pop]

        # Calculate correlation with power at each frequency
        for freq in frequencies:
            # Make sure the lengths match
            if len(pop_rate) != len(power_by_freq[freq]):
                raise ValueError(
                    f"Mismatched lengths for {pop} at {freq} Hz len(pop_rate): {len(pop_rate)}, len(power_by_freq): {len(power_by_freq[freq])}"
                )
            # use spearman for non-parametric correlation
            corr, p_val = stats.spearmanr(pop_rate, power_by_freq[freq])
            correlation_results[pop][freq] = {"correlation": corr, "p_value": p_val}

    return correlation_results, frequencies


def get_spikes_in_cycle(
    spike_df,
    lfp_data,
    spike_fs=1000,
    lfp_fs=400,
    filter_method="butter",
    lowcut=None,
    highcut=None,
    bandwidth=2.0,
    freq_of_interest=None,
):
    """
    Analyze spike timing relative to oscillation phases.

    Parameters:
    -----------
    spike_df : pd.DataFrame
    lfp_data : np.array
        Raw LFP signal
    fs : float
        Sampling frequency of LFP in Hz
    gamma_band : tuple
        Lower and upper bounds of gamma frequency band in Hz

    Returns:
    --------
    phase_data : dict
        Dictionary containing phase values for each spike and neuron population
    """
    phase = get_lfp_phase(
        lfp_data=lfp_data,
        fs=lfp_fs,
        filter_method=filter_method,
        lowcut=lowcut,
        highcut=highcut,
        bandwidth=bandwidth,
        freq_of_interest=freq_of_interest,
    )

    # Get unique neuron populations
    neuron_pops = spike_df["pop_name"].unique()

    # Get the phase at each spike time for each neuron population
    phase_data = {}

    for pop in neuron_pops:
        # Get spike times for this population
        pop_spikes = spike_df[spike_df["pop_name"] == pop]["timestamps"].values

        # Convert spike times to sample indices
        spike_times_seconds = pop_spikes / spike_fs

        # Then convert from seconds to samples at the new sampling rate
        spike_indices = np.round(spike_times_seconds * lfp_fs).astype(int)

        # Ensure spike times are within LFP data range
        valid_indices = (spike_indices >= 0) & (spike_indices < len(phase))

        if np.any(valid_indices):
            valid_samples = spike_indices[valid_indices]
            phase_data[pop] = phase[valid_samples]

    return phase_data
