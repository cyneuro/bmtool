"""
Module for processing BMTK LFP output.
"""

import h5py
import numpy as np
import xarray as xr
from fooof import FOOOF
from fooof.sim.gen import gen_model
import matplotlib.pyplot as plt
from scipy import signal 
import pywt
from bmtool.bmplot import is_notebook
import numba
from numba import cuda


def load_ecp_to_xarray(ecp_file: str, demean: bool = False) -> xr.DataArray:
    """
    Load ECP data from an HDF5 file (BMTK sim) into an xarray DataArray.

    Parameters:
    ----------
    ecp_file : str
        Path to the HDF5 file containing ECP data.
    demean : bool, optional
        If True, the mean of the data will be subtracted (default is False).

    Returns:
    -------
    xr.DataArray
        An xarray DataArray containing the ECP data, with time as one dimension
        and channel_id as another.
    """
    with h5py.File(ecp_file, 'r') as f:
        ecp = xr.DataArray(
            f['ecp']['data'][()].T,
            coords=dict(
                channel_id=f['ecp']['channel_id'][()],
                time=np.arange(*f['ecp']['time'])  # ms
            ),
            attrs=dict(
                fs=1000 / f['ecp']['time'][2]  # Hz
            )
        )
    if demean:
        ecp -= ecp.mean(dim='time')
    return ecp


def ecp_to_lfp(ecp_data: xr.DataArray, cutoff: float = 250, fs: float = 10000,
                    downsample_freq: float = 1000) -> xr.DataArray:
    """
    Apply a low-pass Butterworth filter to an xarray DataArray and optionally downsample. 
    This filters out the high end frequencies turning the ECP into a LFP

    Parameters:
    ----------
    ecp_data : xr.DataArray
        The input data array containing LFP data with time as one dimension.
    cutoff : float
        The cutoff frequency for the low-pass filter in Hz (default is 250Hz).
    fs : float, optional
        The sampling frequency of the data (default is 10000 Hz).
    downsample_freq : float, optional
        The frequency to downsample to (default is 1000 Hz).

    Returns:
    -------
    xr.DataArray
        The filtered (and possibly downsampled) data as an xarray DataArray.
    """
    # Bandpass filter design
    nyq = 0.5 * fs
    cut = cutoff / nyq
    b, a = signal.butter(8, cut, btype='low', analog=False)

    # Initialize an array to hold filtered data
    filtered_data = xr.DataArray(np.zeros_like(ecp_data), coords=ecp_data.coords, dims=ecp_data.dims)

    # Apply the filter to each channel
    for channel in ecp_data.channel_id:
        filtered_data.loc[channel, :] = signal.filtfilt(b, a, ecp_data.sel(channel_id=channel).values)

    # Downsample the filtered data if a downsample frequency is provided
    if downsample_freq is not None:
        downsample_factor = int(fs / downsample_freq)
        filtered_data = filtered_data.isel(time=slice(None, None, downsample_factor))
        # Update the sampling frequency attribute
        filtered_data.attrs['fs'] = downsample_freq

    return filtered_data


def slice_time_series(data: xr.DataArray, time_ranges: tuple) -> xr.DataArray:
    """
    Slice the xarray DataArray based on provided time ranges.
    Can be used to get LFP during certain stimulus times

    Parameters:
    ----------
    data : xr.DataArray
        The input xarray DataArray containing time-series data.
    time_ranges : tuple or list of tuples
        One or more tuples representing the (start, stop) time points for slicing. 
        For example: (start, stop) or [(start1, stop1), (start2, stop2)]

    Returns:
    -------
    xr.DataArray
        A new xarray DataArray containing the concatenated slices.
    """
    # Ensure time_ranges is a list of tuples
    if isinstance(time_ranges, tuple) and len(time_ranges) == 2:
        time_ranges = [time_ranges]

    # List to hold sliced data
    slices = []

    # Slice the data for each time range
    for start, stop in time_ranges:
        sliced_data = data.sel(time=slice(start, stop))
        slices.append(sliced_data)

    # Concatenate all slices along the time dimension if more than one slice
    if len(slices) > 1:
        return xr.concat(slices, dim='time')
    else:
        return slices[0]


def fit_fooof(f: np.ndarray, pxx: np.ndarray, aperiodic_mode: str = 'fixed',
              dB_threshold: float = 3.0, max_n_peaks: int = 10,
              freq_range: tuple = None, peak_width_limits: tuple = None,
              report: bool = False, plot: bool = False, 
              plt_log: bool = False, plt_range: tuple = None,
              figsize: tuple = None, title: str = None) -> tuple:
    """
    Fit a FOOOF model to power spectral density data.

    Parameters:
    ----------
    f : array-like
        Frequencies corresponding to the power spectral density data.
    pxx : array-like
        Power spectral density data to fit.
    aperiodic_mode : str, optional
        The mode for fitting aperiodic components ('fixed' or 'knee', default is 'fixed').
    dB_threshold : float, optional
        Minimum peak height in dB (default is 3).
    max_n_peaks : int, optional
        Maximum number of peaks to fit (default is 10).
    freq_range : tuple, optional
        Frequency range to fit (default is None, which uses the full range).
    peak_width_limits : tuple, optional
        Limits on the width of peaks (default is None).
    report : bool, optional
        If True, will print fitting results (default is False).
    plot : bool, optional
        If True, will plot the fitting results (default is False).
    plt_log : bool, optional
        If True, use a logarithmic scale for the y-axis in plots (default is False).
    plt_range : tuple, optional
        Range for plotting (default is None).
    figsize : tuple, optional
        Size of the figure (default is None).
    title : str, optional
        Title for the plot (default is None).

    Returns:
    -------
    tuple
        A tuple containing the fitting results and the FOOOF model object.
    """
    if aperiodic_mode != 'knee':
        aperiodic_mode = 'fixed'
    
    def set_range(x, upper=f[-1]):
        x = np.array(upper) if x is None else np.array(x)
        return [f[2], x.item()] if x.size == 1 else x.tolist()
    
    freq_range = set_range(freq_range)
    peak_width_limits = set_range(peak_width_limits, np.inf)

    # Initialize a FOOOF object
    fm = FOOOF(peak_width_limits=peak_width_limits, min_peak_height=dB_threshold / 10,
               peak_threshold=0., max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
    
    # Fit the model
    try:
        fm.fit(f, pxx, freq_range)
    except Exception as e:
        fl = np.linspace(f[0], f[-1], int((f[-1] - f[0]) / np.min(np.diff(f))) + 1)
        fm.fit(fl, np.interp(fl, f, pxx), freq_range)
    
    results = fm.get_results()

    if report:
        fm.print_results()
        if aperiodic_mode == 'knee':
            ap_params = results.aperiodic_params
            if ap_params[1] <= 0:
                print('Negative value of knee parameter occurred. Suggestion: Fit without knee parameter.')
            knee_freq = np.abs(ap_params[1]) ** (1 / ap_params[2])
            print(f'Knee location: {knee_freq:.2f} Hz')
    
    if plot:
        plt_range = set_range(plt_range)
        fm.plot(plt_log=plt_log)
        plt.xlim(np.log10(plt_range) if plt_log else plt_range)
        #plt.ylim(-8, -5.5)
        if figsize:
            plt.gcf().set_size_inches(figsize)
        if title:
            plt.title(title)
        if is_notebook():
            pass
        else:
            plt.show()
    
    return results, fm


def generate_resd_from_fooof(fooof_model: FOOOF) -> tuple:
    """
    Generate residuals from a fitted FOOOF model.

    Parameters:
    ----------
    fooof_model : FOOOF
        A fitted FOOOF model object.

    Returns:
    -------
    tuple
        A tuple containing the residual power spectral density and the aperiodic fit.
    """
    results = fooof_model.get_results()
    full_fit, _, ap_fit = gen_model(fooof_model.freqs[1:], results.aperiodic_params,
                                     results.gaussian_params, return_components=True)
    
    full_fit, ap_fit = 10 ** full_fit, 10 ** ap_fit  # Convert back from log
    res_psd = np.insert((10 ** fooof_model.power_spectrum[1:]) - ap_fit, 0, 0.)  # Convert back from log
    res_fit = np.insert(full_fit - ap_fit, 0, 0.)
    ap_fit = np.insert(ap_fit, 0, 0.)

    return res_psd, ap_fit


def calculate_SNR(fooof_model: FOOOF, freq_band: tuple) -> float:
    """
    Calculate the signal-to-noise ratio (SNR) from a fitted FOOOF model.

    Parameters:
    ----------
    fooof_model : FOOOF
        A fitted FOOOF model object.
    freq_band : tuple
        Frequency band (min, max) for SNR calculation.

    Returns:
    -------
    float
        The calculated SNR for the specified frequency band.
    """
    periodic, ap = generate_resd_from_fooof(fooof_model)
    freq = fooof_model.freqs  # Get frequencies from model
    indices = (freq >= freq_band[0]) & (freq <= freq_band[1])  # Get only the band we care about
    band_periodic = periodic[indices]  # Filter based on band
    band_ap = ap[indices]  # Filter
    band_freq = freq[indices]  # Another filter
    periodic_power = np.trapz(band_periodic, band_freq)  # Integrate periodic power
    ap_power = np.trapz(band_ap, band_freq)  # Integrate aperiodic power
    normalized_power = periodic_power / ap_power  # Compute the SNR
    return normalized_power


def wavelet_filter(x: np.ndarray, freq: float, fs: float, bandwidth: float = 1.0, axis: int = -1) -> np.ndarray:
    """
    Compute the Continuous Wavelet Transform (CWT) for a specified frequency using a complex Morlet wavelet.
    """
    wavelet = 'cmor' + str(2 * bandwidth ** 2) + '-1.0'
    scale = pywt.scale2frequency(wavelet, 1) * fs / freq
    x_a = pywt.cwt(x, [scale], wavelet=wavelet, axis=axis)[0][0]
    return x_a


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5, axis: int = -1) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the input data.
    """
    sos = signal.butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
    x_a = signal.sosfiltfilt(sos, data, axis=axis)
    return x_a


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
    - spike_phases: Phases at spike times
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

    


