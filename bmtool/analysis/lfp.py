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


def calculate_plv(x1: np.ndarray, x2: np.ndarray, fs: float, freq_of_interest: float = None, 
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


def calculate_plv_over_time(x1: np.ndarray, x2: np.ndarray, fs: float, 
                            window_size: float, step_size: float, 
                            method: str = 'wavelet', freq_of_interest: float = None, 
                            lowcut: float = None, highcut: float = None, 
                            bandwidth: float = 2.0):
    """
    Calculate the time-resolved Phase Locking Value (PLV) between two signals using a sliding window approach.
    
    Parameters:
    ----------
    x1, x2 : array-like
        Input signals (1D arrays, same length).
    fs : float
        Sampling frequency of the input signals.
    window_size : float
        Length of the window in seconds for PLV calculation.
    step_size : float
        Step size in seconds to slide the window across the signals.
    method : str, optional
        Method to calculate PLV ('wavelet' or 'hilbert'). Defaults to 'wavelet'.
    freq_of_interest : float, optional
        Frequency of interest for the wavelet method. Required if method is 'wavelet'.
    lowcut, highcut : float, optional
        Cutoff frequencies for the Hilbert method. Required if method is 'hilbert'.
    bandwidth : float, optional
        Bandwidth parameter for the wavelet. Defaults to 2.0.
        
    Returns:
    -------
    plv_over_time : 1D array
        Array of PLV values calculated over each window.
    times : 1D array
        The center times of each window where the PLV was calculated.
    """
    # Convert window and step size from seconds to samples
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    
    # Initialize results
    plv_over_time = []
    times = []
    
    # Iterate over the signal with a sliding window
    for start in range(0, len(x1) - window_samples + 1, step_samples):
        end = start + window_samples
        window_x1 = x1[start:end]
        window_x2 = x2[start:end]
        
        # Use the updated calculate_plv function within each window
        plv = calculate_plv(x1=window_x1, x2=window_x2, fs=fs, 
                            method=method, freq_of_interest=freq_of_interest, 
                            lowcut=lowcut, highcut=highcut, bandwidth=bandwidth)
        plv_over_time.append(plv)
        
        # Store the time at the center of the window
        center_time = (start + end) / 2 / fs
        times.append(center_time)
    
    return np.array(plv_over_time), np.array(times)


