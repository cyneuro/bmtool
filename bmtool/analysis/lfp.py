"""
Module for processing BMTK LFP output.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import xarray as xr
from fooof import FOOOF
from fooof.sim.gen import gen_model
from scipy import signal

from ..bmplot.connections import is_notebook


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
    with h5py.File(ecp_file, "r") as f:
        ecp = xr.DataArray(
            f["ecp"]["data"][()].T,
            coords=dict(
                channel_id=f["ecp"]["channel_id"][()],
                time=np.arange(*f["ecp"]["time"]),  # ms
            ),
            attrs=dict(
                fs=1000 / f["ecp"]["time"][2]  # Hz
            ),
        )
    if demean:
        ecp -= ecp.mean(dim="time")
    return ecp


def ecp_to_lfp(
    ecp_data: xr.DataArray, cutoff: float = 250, fs: float = 10000, downsample_freq: float = 1000
) -> xr.DataArray:
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
    b, a = signal.butter(8, cut, btype="low", analog=False)

    # Initialize an array to hold filtered data
    filtered_data = xr.DataArray(
        np.zeros_like(ecp_data), coords=ecp_data.coords, dims=ecp_data.dims
    )

    # Apply the filter to each channel
    for channel in ecp_data.channel_id:
        filtered_data.loc[channel, :] = signal.filtfilt(
            b, a, ecp_data.sel(channel_id=channel).values
        )

    # Downsample the filtered data if a downsample frequency is provided
    if downsample_freq is not None:
        downsample_factor = int(fs / downsample_freq)
        filtered_data = filtered_data.isel(time=slice(None, None, downsample_factor))
        # Update the sampling frequency attribute
        filtered_data.attrs["fs"] = downsample_freq

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
        return xr.concat(slices, dim="time")
    else:
        return slices[0]


def fit_fooof(
    f: np.ndarray,
    pxx: np.ndarray,
    aperiodic_mode: str = "fixed",
    dB_threshold: float = 3.0,
    max_n_peaks: int = 10,
    freq_range: tuple = None,
    peak_width_limits: tuple = None,
    report: bool = False,
    plot: bool = False,
    plt_log: bool = False,
    plt_range: tuple = None,
    figsize: tuple = None,
    title: str = None,
) -> tuple:
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
    if aperiodic_mode != "knee":
        aperiodic_mode = "fixed"

    def set_range(x, upper=f[-1]):
        x = np.array(upper) if x is None else np.array(x)
        return [f[2], x.item()] if x.size == 1 else x.tolist()

    freq_range = set_range(freq_range)
    peak_width_limits = set_range(peak_width_limits, np.inf)

    # Initialize a FOOOF object
    fm = FOOOF(
        peak_width_limits=peak_width_limits,
        min_peak_height=dB_threshold / 10,
        peak_threshold=0.0,
        max_n_peaks=max_n_peaks,
        aperiodic_mode=aperiodic_mode,
    )

    # Fit the model
    try:
        fm.fit(f, pxx, freq_range)
    except Exception as e:
        fl = np.linspace(f[0], f[-1], int((f[-1] - f[0]) / np.min(np.diff(f))) + 1)
        fm.fit(fl, np.interp(fl, f, pxx), freq_range)

    results = fm.get_results()

    if report:
        fm.print_results()
        if aperiodic_mode == "knee":
            ap_params = results.aperiodic_params
            if ap_params[1] <= 0:
                print(
                    "Negative value of knee parameter occurred. Suggestion: Fit without knee parameter."
                )
            knee_freq = np.abs(ap_params[1]) ** (1 / ap_params[2])
            print(f"Knee location: {knee_freq:.2f} Hz")

    if plot:
        plt_range = set_range(plt_range)
        fm.plot(plt_log=plt_log)
        plt.xlim(np.log10(plt_range) if plt_log else plt_range)
        # plt.ylim(-8, -5.5)
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
    full_fit, _, ap_fit = gen_model(
        fooof_model.freqs[1:],
        results.aperiodic_params,
        results.gaussian_params,
        return_components=True,
    )

    full_fit, ap_fit = 10**full_fit, 10**ap_fit  # Convert back from log
    res_psd = np.insert(
        (10 ** fooof_model.power_spectrum[1:]) - ap_fit, 0, 0.0
    )  # Convert back from log
    res_fit = np.insert(full_fit - ap_fit, 0, 0.0)
    ap_fit = np.insert(ap_fit, 0, 0.0)

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


def calculate_wavelet_passband(center_freq, bandwidth, threshold=0.3):
    """
    Calculate the passband of a complex Morlet wavelet filter.

    Parameters
    ----------
    center_freq : float
        Center frequency (Hz) of the wavelet filter
    bandwidth : float
        Bandwidth parameter of the wavelet filter
    threshold : float, optional
        Power threshold to define the passband edges (default: 0.5 = -3dB point)

    Returns
    -------
    tuple
        (lower_bound, upper_bound, passband_width) of the frequency passband in Hz
    """
    # Create a high-resolution frequency axis around the center frequency
    # Extend range to 3x the expected width to ensure we capture the full passband
    expected_width = center_freq * bandwidth / 2
    freq_min = max(0.1, center_freq - 3 * expected_width)
    freq_max = center_freq + 3 * expected_width
    freq_axis = np.linspace(freq_min, freq_max, 1000)

    # Calculate the theoretical frequency response of the Morlet wavelet
    # For a complex Morlet wavelet, the frequency response approximates a Gaussian
    # centered at the center frequency with width related to the bandwidth parameter
    sigma_f = bandwidth * center_freq / 8  # Approximate relationship for cmor wavelet
    response = np.exp(-((freq_axis - center_freq) ** 2) / (2 * sigma_f**2))

    # Find the passband edges (where response crosses the threshold)
    above_threshold = response >= threshold
    if not np.any(above_threshold):
        return (center_freq, center_freq, 0)  # No passband found

    # Find the first and last indices where response is above threshold
    indices = np.where(above_threshold)[0]
    lower_idx = indices[0]
    upper_idx = indices[-1]

    # Get the corresponding frequencies
    lower_bound = freq_axis[lower_idx]
    upper_bound = freq_axis[upper_idx]
    passband_width = upper_bound - lower_bound

    return (lower_bound, upper_bound, passband_width)


def wavelet_filter(
    x: np.ndarray,
    freq: float,
    fs: float,
    bandwidth: float = 1.0,
    axis: int = -1,
    show_passband: bool = False,
) -> np.ndarray:
    """
    Compute the Continuous Wavelet Transform (CWT) for a specified frequency using a complex Morlet wavelet.

    Parameters
    ----------
    x : np.ndarray
        Input signal
    freq : float
        Target frequency for the wavelet filter
    fs : float
        Sampling frequency of the signal
    bandwidth : float, optional
        Bandwidth parameter of the wavelet filter (default is 1.0)
    axis : int, optional
        Axis along which to compute the CWT (default is -1)
    show_passband : bool, optional
        If True, print the passband of the wavelet filter (default is False)

    Returns
    -------
    np.ndarray
        Continuous Wavelet Transform of the input signal
    """
    if show_passband:
        lower_bound, upper_bound, passband_width = calculate_wavelet_passband(
            freq, bandwidth, threshold=0.3
        )  # kinda made up threshold gives the rough idea
        print(f"Wavelet filter at {freq:.1f} Hz Bandwidth: {bandwidth:.1f} Hz:")
        print(
            f"  Passband: {lower_bound:.1f} - {upper_bound:.1f} Hz (width: {passband_width:.1f} Hz)"
        )
    wavelet = "cmor" + str(2 * bandwidth**2) + "-1.0"
    scale = pywt.scale2frequency(wavelet, 1) * fs / freq
    x_a = pywt.cwt(x, [scale], wavelet=wavelet, axis=axis)[0][0]
    return x_a


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5, axis: int = -1
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the input data.
    """
    sos = signal.butter(order, [lowcut, highcut], fs=fs, btype="band", output="sos")
    x_a = signal.sosfiltfilt(sos, data, axis=axis)
    return x_a


def get_lfp_power(
    lfp_data,
    freq_of_interest: float,
    fs: float,
    filter_method: str = "wavelet",
    lowcut: float = None,
    highcut: float = None,
    bandwidth: float = 1.0,
):
    """
    Compute the power of the raw LFP signal in a specified frequency band,
    preserving xarray structure if input is xarray.

    Parameters
    ----------
    lfp_data : np.ndarray or xr.DataArray
        Raw local field potential (LFP) time series data
    freq_of_interest : float
        Center frequency (Hz) for wavelet filtering method
    fs : float
        Sampling frequency (Hz) of the input data
    filter_method : str, optional
        Filtering method to use, either 'wavelet' or 'butter' (default: 'wavelet')
    lowcut : float, optional
        Lower frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    highcut : float, optional
        Upper frequency bound (Hz) for butterworth bandpass filter, required if filter_method='butter'
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 1.0)

    Returns
    -------
    np.ndarray or xr.DataArray
        Power of the filtered signal (magnitude squared) with same structure as input

    Notes
    -----
    - The 'wavelet' method uses a complex Morlet wavelet centered at the specified frequency
    - The 'butter' method uses a Butterworth bandpass filter with the specified cutoff frequencies
    - When using the 'butter' method, both lowcut and highcut must be provided
    - If input is an xarray DataArray, the output will preserve the same structure with coordinates
    """
    import xarray as xr

    # Check if input is xarray
    is_xarray = isinstance(lfp_data, xr.DataArray)

    if is_xarray:
        # Get the raw data from xarray
        raw_data = lfp_data.values
        # Check if 'fs' attribute exists in the xarray and override if necessary
        if "fs" in lfp_data.attrs and fs is None:
            fs = lfp_data.attrs["fs"]
    else:
        raw_data = lfp_data

    if filter_method == "wavelet":
        filtered_signal = wavelet_filter(raw_data, freq_of_interest, fs, bandwidth)
    elif filter_method == "butter":
        if lowcut is None or highcut is None:
            raise ValueError(
                "Both lowcut and highcut must be specified when using 'butter' method."
            )
        filtered_signal = butter_bandpass_filter(raw_data, lowcut, highcut, fs)
    else:
        raise ValueError("Invalid method. Choose 'wavelet' or 'butter'.")

    # Calculate power (magnitude squared of filtered signal)
    power = np.abs(filtered_signal) ** 2

    # If the input was an xarray, return an xarray with the same coordinates
    if is_xarray:
        power_xarray = xr.DataArray(
            power,
            coords=lfp_data.coords,
            dims=lfp_data.dims,
            attrs={
                **lfp_data.attrs,
                "filter_method": filter_method,
                "frequency_of_interest": freq_of_interest,
                "bandwidth": bandwidth,
                "lowcut": lowcut,
                "highcut": highcut,
                "power_type": "magnitude_squared",
            },
        )
        return power_xarray

    return power


def get_lfp_phase(
    lfp_data,
    freq_of_interest: float,
    fs: float,
    filter_method: str = "wavelet",
    lowcut: float = None,
    highcut: float = None,
    bandwidth: float = 1.0,
) -> np.ndarray:
    """
    Calculate the phase of the filtered signal, preserving xarray structure if input is xarray.

    Parameters
    ----------
    lfp_data : np.ndarray or xr.DataArray
        Input LFP data
    freq_of_interest : float
        Frequency of interest (Hz)
    fs : float
        Sampling frequency (Hz)
    filter_method : str, optional
        Method for filtering the signal ('wavelet' or 'butter')
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter when method='wavelet' (default: 1.0)
    lowcut : float, optional
        Low cutoff frequency for Butterworth filter when method='butter'
    highcut : float, optional
        High cutoff frequency for Butterworth filter when method='butter'

    Returns
    -------
    np.ndarray or xr.DataArray
        Phase of the filtered signal with same structure as input

    Notes
    -----
    - The 'wavelet' method uses a complex Morlet wavelet centered at the specified frequency
    - The 'butter' method uses a Butterworth bandpass filter with the specified cutoff frequencies
      followed by Hilbert transform to extract the phase
    - When using the 'butter' method, both lowcut and highcut must be provided
    - If input is an xarray DataArray, the output will preserve the same structure with coordinates
    """
    import xarray as xr

    # Check if input is xarray
    is_xarray = isinstance(lfp_data, xr.DataArray)

    if is_xarray:
        # Get the raw data from xarray
        raw_data = lfp_data.values
        # Check if 'fs' attribute exists in the xarray and override if necessary
        if "fs" in lfp_data.attrs and fs is None:
            fs = lfp_data.attrs["fs"]
    else:
        raw_data = lfp_data

    if filter_method == "wavelet":
        if freq_of_interest is None:
            raise ValueError("freq_of_interest must be provided for the wavelet method.")
        # Wavelet filter returns complex values directly
        filtered_signal = wavelet_filter(raw_data, freq_of_interest, fs, bandwidth)
        # Phase is the angle of the complex signal
        phase = np.angle(filtered_signal)
    elif filter_method == "butter":
        if lowcut is None or highcut is None:
            raise ValueError(
                "Both lowcut and highcut must be specified when using 'butter' method."
            )
        # Butterworth filter returns real values
        filtered_signal = butter_bandpass_filter(raw_data, lowcut, highcut, fs)
        # Apply Hilbert transform to get analytic signal (complex)
        analytic_signal = signal.hilbert(filtered_signal)
        # Phase is the angle of the analytic signal
        phase = np.angle(analytic_signal)
    else:
        raise ValueError(f"Invalid method {filter_method}. Choose 'wavelet' or 'butter'.")

    # If the input was an xarray, return an xarray with the same coordinates
    if is_xarray:
        phase_xarray = xr.DataArray(
            phase,
            coords=lfp_data.coords,
            dims=lfp_data.dims,
            attrs={
                **lfp_data.attrs,
                "filter_method": filter_method,
                "freq_of_interest": freq_of_interest,
                "bandwidth": bandwidth,
                "lowcut": lowcut,
                "highcut": highcut,
            },
        )
        return phase_xarray

    return phase


# windowing functions
def windowed_xarray(da, windows, dim="time", new_coord_name="cycle", new_coord=None):
    """Divide xarray into windows of equal size along an axis
    da: input DataArray
    windows: 2d-array of windows
    dim: dimension along which to divide
    new_coord_name: name of new dimemsion along which to concatenate windows
    new_coord: pandas Index object of new coordinates. Defaults to integer index
    """
    win_da = [da.sel({dim: slice(*w)}) for w in windows]
    n_win = min(x.coords[dim].size for x in win_da)
    idx = {dim: slice(n_win)}
    coords = da.coords[dim].isel(idx).coords
    win_da = [x.isel(idx).assign_coords(coords) for x in win_da]
    if new_coord is None:
        new_coord = pd.Index(range(len(win_da)), name=new_coord_name)
    win_da = xr.concat(win_da, dim=new_coord)
    return win_da


def group_windows(win_da, win_grp_idx={}, win_dim="cycle"):
    """Group windows into a dictionary of DataArrays
    win_da: input windowed DataArrays
    win_grp_idx: dictionary of {window group id: window indices}
    win_dim: dimension for different windows
    Return: dictionaries of {window group id: DataArray of grouped windows}
        win_on / win_off for windows selected / not selected by `win_grp_idx`
    """
    win_on, win_off = {}, {}
    for g, w in win_grp_idx.items():
        win_on[g] = win_da.sel({win_dim: w})
        win_off[g] = win_da.drop_sel({win_dim: w})
    return win_on, win_off


def average_group_windows(win_da, win_dim="cycle", grp_dim="unique_cycle"):
    """Average over windows in each group and stack groups in a DataArray
    win_da: input dictionary of {window group id: DataArray of grouped windows}
    win_dim: dimension for different windows
    grp_dim: dimension along which to stack average of window groups
    """
    win_avg = {
        g: xr.concat(
            [x.mean(dim=win_dim), x.std(dim=win_dim)], pd.Index(("mean_", "std_"), name="stats")
        )
        for g, x in win_da.items()
    }
    win_avg = xr.concat(win_avg.values(), dim=pd.Index(win_avg.keys(), name=grp_dim))
    win_avg = win_avg.to_dataset(dim="stats")
    return win_avg


# used for avg spectrogram across different trials
def get_windowed_data(
    x, windows, win_grp_idx, dim="time", win_dim="cycle", win_coord=None, grp_dim="unique_cycle"
):
    """Apply functions of windowing to data
    x: DataArray
    windows: `windows` for `windowed_xarray`
    win_grp_idx: `win_grp_idx` for `group_windows`
    dim: dimension along which to divide
    win_dim: dimension for different windows
    win_coord: pandas Index object of `win_dim` coordinates
    grp_dim: dimension along which to stack average of window groups.
        If None or empty or False, do not calculate average.
    Return: data returned by three functions,
        `windowed_xarray`, `group_windows`, `average_group_windows`
    """
    x_win = windowed_xarray(x, windows, dim=dim, new_coord_name=win_dim, new_coord=win_coord)
    x_win_onff = group_windows(x_win, win_grp_idx, win_dim=win_dim)
    if grp_dim:
        x_win_avg = [average_group_windows(x, win_dim=win_dim, grp_dim=grp_dim) for x in x_win_onff]
    else:
        x_win_avg = None
    return x_win, x_win_onff, x_win_avg


# cone of influence in frequency for cmorxx-1.0 wavelet. need to add logic to calculate in function
f0 = 2 * np.pi
CMOR_COI = 2**-0.5
CMOR_FLAMBDA = 4 * np.pi / (f0 + (2 + f0**2) ** 0.5)
COI_FREQ = 1 / (CMOR_COI * CMOR_FLAMBDA)


def cwt_spectrogram(
    x,
    fs,
    nNotes=6,
    nOctaves=np.inf,
    freq_range=(0, np.inf),
    bandwidth=1.0,
    axis=-1,
    detrend=False,
    normalize=False,
):
    """Calculate spectrogram using continuous wavelet transform"""
    x = np.asarray(x)
    N = x.shape[axis]
    times = np.arange(N) / fs
    # detrend and normalize
    if detrend:
        x = signal.detrend(x, axis=axis, type="linear")
    if normalize:
        x = x / x.std()
    # Define some parameters of our wavelet analysis.
    # range of scales (in time) that makes sense
    # min = 2 (Nyquist frequency)
    # max = np.floor(N/2)
    nOctaves = min(nOctaves, np.log2(2 * np.floor(N / 2)))
    scales = 2 ** np.arange(1, nOctaves, 1 / nNotes)
    # cwt and the frequencies used.
    # Use the complex morelet with bw=2*bandwidth^2 and center frequency of 1.0
    # bandwidth is sigma of the gaussian envelope
    wavelet = "cmor" + str(2 * bandwidth**2) + "-1.0"
    frequencies = pywt.scale2frequency(wavelet, scales) * fs
    scales = scales[(frequencies >= freq_range[0]) & (frequencies <= freq_range[1])]
    coef, frequencies = pywt.cwt(
        x, scales[::-1], wavelet=wavelet, sampling_period=1 / fs, axis=axis
    )
    power = np.real(coef * np.conj(coef))  # equivalent to power = np.abs(coef)**2
    # cone of influence in terms of wavelength
    coi = N / 2 - np.abs(np.arange(N) - (N - 1) / 2)
    # cone of influence in terms of frequency
    coif = COI_FREQ * fs / coi
    return power, times, frequencies, coif


def cwt_spectrogram_xarray(
    x, fs, time=None, axis=-1, downsample_fs=None, channel_coords=None, **cwt_kwargs
):
    """Calculate spectrogram using continuous wavelet transform and return an xarray.Dataset
    x: input array
    fs: sampling frequency (Hz)
    axis: dimension index of time axis in x
    downsample_fs: downsample to the frequency if specified
    channel_coords: dictionary of {coordinate name: index} for channels
    cwt_kwargs: keyword arguments for cwt_spectrogram()
    """
    x = np.asarray(x)
    T = x.shape[axis]  # number of time points
    t = np.arange(T) / fs if time is None else np.asarray(time)
    if downsample_fs is None or downsample_fs >= fs:
        downsample_fs = fs
        downsampled = x
    else:
        num = int(T * downsample_fs / fs)
        downsample_fs = num / T * fs
        downsampled, t = signal.resample(x, num=num, t=t, axis=axis)
    downsampled = np.moveaxis(downsampled, axis, -1)
    sxx, _, f, coif = cwt_spectrogram(downsampled, downsample_fs, **cwt_kwargs)
    sxx = np.moveaxis(sxx, 0, -2)  # shape (... , freq, time)
    if channel_coords is None:
        channel_coords = {f"dim_{i:d}": range(d) for i, d in enumerate(sxx.shape[:-2])}
    sxx = xr.DataArray(sxx, coords={**channel_coords, "frequency": f, "time": t}).to_dataset(
        name="PSD"
    )
    sxx.update(dict(cone_of_influence_frequency=xr.DataArray(coif, coords={"time": t})))
    return sxx
