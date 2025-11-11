import matplotlib.pyplot as plt
import numpy as np
from fooof.sim.gen import gen_aperiodic
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
from ..analysis.spikes import get_population_spike_rate
from ..analysis.lfp import get_lfp_power
from matplotlib.figure import Figure


def plot_spectrogram(
    sxx_xarray: Any,
    remove_aperiodic: Optional[Any] = None,
    log_power: bool = False,
    plt_range: Optional[Tuple[float, float]] = None,
    clr_freq_range: Optional[Tuple[float, float]] = None,
    pad: float = 0.03,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Figure:
    """
    Plot a power spectrogram with optional aperiodic removal and frequency-based coloring.

    Parameters
    ----------
    sxx_xarray : array-like
        Spectrogram data as an xarray DataArray with PSD values.
    remove_aperiodic : optional
        FOOOF model object for aperiodic subtraction. If None, raw spectrum is displayed.
    log_power : bool or str, optional
        If True or 'dB', convert power to log scale. Default is False.
    plt_range : tuple of float, optional
        Frequency range to display as (f_min, f_max). If None, displays full range.
    clr_freq_range : tuple of float, optional
        Frequency range to use for determining color limits. If None, uses full range.
    pad : float, optional
        Padding for colorbar. Default is 0.03.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    vmin : float, optional
        Minimum value for colorbar scaling. If None, computed from data.
    vmax : float, optional
        Maximum value for colorbar scaling. If None, computed from data.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the spectrogram.

    Examples
    --------
    >>> fig = plot_spectrogram(
    ...     sxx_xarray, log_power='dB',
    ...     plt_range=(10, 100), clr_freq_range=(20, 50)
    ... )
    """
    sxx = sxx_xarray.PSD.values.copy()
    t = sxx_xarray.time.values.copy()
    f = sxx_xarray.frequency.values.copy()

    cbar_label = "PSD" if remove_aperiodic is None else "PSD Residual"
    if log_power:
        with np.errstate(divide="ignore"):
            sxx = np.log10(sxx)
        cbar_label += " dB" if log_power == "dB" else " log(power)"

    if remove_aperiodic is not None:
        f1_idx = 0 if f[0] else 1
        ap_fit = gen_aperiodic(f[f1_idx:], remove_aperiodic.aperiodic_params)
        sxx[f1_idx:, :] -= (ap_fit if log_power else 10**ap_fit)[:, None]
        sxx[:f1_idx, :] = 0.0

    if log_power == "dB":
        sxx *= 10

    if ax is None:
        _, ax = plt.subplots(1, 1)
    plt_range = np.array(f[-1]) if plt_range is None else np.array(plt_range)
    if plt_range.size == 1:
        plt_range = [f[0 if f[0] else 1] if log_power else 0.0, plt_range.item()]
    f_idx = (f >= plt_range[0]) & (f <= plt_range[1])
    
    # Determine vmin and vmax: explicit parameters take precedence, then clr_freq_range, then None
    if vmin is None:
        if clr_freq_range is not None:
            c_idx = (f >= clr_freq_range[0]) & (f <= clr_freq_range[1])
            vmin = sxx[c_idx, :].min()
    
    if vmax is None:
        if clr_freq_range is not None:
            c_idx = (f >= clr_freq_range[0]) & (f <= clr_freq_range[1])
            vmax = sxx[c_idx, :].max()

    f = f[f_idx]
    pcm = ax.pcolormesh(t, f, sxx[f_idx, :], shading="gouraud", vmin=vmin, vmax=vmax, rasterized=True)
    if "cone_of_influence_frequency" in sxx_xarray:
        coif = sxx_xarray.cone_of_influence_frequency
        ax.plot(t, coif)
        ax.fill_between(t, coif, step="mid", alpha=0.2)
    ax.set_xlim(t[0], t[-1])
    # ax.set_xlim(t[0],0.2)
    ax.set_ylim(f[0], f[-1])
    plt.colorbar(mappable=pcm, ax=ax, label=cbar_label, pad=pad)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Frequency (Hz)")
    return ax.figure


def plot_population_spike_rates_with_lfp(
    spikes_df: pd.DataFrame,
    lfp: Any,
    freq_of_interest: List[float],
    freq_labels: List[str],
    freq_colors: List[str],
    time_range: Any,
    pop_names: List[str],
    pop_color: Dict[str, str],
    pop_groups: Optional[List[List[str]]] = None,
    FR_type: str = 'smoothed',
    stimulus_time: Optional[float] = None,
) -> Optional[Figure]:
    """
    Plot population spike rates with LFP power overlays, with optional trial averaging.

    Parameters
    ----------
    spikes_df : pd.DataFrame
        DataFrame with spike data.
    lfp : array-like
        LFP data (xarray or similar format).
    freq_of_interest : list of float
        List of frequencies for LFP power analysis (required).
    freq_labels : list of str
        Labels for the frequencies (required).
    freq_colors : list of str
        Colors for the frequency plots (required).
    time_range : tuple of float or list of tuple
        If tuple (start, end): plots continuous data in that time range.
        If list of tuples: trial times for averaging. E.g., [(1000,2000), (2500,3500)].
        For trial averaging, mean is computed across trials (required).
    pop_names : list of str
        List of population names (required).
    pop_color : dict
        Dictionary mapping population names to colors (required).
    pop_groups : list of list of str, optional
        List of population groups to plot on the same subplot. 
        E.g., [['PV', 'SST'], ['ET', 'IT']] plots PV and SST on one plot, ET and IT on another.
        If None, each population gets its own subplot (default).
    FR_type : str, optional
        Type of firing rate to plot ('raw', 'smoothed', etc.). Default is 'smoothed'.
    stimulus_time : float, optional
        Time of stimulus onset. 
        For trial averaging: relative to the start of the trial window (e.g., stimulus_time=200 means 200ms after trial start).
        For continuous plots: absolute time value (e.g., stimulus_time=2500 means stimulus at 2500ms).
        When provided, the x-axis will be relative to stimulus time (0 = stimulus onset). Default is None.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object containing the plot, or None if no data to plot.

    Examples
    --------
    >>> # Continuous plot
    >>> fig = plot_population_spike_rates_with_lfp(
    ...     spikes_df, lfp, [40, 80], ['Beta', 'Gamma'],
    ...     ['blue', 'red'], (0, 10), ['PV', 'SST'],
    ...     {'PV': 'blue', 'SST': 'red'},
    ...     pop_groups=[['PV', 'SST']]
    ... )
    
    >>> # Trial-averaged plot
    >>> fig = plot_population_spike_rates_with_lfp(
    ...     spikes_df, lfp, [40, 80], ['Beta', 'Gamma'],
    ...     ['blue', 'red'], [(1000,2000), (2500,3500)], ['PV', 'SST'],
    ...     {'PV': 'blue', 'SST': 'red'},
    ...     pop_groups=[['PV', 'SST']]
    ... )
    """
    # Compute spike rates
    spike_rate = get_population_spike_rate(spikes_df, fs=400, network_name='cortex')

    # Compute power for each frequency of interest
    powers = [
        get_lfp_power(lfp, freq_of_interest=freq, fs=lfp.fs, filter_method="wavelet", bandwidth=1.0)
        for freq in freq_of_interest
    ]
    
    # Determine if we're doing trial averaging
    is_trial_avg = isinstance(time_range, list) and len(time_range) > 0 and isinstance(time_range[0], tuple)
    
    # Extract and align trials if needed
    spike_rate_trials: Optional[List] = None
    power_trials: Optional[List] = None
    target_length: Optional[int] = None
    trial_start: float = 0.0
    trial_duration: float = 0.0
    if is_trial_avg:
        spike_rate_trials, power_trials, trial_times = _extract_trials(
            spike_rate, powers, time_range
        )
        # trial_times from _extract_trials is normalized (0 to 1)
        # Convert to actual milliseconds based on first trial duration
        trial_start = float(time_range[0][0])
        trial_end = float(time_range[0][1])
        trial_duration = trial_end - trial_start
        
        # Convert normalized times to milliseconds
        plot_time = trial_times * trial_duration
        
        # Adjust for stimulus if provided (stimulus_time is relative to trial start)
        if stimulus_time is not None:
            plot_time = plot_time - stimulus_time
        
        target_length = len(trial_times)
    else:
        # For continuous plots, time_range is a tuple (start, end)
        # We'll just pass the time_range for now; actual shifting happens during plotting
        plot_time = time_range
    
    # Determine plot groups
    if pop_groups is None:
        # Default: each population gets its own subplot
        plot_groups = [[pop] for pop in pop_names]
    else:
        plot_groups = pop_groups
    
    # Plotting
    num_subplots = len(plot_groups)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 3.5 * num_subplots))
    if num_subplots == 1:
        axes = [axes]
    
    for ax_idx, group in enumerate(plot_groups):
        ax = axes[ax_idx]
        
        # Filter valid populations in this group
        valid_pops = [pop for pop in group if pop in spike_rate.population.values]
        
        if not valid_pops:
            continue
        
        # Plot spike rates for each population in the group
        fr_handles = []
        if is_trial_avg and spike_rate_trials is not None:
            # Plot trial-averaged firing rates with SEM shading
            for pop in valid_pops:
                fr_mean, fr_sem = _compute_trial_average(spike_rate_trials, pop, FR_type, target_length=target_length)
                line, = ax.plot(plot_time, fr_mean,
                               color=pop_color[pop], 
                               label=f'{pop} FR',
                               linewidth=2)
                ax.fill_between(plot_time, fr_mean - fr_sem, fr_mean + fr_sem,
                               color=pop_color[pop], alpha=0.2)
                fr_handles.append(line)
        else:
            # Plot continuous firing rates
            plot_time_values = spike_rate.time.values
            if stimulus_time is not None and not is_trial_avg:
                # Shift time axis relative to stimulus
                plot_time_values = plot_time_values - stimulus_time
            
            for pop in valid_pops:
                line, = ax.plot(plot_time_values, 
                               spike_rate.sel(type=FR_type, population=pop).values,
                               color=pop_color[pop], 
                               label=f'{pop} FR',
                               linewidth=2)
                fr_handles.append(line)
        
        # Set labels and title
        group_title = ' + '.join(valid_pops)
        avg_text = ' (Trial Avg)' if is_trial_avg else ''
        ax.set_title(group_title + avg_text, fontsize=12)
        ax.set_ylabel('Spike Rate (Hz)', fontsize=11)
        ax.tick_params(axis='y')
        
        # Twin axis for LFP power
        ax2 = ax.twinx()
        lfp_handles = []
        if is_trial_avg and power_trials is not None:
            # Plot trial-averaged LFP power with SEM shading
            for power_trial, label, color in zip(power_trials, freq_labels, freq_colors):
                power_mean, power_sem = _compute_trial_average_power(power_trial, target_length=target_length)
                line, = ax2.plot(plot_time, power_mean,
                                color=color, label=label, linestyle='--', linewidth=2)
                ax2.fill_between(plot_time, power_mean - power_sem, power_mean + power_sem,
                                color=color, alpha=0.1)
                lfp_handles.append(line)
        else:
            # Plot continuous LFP power
            for power, label, color in zip(powers, freq_labels, freq_colors):
                plot_time_lfp = power['time'].values
                if stimulus_time is not None and not is_trial_avg:
                    # Shift time axis relative to stimulus
                    plot_time_lfp = plot_time_lfp - stimulus_time
                
                line, = ax2.plot(plot_time_lfp, power.values.squeeze(), 
                                color=color, label=label, linestyle='--', linewidth=2)
                lfp_handles.append(line)
        
        ax2.set_ylabel('LFP Power', fontsize=11)
        ax2.tick_params(axis='y')
        
        # Combined legend
        all_handles = fr_handles + lfp_handles
        all_labels = [h.get_label() for h in all_handles]
        ax.legend(all_handles, all_labels, loc='upper right', fontsize=10)
        
        if is_trial_avg:
            ax.set_xlim(plot_time[0], plot_time[-1])
            if stimulus_time is not None:
                ax.set_xlabel('Time relative to stimulus (ms)', fontsize=11)
            else:
                ax.set_xlabel('Time from trial start (ms)', fontsize=11)
        else:
            # For continuous plots
            if stimulus_time is not None:
                # Shift xlim by stimulus time
                xlim = (plot_time[0] - stimulus_time, plot_time[1] - stimulus_time)
                ax.set_xlim(xlim)
                ax.set_xlabel('Time relative to stimulus (ms)', fontsize=11)
            else:
                ax.set_xlim(plot_time)
                ax.set_xlabel('Time (ms)', fontsize=11)
    
    plt.tight_layout()
    return fig


def _extract_trials(spike_rate: Any, powers: List[Any], time_range: List[Tuple[float, float]]) -> Tuple[List, List, np.ndarray]:
    """
    Extract data segments for each trial and align them.
    
    Parameters
    ----------
    spike_rate : xarray.DataArray
        Spike rate data with time dimension.
    powers : list of xarray.DataArray
        LFP power data for each frequency.
    time_range : list of tuple
        List of (start, end) tuples for each trial.
    
    Returns
    -------
    spike_rate_trials : list of arrays
        Firing rate data for each trial, shape (n_trials, n_populations, n_timepoints).
    power_trials : list of list of arrays
        Power data for each trial and frequency.
    trial_times : np.ndarray
        Time array for a single trial (relative to trial start).
    """
    spike_rate_trials = []
    power_trials = [[] for _ in powers]
    trial_lengths = []
    
    # Extract data for each trial and track actual trial lengths
    for trial_start, trial_end in time_range:
        # Extract spike rate for this trial
        sr_trial = spike_rate.sel(time=slice(trial_start, trial_end))
        if len(sr_trial.time) == 0:
            continue
        spike_rate_trials.append(sr_trial)
        trial_lengths.append(len(sr_trial.time))
        
        # Extract power for this trial
        for i, power in enumerate(powers):
            p_trial = power.sel(time=slice(trial_start, trial_end))
            if len(p_trial.time) > 0:
                power_trials[i].append(p_trial.values.squeeze())
    
    # Use the maximum trial length to create a common time axis and resample all data
    if trial_lengths:
        max_length = max(trial_lengths)
        # Create a normalized time axis (0 to 1) for interpolation
        trial_times = np.linspace(0, 1, max_length)
    else:
        trial_times = np.array([])
    
    return spike_rate_trials, power_trials, trial_times


def _compute_trial_average(spike_rate_trials: List[Any], pop_name: str, data_type: str = 'smoothed', target_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and SEM of firing rates across trials, resampling to a common length.
    
    Parameters
    ----------
    spike_rate_trials : list of xarray.DataArray
        Spike rate data for each trial.
    pop_name : str
        Population name to extract.
    data_type : str
        Type of data ('raw', 'smoothed', etc.). Default is 'smoothed'.
    target_length : int, optional
        Target length for resampling. If None, uses max length.
    
    Returns
    -------
    mean : np.ndarray
        Mean firing rate across trials.
    sem : np.ndarray
        Standard error of the mean across trials.
    """
    # Extract data for this population from each trial
    trial_data = []
    for sr_trial in spike_rate_trials:
        if pop_name in sr_trial.population.values:
            pop_data = sr_trial.sel(type=data_type, population=pop_name).values
            trial_data.append(pop_data)
    
    if not trial_data:
        return np.array([]), np.array([])
    
    # Determine target length if not provided
    if target_length is None:
        target_length = max(len(d) for d in trial_data)
    
    # Resample all trials to target length
    resampled_data = []
    for data in trial_data:
        if len(data) != target_length:
            # Resample using linear interpolation
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, target_length)
            resampled = np.interp(x_new, x_old, data)
        else:
            resampled = data
        resampled_data.append(resampled)
    
    trial_array = np.array(resampled_data)
    mean = np.mean(trial_array, axis=0)
    sem = np.std(trial_array, axis=0) / np.sqrt(len(trial_array))
    
    return mean, sem


def _compute_trial_average_power(power_trials: List[np.ndarray], target_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and SEM of LFP power across trials, resampling to a common length.
    
    Parameters
    ----------
    power_trials : list of np.ndarray
        Power data for each trial.
    target_length : int, optional
        Target length for resampling. If None, uses max length.
    
    Returns
    -------
    mean : np.ndarray
        Mean power across trials.
    sem : np.ndarray
        Standard error of the mean across trials.
    """
    if not power_trials:
        return np.array([]), np.array([])
    
    # Determine target length if not provided
    if target_length is None:
        target_length = max(len(p) for p in power_trials)
    
    # Resample all trials to target length
    resampled_power = []
    for power in power_trials:
        if len(power) != target_length:
            # Resample using linear interpolation
            x_old = np.linspace(0, 1, len(power))
            x_new = np.linspace(0, 1, target_length)
            resampled = np.interp(x_new, x_old, power)
        else:
            resampled = power
        resampled_power.append(resampled)
    
    power_array = np.array(resampled_power)
    mean = np.mean(power_array, axis=0)
    sem = np.std(power_array, axis=0) / np.sqrt(len(power_array))
    
    return mean, sem


def plot_spike_rate_coherence(
    spike_rates: Any,
    fooof_params: Optional[Dict] = None,
    plt_range: Optional[Tuple[float, float]] = None,
    plt_log: bool = False,
    plt_db: bool = True,
    figsize: Tuple[int, int] = (10, 3),
    ax: Optional[plt.Axes] = None,
) -> Figure:
    """
    Plot coherence between spike rate populations.
    
    Computes coherence exactly like Analyze_PSD_ziao notebook: calculates coherence
    between population pairs and applies FOOOF fitting to coherence spectra.
    
    Parameters
    ----------
    spike_rates : xr.DataArray
        Spike rate data with dimensions (population, time) and 'fs' attribute
    fooof_params : dict, optional
        Parameters for FOOOF fitting. If None, uses default parameters
    plt_range : tuple, optional
        Frequency range to display (default: [2, 100])
    plt_log : bool
        Use log scale for frequency axis, default: False
    plt_db : bool
        Plot power in dB, default: True
    figsize : tuple
        Figure size, default: (10, 3)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the coherence plots
    
    Examples
    --------
    >>> fig = plot_spike_rate_coherence(spike_rates=spike_rate_data)
    """
    from scipy import signal
    from ..analysis.lfp import fit_fooof
    
    # Extract fs from spike_rates attributes
    if not hasattr(spike_rates, 'attrs') or 'fs' not in spike_rates.attrs:
        raise ValueError("spike_rates must have 'fs' attribute")
    fs = spike_rates.attrs['fs']
    
    # Set default parameters
    if fooof_params is None:
        fooof_params = dict(aperiodic_mode='knee', freq_range=(1, 100), 
                           peak_width_limits=100., max_n_peaks=1, dB_threshold=0.05)
    
    if plt_range is None:
        plt_range = [2., 100.]
    
    # Get population pairs like in Analyze_PSD_ziao
    pop_names = spike_rates.population.values
    n_pops = len(pop_names)
    grp_pairs = [[i, j] for i in range(n_pops) for j in range(i+1, n_pops)]
    npairs = len(grp_pairs)
    
    if npairs == 0:
        raise ValueError("Need at least 2 populations for coherence analysis")
    
    # Create figure with max 3 plots per row
    if ax is None:
        ncols = min(npairs, 3)
        nrows = (npairs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1]))
    else:
        fig = ax.get_figure()
        axes = [ax]
    
    if npairs == 1:
        axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes
    
    # Calculate coherence for each pair
    for i, grp_pair in enumerate(grp_pairs):
        if isinstance(axes, np.ndarray):
            ax = axes.flat[i]
        else:
            ax = axes[i]
        
        pop1_name = pop_names[grp_pair[0]]
        pop2_name = pop_names[grp_pair[1]]
        
        # Get spike rate data for the pair
        signal1 = spike_rates.sel(type='smoothed', population=pop1_name).values
        signal2 = spike_rates.sel(type='smoothed', population=pop2_name).values
        
        # Check if both populations have non-zero std
        if np.std(signal1) == 0 or np.std(signal2) == 0:
            ax.text(0.5, 0.5, 'No variation in data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'Coherence {pop1_name}-{pop2_name}')
            continue
        
        # Calculate coherence over entire time series
        f, cxy = signal.coherence(signal1, signal2, fs=fs)
        
        # Filter valid coherence values (positive and not NaN)
        idx = (cxy > 0) & np.isfinite(cxy)
        if not np.any(idx):
            ax.text(0.5, 0.5, 'No valid coherence', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'Coherence {pop1_name}-{pop2_name}')
            continue
        
        # Apply FOOOF to coherence (exactly like Analyze_PSD_ziao)
        f_filtered = f[idx]
        cxy_filtered = cxy[idx]
        
        plt.sca(ax)
        fooof_results, fm = fit_fooof(f_filtered, cxy_filtered, **fooof_params, 
                                   report=False, plot=True)
        
        # Formatting like Analyze_PSD_ziao
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.set_title(f'Coherence {pop1_name}-{pop2_name}')
        ax.set_xlim(plt_range)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
