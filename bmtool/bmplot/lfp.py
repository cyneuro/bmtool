import matplotlib.pyplot as plt
import numpy as np
from fooof.sim.gen import gen_aperiodic
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
from ..analysis.spikes import get_population_spike_rate
from ..analysis.lfp import get_lfp_power, load_ecp_to_xarray, ecp_to_lfp
from matplotlib.figure import Figure


def plot_spectrogram(
    sxx_xarray: Any,
    remove_aperiodic: Optional[Any] = None,
    log_power: bool = False,
    plt_range: Optional[Tuple[float, float]] = None,
    clr_freq_range: Optional[Tuple[float, float]] = None,
    pad: float = 0.03,
    ax: Optional[plt.Axes] = None,
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
    if clr_freq_range is None:
        vmin, vmax = None, None
    else:
        c_idx = (f >= clr_freq_range[0]) & (f <= clr_freq_range[1])
        vmin, vmax = sxx[c_idx, :].min(), sxx[c_idx, :].max()

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
    freq_of_interest: List[float],
    freq_labels: List[str],
    freq_colors: List[str],
    time_range: Tuple[float, float],
    pop_names: List[str],
    pop_color: Dict[str, str],
    trial_path: str,
    filter_column: Optional[str] = None,
    filter_value: Optional[Any] = None,
) -> Optional[Figure]:
    """
    Plot population spike rates with LFP power overlays.

    Parameters
    ----------
    spikes_df : pd.DataFrame
        DataFrame with spike data.
    freq_of_interest : list of float
        List of frequencies for LFP power analysis (required).
    freq_labels : list of str
        Labels for the frequencies (required).
    freq_colors : list of str
        Colors for the frequency plots (required).
    time_range : tuple of float
        Tuple (start, end) for x-axis time limits (required).
    pop_names : list of str
        List of population names (required).
    pop_color : dict
        Dictionary mapping population names to colors (required).
    trial_path : str
        Path to trial data (required).
    filter_column : str, optional
        Column name to filter spikes_df on (optional).
    filter_value : any, optional
        Value to filter for in filter_column (optional).

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object containing the plot, or None if no data to plot.

    Examples
    --------
    >>> fig = plot_population_spike_rates_with_lfp(
    ...     spikes_df, [40, 80], ['Beta', 'Gamma'],
    ...     ['blue', 'red'], (0, 10), ['PV', 'SST'],
    ...     {'PV': 'blue', 'SST': 'red'}, 'trial_data.h5'
    ... )
    """
    # Compute spike rates based on filtering
    if filter_column and filter_column in spikes_df.columns:
        filtered_df = spikes_df[spikes_df[filter_column] == filter_value]
        if not filtered_df.empty:
            spike_rate = get_population_spike_rate(filtered_df, fs=400, network_name='cortex')
            plot_title = f'{filter_column} {filter_value}'
            save_suffix = f'_{filter_column}_{filter_value}'
        else:
            print(f"No data found for {filter_column} == {filter_value}.")
            return
    else:
        spike_rate = get_population_spike_rate(spikes_df, fs=400, network_name='cortex')
        plot_title = 'Overall Spike Rates'
        save_suffix = '_overall'

    # Load LFP data and compute power for each frequency of interest
    ecp = load_ecp_to_xarray(ecp_file=trial_path + "/ecp.h5")
    lfp = ecp_to_lfp(ecp)
    powers = [
        get_lfp_power(lfp, freq_of_interest=freq, fs=lfp.fs, filter_method="wavelet", bandwidth=1.0)
        for freq in freq_of_interest
    ]
    
    # Plotting
    fig, axes = plt.subplots(len(spike_rate.population), 1, figsize=(12, 10))
    for i, pop in enumerate(pop_names):
        if pop in spike_rate.population.values:
            ax = axes.flat[i]
            spike_rate.sel(type='raw', population=pop).plot(ax=ax, color=pop_color[pop])
            ax.set_title(f'{pop}')
            ax.set_ylabel('Spike Rate (Hz)', color=pop_color[pop])
            ax.tick_params(axis='y', labelcolor=pop_color[pop])
            
            # Twin axis for LFP power
            ax2 = ax.twinx()
            for power, label, color in zip(powers, freq_labels, freq_colors):
                ax2.plot(power['time'], power.values.squeeze(), color=color, label=label)
            ax2.set_ylabel('LFP Power', color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            ax2.legend(loc='upper right')
            
            ax.set_xlim(time_range)
    
    fig.suptitle(plot_title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
