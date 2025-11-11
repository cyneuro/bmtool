from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy import stats

from bmtool.analysis import entrainment as bmentr
from bmtool.analysis import spikes as bmspikes
from bmtool.analysis.lfp import get_lfp_power


def calculate_trial_statistics(
    data: np.ndarray,
    error_type: str = "ci",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and error statistics across trials.

    Computes trial-averaged statistics with proper handling of NaN values. Supports
    three error types: 95% confidence intervals (via t-distribution), standard error
    of the mean (SEM), and standard deviation (SD).

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_trials, n_values) containing trial-wise data.
        Can contain NaN values which are ignored in calculations.
    error_type : str, optional
        Type of error to compute: "ci" for 95% confidence interval, "sem" for
        standard error of the mean, or "std" for standard deviation (default: "ci").

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - mean_values : 1D array of mean values across trials (one per value).
        - error_values : 1D array of error bounds/bars corresponding to each mean.
          For "ci", represents the half-width of the 95% confidence interval.
          For "sem" and "std", represents the error magnitude.

    Raises
    ------
    ValueError
        If error_type is not 'ci', 'sem', or 'std'.
        If data is not 2D or is empty.

    Notes
    -----
    - NaN values are ignored using numpy's nanmean and nanstd functions.
    - For "ci", uses the t-distribution with degrees of freedom = min(valid_counts) - 1.
    - Confidence intervals are computed at 95% (α = 0.05, two-tailed).
    - If fewer than 2 valid trials exist for a value, error is set to NaN.

    Examples
    --------
    >>> data = np.array([[1, 2, 3], [1.1, 2.2, 3.1], [0.9, 1.9, 3.2]])  # 3 trials, 3 values
    >>> mean, error = calculate_trial_statistics(data, error_type='ci')
    >>> print(mean, error)
    """
    if error_type not in ["ci", "sem", "std"]:
        raise ValueError(
            "error_type must be 'ci' for confidence interval, 'sem' for standard error, "
            "or 'std' for standard deviation."
        )

    if data.ndim != 2 or data.size == 0:
        raise ValueError("data must be a non-empty 2D array of shape (n_trials, n_values).")

    # Calculate mean across trials, ignoring NaNs
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_values = np.nanmean(data, axis=0)

        # Count valid trials per value (trials without NaN)
        valid_counts = np.sum(~np.isnan(data), axis=0)

        # Calculate standard deviation across trials, ignoring NaNs
        std_values = np.nanstd(data, axis=0, ddof=1)

        if error_type == "ci":
            # Calculate 95% confidence interval using t-distribution
            # SEM = std / sqrt(n_valid)
            sem_values = std_values / np.sqrt(np.maximum(valid_counts, 1))

            # Find the minimum valid count to use a conservative t-value
            # (all values use the same t-value for consistency)
            min_valid = np.min(valid_counts[valid_counts > 1])

            if min_valid > 1:
                # Two-tailed t-distribution at 95% confidence level
                t_value = stats.t.ppf(0.975, min_valid - 1)
                error_values = t_value * sem_values
            else:
                # Insufficient data for meaningful CI
                error_values = np.full_like(mean_values, np.nan)

        elif error_type == "sem":
            # Standard error of the mean
            sem_values = std_values / np.sqrt(np.maximum(valid_counts, 1))
            error_values = sem_values

        else:  # error_type == "std"
            # Standard deviation (no division by sqrt(n))
            error_values = std_values

    return mean_values, error_values


def plot_spike_power_correlation(
    spike_df: pd.DataFrame,
    lfp_data: xr.DataArray,
    fs: float,
    pop_names: List[str],
    filter_method: str = "wavelet",
    bandwidth: float = 2.0,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    freq_range: Tuple[float, float] = (10, 100),
    freq_step: float = 5,
    type_name: str = "raw",
    figsize: Tuple[float, float] = (12, 8),
) -> Figure:
    """
    Calculate and plot spike rate-LFP power correlation across frequencies for full signal.

    Analyzes the relationship between population spike rates and LFP power across a range
    of frequencies, using Spearman correlation for the entire signal duration.

    Parameters
    ----------
    spike_df : pd.DataFrame
        DataFrame containing spike data with columns 'timestamps', 'node_ids', and 'pop_name'.
    lfp_data : xr.DataArray
        LFP data with time dimension.
    fs : float
        Sampling frequency in Hz.
    pop_names : List[str]
        List of population names to analyze.
    filter_method : str, optional
        Filtering method: 'wavelet' or 'butter' (default: 'wavelet').
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter (default: 2.0).
    lowcut : float, optional
        Lower frequency bound (Hz) for butterworth filter. Required if filter_method='butter'.
    highcut : float, optional
        Upper frequency bound (Hz) for butterworth filter. Required if filter_method='butter'.
    freq_range : Tuple[float, float], optional
        Min and max frequency to analyze in Hz (default: (10, 100)).
    freq_step : float, optional
        Step size for frequency analysis in Hz (default: 5).
    type_name : str, optional
        Which type of spike rate to use (default: 'raw').
    figsize : Tuple[float, float], optional
        Figure size (width, height) in inches (default: (12, 8)).

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the correlation plot.

    Notes
    -----
    - Uses Spearman correlation (rank-based, robust to outliers).
    - Pre-computes LFP power at all frequencies for efficiency.

    Examples
    --------
    >>> fig = plot_spike_power_correlation(
    ...     spike_df=spike_df,
    ...     lfp_data=lfp,
    ...     fs=400,
    ...     pop_names=['PV', 'SST'],
    ...     freq_range=(10, 100),
    ...     freq_step=5
    ... )
    """
    # Compute spike rate for all spikes
    spike_rate = bmspikes.get_population_spike_rate(spike_df, fs=fs)

    # Setup frequencies for analysis
    frequencies = np.arange(freq_range[0], freq_range[1] + 1, freq_step)

    # Pre-calculate LFP power for all frequencies
    power_by_freq = {}
    for freq in frequencies:
        power_by_freq[freq] = get_lfp_power(
            lfp_data, freq, fs, filter_method, lowcut=lowcut, highcut=highcut, bandwidth=bandwidth
        )

    # Calculate correlations for each population and frequency
    results = {}
    for pop in pop_names:
        results[pop] = {}
        pop_spike_rate = spike_rate.sel(population=pop, type=type_name)

        for freq in frequencies:
            lfp_power = power_by_freq[freq]

            if len(pop_spike_rate) != len(lfp_power):
                print(f"Warning: Length mismatch for {pop} at {freq} Hz")
                print(f"{len(pop_spike_rate)} {len(lfp_power)}")
                continue

            corr, p_val = stats.spearmanr(pop_spike_rate.values, lfp_power.values)
            results[pop][freq] = {"correlation": corr, "p_value": p_val}

    # Create plot
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=figsize)

    colors = plt.get_cmap("tab10")
    for i, pop in enumerate(pop_names):
        plot_freqs = []
        plot_corrs = []

        for freq in frequencies:
            if freq in results[pop] and not np.isnan(results[pop][freq]["correlation"]):
                plot_freqs.append(freq)
                plot_corrs.append(results[pop][freq]["correlation"])

        if len(plot_freqs) == 0:
            continue

        plot_freqs = np.array(plot_freqs)
        plot_corrs = np.array(plot_corrs)
        color = colors(i)

        plt.plot(
            plot_freqs, plot_corrs, marker="o", label=pop, linewidth=2, markersize=6, color=color
        )

    # Formatting
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Spike Rate-Power Correlation", fontsize=12)

    plt.title(
        "Spike Rate-LFP Power Correlation",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    # Setup legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=colors(i), marker="o", linestyle="-", label=pop)
        for i, pop in enumerate(pop_names)
    ]
    plt.legend(handles=legend_elements, fontsize=10, loc="best")

    # Axis formatting
    if len(frequencies) > 10:
        plt.xticks(frequencies[::2])
    else:
        plt.xticks(frequencies)
    plt.xlim(frequencies[0], frequencies[-1])

    y_min, y_max = plt.ylim()
    plt.ylim(min(y_min, -0.1), max(y_max, 0.1))

    plt.tight_layout()
    return fig


def plot_trial_avg_spike_power_correlation(
    spike_df: pd.DataFrame,
    lfp_data: xr.DataArray,
    time_windows: List[Tuple[float, float]],
    fs: float,
    pop_names: List[str],
    filter_method: str = "wavelet",
    bandwidth: float = 2.0,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    freq_range: Tuple[float, float] = (10, 100),
    freq_step: float = 5,
    type_name: str = "raw",
    error_type: str = "ci",
    figsize: Tuple[float, float] = (12, 8),
) -> Figure:
    """
    Calculate and plot trial-averaged spike rate-LFP power correlation across frequencies.

    Computes spike rate-LFP power correlation for each trial separately, then averages
    results across trials with optional error bands.

    Parameters
    ----------
    spike_df : pd.DataFrame
        DataFrame containing spike data with columns 'timestamps', 'node_ids', and 'pop_name'.
    lfp_data : xr.DataArray
        LFP data with time dimension.
    time_windows : List[Tuple[float, float]]
        List of (start, end) time tuples in milliseconds for each trial.
    fs : float
        Sampling frequency in Hz.
    pop_names : List[str]
        List of population names to analyze.
    filter_method : str, optional
        Filtering method: 'wavelet' or 'butter' (default: 'wavelet').
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter (default: 2.0).
    lowcut : float, optional
        Lower frequency bound (Hz) for butterworth filter.
    highcut : float, optional
        Upper frequency bound (Hz) for butterworth filter.
    freq_range : Tuple[float, float], optional
        Min and max frequency to analyze in Hz (default: (10, 100)).
    freq_step : float, optional
        Step size for frequency analysis in Hz (default: 5).
    type_name : str, optional
        Which type of spike rate to use (default: 'raw').
    error_type : str, optional
        Type of error bars: "ci" for 95% CI, "sem" for SEM, or "std" for SD (default: "ci").
    figsize : Tuple[float, float], optional
        Figure size (width, height) in inches (default: (12, 8)).

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the trial-averaged correlation plot.

    Raises
    ------
    ValueError
        If error_type is invalid.

    Notes
    -----
    - Uses calculate_trial_statistics helper for consistent error computation.
    - NaN values are handled gracefully with warnings for problematic trials.

    Examples
    --------
    >>> time_windows = [(1000, 2000), (2500, 3500), (4000, 5000)]
    >>> fig = plot_trial_avg_spike_power_correlation(
    ...     spike_df=spike_df,
    ...     lfp_data=lfp,
    ...     time_windows=time_windows,
    ...     fs=400,
    ...     pop_names=['PV', 'SST'],
    ...     error_type='ci'
    ... )
    """
    if error_type not in ["ci", "sem", "std"]:
        raise ValueError(
            "error_type must be 'ci' for confidence interval, 'sem' for standard error, "
            "or 'std' for standard deviation"
        )

    # Validate that fs matches LFP data sampling rate if available
    if hasattr(lfp_data, 'fs') and lfp_data.fs != fs:
        raise ValueError(
            f"Provided fs ({fs} Hz) does not match LFP data sampling rate ({lfp_data.fs} Hz). "
        )

    # Setup frequencies for analysis
    frequencies = np.arange(freq_range[0], freq_range[1] + 1, freq_step)

    # Pre-calculate LFP power for all frequencies (same for all trials)
    power_by_freq = {}
    for freq in frequencies:
        power_by_freq[freq] = get_lfp_power(
            lfp_data, freq, fs, filter_method, lowcut=lowcut, highcut=highcut, bandwidth=bandwidth
        )

    # Storage: dict of pop_name -> list of trial_correlations per frequency
    all_correlations = {pop: {freq: [] for freq in frequencies} for pop in pop_names}

    # Process each trial
    for trial_idx, (start_time, end_time) in enumerate(time_windows):
        # Extract spikes for this trial
        trial_spikes = spike_df[
            (spike_df["timestamps"] >= start_time) & (spike_df["timestamps"] <= end_time)
        ].copy()

        if len(trial_spikes) == 0:
            print(f"Warning: No spikes found in trial {trial_idx} ({start_time}-{end_time} ms)")
            continue

        # Compute spike rate for this trial
        trial_spike_rate = bmspikes.get_population_spike_rate(
            trial_spikes, fs=fs, t_start=start_time, t_stop=end_time
        )

        # Calculate correlations for each population and frequency
        for pop in pop_names:
            if pop not in trial_spike_rate.population.values:
                print(f"Warning: Population {pop} not found in trial {trial_idx}")
                continue

            pop_spike_rate = trial_spike_rate.sel(population=pop, type=type_name)

            for freq in frequencies:
                try:
                    lfp_power = power_by_freq[freq]
                    trial_lfp_power = lfp_power.sel(time=slice(start_time, end_time))

                    if len(trial_lfp_power) < 2 or len(pop_spike_rate) < 2:
                        continue

                    # Align time coordinates
                    common_times = np.intersect1d(pop_spike_rate.time.values,
                                                 trial_lfp_power.time.values)
                    if len(common_times) < 2:
                        continue

                    trial_sr = pop_spike_rate.sel(time=common_times).values
                    trial_lfp = trial_lfp_power.sel(time=common_times).values

                    # Compute correlation
                    corr, _ = stats.spearmanr(trial_sr, trial_lfp)
                    if not np.isnan(corr):
                        all_correlations[pop][freq].append(corr)

                except Exception as e:
                    print(
                        f"Warning: Error computing correlation for {pop} at {freq} Hz "
                        f"in trial {trial_idx}: {e}"
                    )
                    continue

    # Calculate trial statistics for each population/frequency
    results = {pop: {} for pop in pop_names}
    for pop in pop_names:
        for freq in frequencies:
            if len(all_correlations[pop][freq]) > 0:
                freq_data = np.array(all_correlations[pop][freq])
                mean_corr, error_corr = calculate_trial_statistics(
                    freq_data.reshape(-1, 1), error_type=error_type
                )
                results[pop][freq] = {
                    "mean": mean_corr[0],
                    "error": error_corr[0],
                    "n_trials": len(freq_data),
                }

    # Create plot
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=figsize)

    colors = plt.get_cmap("tab10")
    for i, pop in enumerate(pop_names):
        plot_freqs = []
        plot_means = []
        plot_errors = []

        for freq in frequencies:
            if freq in results[pop] and "mean" in results[pop][freq]:
                plot_freqs.append(freq)
                plot_means.append(results[pop][freq]["mean"])
                plot_errors.append(results[pop][freq]["error"])

        if len(plot_freqs) == 0:
            continue

        plot_freqs = np.array(plot_freqs)
        plot_means = np.array(plot_means)
        plot_errors = np.array(plot_errors)
        color = colors(i)

        # Plot line
        plt.plot(
            plot_freqs, plot_means, marker="o", label=pop, linewidth=2, markersize=6, color=color
        )

        # Plot error band
        plt.fill_between(
            plot_freqs,
            plot_means - plot_errors,
            plot_means + plot_errors,
            alpha=0.2,
            color=color,
        )

    # Formatting
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Spike Rate-Power Correlation", fontsize=12)

    error_labels = {"ci": "95% CI", "sem": "±SEM", "std": "±1 SD"}
    error_label = error_labels[error_type]
    plt.title(
        f"Trial-Averaged Spike Rate-LFP Power Correlation ({error_label})",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    # Setup legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=colors(i), marker="o", linestyle="-", label=pop)
        for i, pop in enumerate(pop_names)
    ]
    legend_elements.append(Line2D([0], [0], color="gray", alpha=0.3, linewidth=10, label=error_label))
    plt.legend(handles=legend_elements, fontsize=10, loc="best")

    # Axis formatting
    if len(frequencies) > 10:
        plt.xticks(frequencies[::2])
    else:
        plt.xticks(frequencies)
    plt.xlim(frequencies[0], frequencies[-1])

    y_min, y_max = plt.ylim()
    plt.ylim(min(y_min, -0.1), max(y_max, 0.1))

    plt.tight_layout()
    return fig


def plot_cycle_with_spike_histograms(phase_data, pop_names: List[str], bins: int = 36):
    """
    Plot an idealized cycle with spike histograms for different neuron populations.

    Parameters
    -----------
    phase_data : dict
        Dictionary containing phase values for each spike and neuron population
    pop_names : List[str]
        List of population names to be plotted
    bins : int, optional
        Number of bins for the phase histogram (default 36 gives 10-degree bins)

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the cycle and histograms
    """
    sns.set_style("whitegrid")
    # Create a figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(len(pop_names) + 1, 1, height_ratios=[1.5] + [1] * len(pop_names))

    # Top subplot: Idealized gamma cycle
    ax_gamma = fig.add_subplot(gs[0])

    # Create an idealized gamma cycle
    x = np.linspace(-np.pi, np.pi, 1000)
    y = np.sin(x)

    ax_gamma.plot(x, y, "b-", linewidth=2)
    ax_gamma.set_title("Cycle with Neuron Population Spike Distributions", fontsize=14)
    ax_gamma.set_ylabel("Amplitude", fontsize=12)
    ax_gamma.set_xlim(-np.pi, np.pi)
    ax_gamma.set_xticks(np.linspace(-np.pi, np.pi, 9))
    ax_gamma.set_xticklabels(["-180°", "-135°", "-90°", "-45°", "0°", "45°", "90°", "135°", "180°"])
    ax_gamma.grid(True)
    ax_gamma.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax_gamma.axvline(x=0, color="k", linestyle="--", alpha=0.3)

    # Generate a color map for the different populations
    colors = plt.cm.tab10(np.linspace(0, 1, len(pop_names)))

    # Add histograms for each neuron population
    for i, pop_name in enumerate(pop_names):
        ax_hist = fig.add_subplot(gs[i + 1], sharex=ax_gamma)

        # Compute histogram
        hist, bin_edges = np.histogram(phase_data[pop_name], bins=bins, range=(-np.pi, np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize histogram
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist) * 100  # Convert to percentage

        # Plot histogram
        ax_hist.bar(bin_centers, hist, width=2 * np.pi / bins, alpha=0.7, color=colors[i])
        ax_hist.set_ylabel(f"{pop_name}\nSpikes (%)", fontsize=10)

        # Add grid to align with gamma cycle
        ax_hist.grid(True, alpha=0.3)
        ax_hist.set_ylim(0, max(hist) * 1.2)  # Add some headroom

    # Set x-label for the last subplot
    ax_hist.set_xlabel("Phase (degrees)", fontsize=12)

    plt.tight_layout()
    return fig


def plot_entrainment_by_population(ppc_dict: Dict[str, Dict[str, Dict[float, float]]], pop_names: List[str], freqs: List[float], figsize: Tuple[float, float] = (15, 8), title: Optional[str] = None):
    """
    Plot PPC for all node populations on one graph with mean and standard error.

    Parameters:
    -----------
    ppc_dict : Dict[str, Dict[str, Dict[float, float]]]
        Dictionary containing PPC data organized by population, node, and frequency
    pop_names : List[str]
        List of population names to plot data for
    freqs : List[float]
        List of frequencies to plot
    figsize : Tuple[float, float], optional
        Figure size for the plot
    title : str, optional
        Title for the plot

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the bar plot
    """
    # Set up the visualization style
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=figsize)

    # Calculate the width of each group of bars
    n_groups = len(freqs)
    n_populations = len(pop_names)
    group_width = 0.8
    bar_width = group_width / n_populations

    # Color palette for different populations
    pop_colors = sns.color_palette(n_colors=n_populations)

    # For tracking x-axis positions and labels
    x_centers = np.arange(n_groups)
    tick_labels = [str(freq) for freq in freqs]

    # Process and plot data for each population
    for i, pop in enumerate(pop_names):
        # Store mean and SE for each frequency in this population
        means = []
        errors = []
        valid_freqs_idx = []

        # Collect and process data for all frequencies in this population
        for freq_idx, freq in enumerate(freqs):
            freq_values = []

            # Collect values across all nodes for this frequency
            for node in ppc_dict[pop]:
                try:
                    ppc_value = ppc_dict[pop][node][freq]
                    freq_values.append(ppc_value)
                except KeyError:
                    continue

            # If we have data for this frequency
            if freq_values:
                mean_val = np.mean(freq_values)
                se_val = stats.sem(freq_values)
                means.append(mean_val)
                errors.append(se_val)
                valid_freqs_idx.append(freq_idx)

        # Calculate x positions for this population's bars
        # Each population's bars are offset within their frequency group
        x_positions = x_centers[valid_freqs_idx] + (i - n_populations / 2 + 0.5) * bar_width

        # Plot bars with error bars
        plt.bar(
            x_positions, means, width=bar_width * 0.9, color=pop_colors[i], alpha=0.7, label=pop
        )
        plt.errorbar(x_positions, means, yerr=errors, fmt="none", ecolor="black", capsize=4)

    # Set up the plot labels and legend
    plt.xlabel("Frequency")
    plt.ylabel("PPC Value")
    if title:
        plt.title(title)
    plt.xticks(x_centers, tick_labels)
    plt.legend(title="Population")

    # Adjust layout and save
    plt.tight_layout()
    return fig


def plot_entrainment_swarm_plot(ppc_dict: Dict[str, Dict[str, Dict[float, float]]], pop_names: List[str], freq: Union[float, int], save_path: Optional[str] = None, title: Optional[str] = None):
    """
    Plot a swarm plot of the entrainment for different populations at a single frequency.

    Parameters:
    -----------
    ppc_dict : Dict[str, Dict[str, Dict[float, float]]]
        Dictionary containing PPC values organized by population, node, and frequency
    pop_names : List[str]
        List of population names to include in the plot
    freq : Union[float, int]
        The specific frequency to plot
    save_path : str, optional
        Path to save the figure. If None, figure is just displayed.
    title : str, optional
        Title for the plot

    Returns:
    --------
    matplotlib.figure.Figure
        The figure object for further customization if needed
    """
    # Set the style
    sns.set_style("whitegrid")

    # Prepare data for the swarm plot
    data_list = []

    for pop in pop_names:
        values = []
        node_ids = []

        for node in ppc_dict[pop]:
            if freq in ppc_dict[pop][node] and ppc_dict[pop][node][freq] is not None:
                data_list.append(
                    {"Population": pop, "Node": node, "PPC Difference": ppc_dict[pop][node][freq]}
                )

    # Create DataFrame in long format
    df = pd.DataFrame(data_list)

    if df.empty:
        print(f"No data available for frequency {freq}.")
        return None

    # Print mean PPC change for each population)
    for pop in pop_names:
        subset = df[df["Population"] == pop]
        if not subset.empty:
            mean_val = subset["PPC Difference"].mean()
            std_val = subset["PPC Difference"].std()
            n = len(subset)
            sem_val = std_val / np.sqrt(n)  # Standard error of the mean
            print(f"{pop}: {mean_val:.4f} ± {sem_val:.4f} (n={n})")

    # Create figure
    fig = plt.figure(figsize=(max(8, len(pop_names) * 1.5), 8))

    # Create swarm plot
    ax = sns.swarmplot(
        x="Population",
        y="PPC Difference",
        data=df,
        size=3,
        # palette='Set2'
    )

    # Add sample size annotations
    for i, pop in enumerate(pop_names):
        subset = df[df["Population"] == pop]
        if not subset.empty:
            n = len(subset)
            y_min = subset["PPC Difference"].min()
            y_max = subset["PPC Difference"].max()

            # Position annotation below the lowest point
            plt.annotate(
                f"n={n}", (i, y_min - 0.05 * (y_max - y_min) - 0.05), ha="center", fontsize=10
            )

    # Add reference line at y=0
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.7)

    # Add horizontal lines for mean values
    for i, pop in enumerate(pop_names):
        subset = df[df["Population"] == pop]
        if not subset.empty:
            mean_val = subset["PPC Difference"].mean()
            plt.plot([i - 0.25, i + 0.25], [mean_val, mean_val], "r-", linewidth=2)

    # Calculate and display statistics
    if len(pop_names) > 1:
        # Print statistical test results
        print(f"\nMann-Whitney U Test Results at {freq} Hz:")
        print("-" * 60)

        # Add p-values for pairwise comparisons
        y_max = df["PPC Difference"].max()
        y_min = df["PPC Difference"].min()
        y_range = y_max - y_min

        # Perform t-tests between populations if there are at least 2
        for i in range(len(pop_names)):
            for j in range(i + 1, len(pop_names)):
                pop1 = pop_names[i]
                pop2 = pop_names[j]

                vals1 = df[df["Population"] == pop1]["PPC Difference"].values
                vals2 = df[df["Population"] == pop2]["PPC Difference"].values

                if len(vals1) > 1 and len(vals2) > 1:
                    # Perform Mann-Whitney U test (non-parametric)
                    u_stat, p_val = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")

                    # Add significance markers
                    sig_str = "ns"
                    if p_val < 0.05:
                        sig_str = "*"
                    if p_val < 0.01:
                        sig_str = "**"
                    if p_val < 0.001:
                        sig_str = "***"

                    # Position the significance bar
                    bar_height = y_max + 0.1 * y_range * (1 + (j - i - 1) * 0.5)

                    # Draw the bar
                    plt.plot([i, j], [bar_height, bar_height], "k-")
                    plt.plot([i, i], [bar_height - 0.02 * y_range, bar_height], "k-")
                    plt.plot([j, j], [bar_height - 0.02 * y_range, bar_height], "k-")

                    # Add significance marker
                    plt.text(
                        (i + j) / 2,
                        bar_height + 0.01 * y_range,
                        sig_str,
                        ha="center",
                        va="bottom",
                        fontsize=12,
                    )

                    # Print the statistical comparison
                    print(f"{pop1} vs {pop2}: U={u_stat:.1f}, p={p_val:.4f} {sig_str}")

    # Add labels and title
    plt.xlabel("Population", fontsize=14)
    plt.ylabel("PPC", fontsize=14)
    if title:
        plt.title(title, fontsize=16)

    # Adjust y-axis limits to make room for annotations
    y_min, y_max = plt.ylim()
    plt.ylim(y_min - 0.15 * (y_max - y_min), y_max + 0.25 * (y_max - y_min))

    # Add gridlines
    plt.grid(True, linestyle="--", alpha=0.7, axis="y")

    # Adjust layout
    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(f"{save_path}/ppc_change_swarm_plot_{freq}Hz.png", dpi=300, bbox_inches="tight")

    return fig


def plot_trial_avg_entrainment(
    spike_df: pd.DataFrame,
    lfp: xr.DataArray,
    time_windows: List[Tuple[float, float]],
    entrainment_method: str,
    pop_names: List[str],
    freqs: Union[List[float], np.ndarray],
    firing_quantile: float,
    spike_fs: float = 1000,
    error_type: str = "ci",
) -> Figure:
    """
    Plot trial-averaged entrainment for specified population names. Only supports wavelet filter current, could easily add other support

    Parameters:
    -----------
    spike_df : pd.DataFrame
        Spike data containing timestamps, node_ids, and pop_name columns
    lfp : xr.DataArray
        Xarray for a channel of the lfp data
    time_windows : List[Tuple[float, float]]
        List of windows to analysis with start and stp time [(start_time, end_time), ...] for each trial
    entrainment_method : str
        Method for entrainment calculation ('ppc', 'ppc2' or 'plv')
    pop_names : List[str]
        List of population names to process (e.g., ['FSI', 'LTS'])
    freqs : Union[List[float], np.ndarray]
        Array of frequencies to analyze (Hz)
    firing_quantile : float
        Upper quantile threshold for selecting high-firing cells (e.g., 0.8 for top 20%)
    spike_fs : float, optional
        fs for spike data. Default is 1000
    error_type : str, optional
        Type of error bars to plot: "ci" for 95% confidence interval, "sem" for standard error, "std" for standard deviation

    Raises:
    -------
    ValueError
        If entrainment_method is not 'ppc', 'ppc2' or 'plv'
        If error_type is not 'ci', 'sem', or 'std'
        If no spikes found for a population in a trial

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    sns.set_style("whitegrid")
    # Validate inputs
    if entrainment_method not in ["ppc", "plv", "ppc2"]:
        raise ValueError("entrainment_method must be 'ppc', ppc2 or 'plv'")

    if error_type not in ["ci", "sem", "std"]:
        raise ValueError(
            "error_type must be 'ci' for confidence interval, 'sem' for standard error, or 'std' for standard deviation"
        )

    if not (0 <= firing_quantile < 1):
        raise ValueError("firing_quantile must be between 0 and 1")

    # Convert freqs to numpy array for easier indexing
    freqs = np.array(freqs)

    # Collect all PPC/PLV values across trials for each population
    all_plv_data = {}  # Dictionary to store results for each population

    # Initialize storage for each population
    for pop_name in pop_names:
        all_plv_data[pop_name] = []  # Will be shape (n_trials, n_freqs)

    # Loop through all pulse groups to collect data
    for trial_idx in range(len(time_windows)):
        plv_lists = {}  # Store PLV lists for this trial

        # Initialize PLV lists for each population
        for pop_name in pop_names:
            plv_lists[pop_name] = []

        # Filter spikes for this trial
        network_spikes = spike_df[
            (spike_df["timestamps"] >= time_windows[trial_idx][0])
            & (spike_df["timestamps"] <= time_windows[trial_idx][1])
        ].copy()

        # Process each population
        pop_spike_data = {}
        for pop_name in pop_names:
            # Get spikes for this population
            pop_spikes = network_spikes[network_spikes["pop_name"] == pop_name]

            if len(pop_spikes) == 0:
                print(f"Warning: No spikes found for population {pop_name} in trial {trial_idx}")
                # Add NaN values for this trial/population
                plv_lists[pop_name] = [np.nan] * len(freqs)
                continue

            # Filter to get the top firing cells
            # firing_quantile of 0.8 gets the top 20% of firing cells to use
            pop_spikes = bmspikes.find_highest_firing_cells(
                pop_spikes, upper_quantile=firing_quantile
            )

            if len(pop_spikes) == 0:
                print(
                    f"Warning: No high-firing spikes found for population {pop_name} in trial {trial_idx}"
                )
                plv_lists[pop_name] = [np.nan] * len(freqs)
                continue

            pop_spike_data[pop_name] = pop_spikes

        # Calculate PPC/PLV for each frequency and each population
        for freq_idx, freq in enumerate(freqs):
            for pop_name in pop_names:
                if pop_name not in pop_spike_data:
                    continue  # Skip if no data for this population

                pop_spikes = pop_spike_data[pop_name]

                try:
                    if entrainment_method == "ppc":
                        result = bmentr.calculate_ppc(
                            pop_spikes["timestamps"].values,
                            lfp,
                            spike_fs=spike_fs,
                            lfp_fs=lfp.fs,
                            freq_of_interest=freq,
                            filter_method="wavelet",
                            ppc_method="gpu",
                        )
                    elif entrainment_method == "plv":
                        result = bmentr.calculate_spike_lfp_plv(
                            pop_spikes["timestamps"].values,
                            lfp,
                            spike_fs=spike_fs,
                            lfp_fs=lfp.fs,
                            freq_of_interest=freq,
                            filter_method="wavelet",
                        )
                    elif entrainment_method == "ppc2":
                        result = bmentr.calculate_ppc2(
                            pop_spikes["timestamps"].values,
                            lfp,
                            spike_fs=spike_fs,
                            lfp_fs=lfp.fs,
                            freq_of_interest=freq,
                            filter_method="wavelet",
                        )

                    plv_lists[pop_name].append(result)

                except Exception as e:
                    print(
                        f"Warning: Error calculating {entrainment_method} for {pop_name} at {freq}Hz in trial {trial_idx}: {e}"
                    )
                    plv_lists[pop_name].append(np.nan)

        # Store this trial's results for each population
        for pop_name in pop_names:
            if pop_name in plv_lists and len(plv_lists[pop_name]) == len(freqs):
                all_plv_data[pop_name].append(plv_lists[pop_name])
            else:
                # Fill with NaNs if data is missing
                all_plv_data[pop_name].append([np.nan] * len(freqs))

    # Convert to numpy arrays and calculate statistics
    mean_plv = {}
    error_plv = {}

    for pop_name in pop_names:
        all_plv_data[pop_name] = np.array(all_plv_data[pop_name])  # Shape: (n_trials, n_freqs)

        # Use helper function to calculate trial statistics
        mean_plv[pop_name], error_plv[pop_name] = calculate_trial_statistics(
            all_plv_data[pop_name], error_type=error_type
        )

    # Create the combined plot
    fig = plt.figure(figsize=(12, 8))

    # Define markers and colors for different populations
    markers = ["o-", "s-", "^-", "D-", "v-", "<-", ">-", "p-"]
    colors = sns.color_palette(n_colors=len(pop_names))

    # Plot each population
    for i, pop_name in enumerate(pop_names):
        marker = markers[i % len(markers)]  # Cycle through markers if more populations than markers
        color = colors[i]

        # Only plot if we have valid data
        valid_mask = ~np.isnan(mean_plv[pop_name])
        if np.any(valid_mask):
            plt.plot(
                freqs[valid_mask],
                mean_plv[pop_name][valid_mask],
                marker,
                linewidth=2,
                label=pop_name,
                color=color,
                markersize=6,
            )

            # Add error bars/shading if available
            if not np.all(np.isnan(error_plv[pop_name])):
                plt.fill_between(
                    freqs[valid_mask],
                    (mean_plv[pop_name] - error_plv[pop_name])[valid_mask],
                    (mean_plv[pop_name] + error_plv[pop_name])[valid_mask],
                    alpha=0.3,
                    color=color,
                )

    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel(f"{entrainment_method.upper()}", fontsize=12)

    # Calculate percentage for title and update title based on error type
    firing_percentage = round(float((1 - firing_quantile) * 100), 1)
    error_labels = {"ci": "95% CI", "sem": "±SEM", "std": "±1 SD"}
    error_label = error_labels[error_type]
    plt.title(
        f"{entrainment_method.upper()} Across Trials for Top {firing_percentage}% Firing Cells ({error_label})",
        fontsize=14,
    )

    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_fr_hist_phase_amplitude(
    fr_hist: np.ndarray, 
    pop_names: List[str], 
    freq_labels: List[str], 
    nbins_pha: int = 16, 
    nbins_amp: int = 16,
    common_clim: bool = True, 
    figsize: Tuple[float, float] = (3, 2),
    cmap: str = 'viridis',
    title: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot firing rate histograms binned by LFP phase and amplitude. 
    Check out the bmtool/bmtool/analysis/entrainment.py function
    compute_fr_hist_phase_amplitude
    
    Parameters
    ----------
    fr_hist : np.ndarray
        Firing rate histogram of shape (n_pop, n_freq, nbins_pha, nbins_amp)
    pop_names : List[str]
        List of population names
    freq_labels : List[str]
        List of frequency labels for subplot titles (e.g., ['Beta', 'Gamma'])
    nbins_pha : int, default=16
        Number of phase bins
    nbins_amp : int, default=16
        Number of amplitude bins
    common_clim : bool, default=True
        Whether to use common color limits across all subplots
    figsize : Tuple[float, float], default=(3, 2)
        Size of each subplot
    cmap : str, default='RdBu_r'
        Colormap to use
    title : Optional[str], default=None
        Overall title for the figure
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes objects
        
    Examples
    --------
    >>> fig, axs = plot_fr_hist_phase_amplitude(
    ...     fr_hist, ['PV', 'SST'], ['Beta', 'Gamma'], 
    ...     common_clim=True, cmap='RdBu_r', title='LFP Phase-Amplitude Coupling'
    ... )
    """
    pha_bins = np.linspace(-np.pi, np.pi, nbins_pha + 1)
    quantiles = np.linspace(0, 1, nbins_amp + 1)
    
    n_pop = len(pop_names)
    n_freq = len(freq_labels)
    
    fig, axs = plt.subplots(n_pop, n_freq, 
                           figsize=(figsize[0] * n_freq, figsize[1] * n_pop),
                           squeeze=False)

    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    for i, p in enumerate(pop_names):
        if common_clim:
            vmin, vmax = fr_hist.min(), fr_hist.max()
        else:
            vmin, vmax = None, None
            
        for j, freq_label in enumerate(freq_labels):
            ax = axs[i, j]
            pcm = ax.pcolormesh(pha_bins, quantiles, fr_hist[i, j].T, 
                               vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(p)
            
            if i < n_pop - 1:
                ax.get_xaxis().set_visible(False)
            else:
                ax.set_xlabel(freq_label.title() + ' Phase')
                ax.set_xticks((-np.pi, 0, np.pi))
                ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
                
            if j > 0:
                ax.get_yaxis().set_visible(False)
            else:
                ax.set_ylabel('Amplitude (quantile)')
                
            if not common_clim:
                plt.colorbar(mappable=pcm, ax=ax, 
                           label='Firing rate (% Change)' if j == n_freq - 1 else None, 
                           pad=0.02)
                           
        if common_clim:
            plt.colorbar(mappable=pcm, ax=axs[i], 
                        label='Firing rate (% Change)', pad=0.02)
    
    return fig, axs


def plot_trial_avg_spike_rate_plv(
    spike_rate: xr.DataArray,
    time_windows: List[Tuple[float, float]],
    pop_names: List[str],
    freqs: Union[List[float], np.ndarray],
    pop_pairs: Optional[List[Tuple[str, str]]] = None,
    error_type: str = "ci",
    figsize: Tuple[float, float] = (12, 8),
    filter_method: str = "wavelet",
    bandwidth: float = 1.0,
) -> Figure:
    """
    Plot trial-averaged Phase Locking Value (PLV) between spike rate pairs across frequencies.

    This function computes PLV between spike rates of different population pairs for each trial,
    then averages the results across trials with optional error bars/bands.

    Parameters
    ----------
    spike_rate : xr.DataArray
        Pre-computed spike rate data with dimensions (time, population, type).
        Must have an 'fs' attribute (sampling frequency in Hz).
        Time dimension should be in milliseconds.
    time_windows : List[Tuple[float, float]]
        List of (start, end) time tuples in milliseconds for each trial window.
    pop_names : List[str]
        List of population names to consider (must match 'population' coords in spike_rate).
    freqs : Union[List[float], np.ndarray]
        Array of frequencies (Hz) to analyze.
    pop_pairs : Optional[List[Tuple[str, str]]], optional
        List of (pop1, pop2) tuples specifying population pairs to analyze.
        If None, all unique pairs from pop_names are generated (default: None).
    error_type : str, optional
        Type of error bars to plot: "ci" for 95% confidence interval, "sem" for standard error,
        or "std" for standard deviation (default: "ci").
    figsize : Tuple[float, float], optional
        Figure size (width, height) in inches (default: (12, 8)).
    filter_method : str, optional
        Filtering method for PLV calculation: 'wavelet' or 'butter' (default: 'wavelet').
    bandwidth : float, optional
        Bandwidth parameter for wavelet filter (default: 1.0).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the trial-averaged PLV plot.

    Raises
    ------
    ValueError
        If error_type is not 'ci', 'sem', or 'std'.
        If pop_pairs contains invalid population names.
        If spike_rate does not have required dimensions/attributes.

    Notes
    -----
    - Uses 'raw' type from spike_rate dimension (hardcoded for spike rate analysis).
    - Time units in time_windows must match spike_rate.time units (milliseconds).
    - Generates all unique population pairs if pop_pairs is None.

    Examples
    --------
    >>> fig = plot_trial_avg_spike_rate_plv(
    ...     spike_rate=spike_rate,
    ...     time_windows=[(1000, 2000), (2500, 3500), (4000, 5000)],
    ...     pop_names=['ET', 'IT', 'PV', 'SST'],
    ...     freqs=[10, 20, 30, 40, 50],
    ...     error_type='ci'
    ... )
    """
    sns.set_style("whitegrid")

    # Validate inputs
    if error_type not in ["ci", "sem", "std"]:
        raise ValueError(
            "error_type must be 'ci' for confidence interval, 'sem' for standard error, or 'std' for standard deviation"
        )

    if "fs" not in spike_rate.attrs:
        raise ValueError("spike_rate must have 'fs' attribute (sampling frequency).")

    if "time" not in spike_rate.dims or "population" not in spike_rate.dims:
        raise ValueError("spike_rate must have 'time' and 'population' dimensions.")

    # Convert freqs to numpy array
    freqs = np.array(freqs)

    # Generate population pairs if not provided
    if pop_pairs is None:
        n_pops = len(pop_names)
        pop_pairs = [
            (pop_names[i], pop_names[j]) for i in range(n_pops) for j in range(i + 1, n_pops)
        ]

    # Validate population pairs
    available_pops = set(spike_rate.population.values)
    for pop1, pop2 in pop_pairs:
        if pop1 not in available_pops or pop2 not in available_pops:
            raise ValueError(
                f"Population pair ({pop1}, {pop2}) contains names not in spike_rate. "
                f"Available: {available_pops}"
            )

    # Get sampling frequency
    fs = spike_rate.fs

    # Storage for PLV data: dict of pair -> list of trials -> list of freqs
    all_plv_data = {f"{p1}-{p2}": [] for p1, p2 in pop_pairs}

    # Loop through each trial window
    for trial_idx, (start_time, end_time) in enumerate(time_windows):
        try:
            # Slice spike rate data for this trial window
            trial_data = spike_rate.sel(time=slice(start_time, end_time), type="raw")

            if trial_data.time.size == 0:
                print(f"Warning: No data in trial {trial_idx} for window ({start_time}, {end_time})")
                for pair_key in all_plv_data.keys():
                    all_plv_data[pair_key].append([np.nan] * len(freqs))
                continue

            # Calculate PLV for each population pair at each frequency
            trial_plv_values = {}
            for pop1, pop2 in pop_pairs:
                pair_key = f"{pop1}-{pop2}"
                trial_plv_values[pair_key] = []

                try:
                    # Extract spike rate signals for this pair
                    sr1 = trial_data.sel(population=pop1).values
                    sr2 = trial_data.sel(population=pop2).values

                    if len(sr1) < 2 or len(sr2) < 2:
                        print(
                            f"Warning: Insufficient data for {pair_key} in trial {trial_idx}"
                        )
                        trial_plv_values[pair_key] = [np.nan] * len(freqs)
                        continue

                    # Calculate PLV for each frequency
                    for freq in freqs:
                        try:
                            plv = bmentr.calculate_signal_signal_plv(
                                sr1,
                                sr2,
                                fs=fs,
                                freq_of_interest=freq,
                                filter_method=filter_method,
                                bandwidth=bandwidth,
                            )
                            trial_plv_values[pair_key].append(plv)
                        except Exception as e:
                            print(
                                f"Warning: Error calculating PLV for {pair_key} at {freq} Hz in trial {trial_idx}: {e}"
                            )
                            trial_plv_values[pair_key].append(np.nan)

                except Exception as e:
                    print(f"Warning: Error processing {pair_key} in trial {trial_idx}: {e}")
                    trial_plv_values[pair_key] = [np.nan] * len(freqs)

            # Store trial results
            for pair_key in all_plv_data.keys():
                if pair_key in trial_plv_values and len(trial_plv_values[pair_key]) == len(freqs):
                    all_plv_data[pair_key].append(trial_plv_values[pair_key])
                else:
                    all_plv_data[pair_key].append([np.nan] * len(freqs))

        except Exception as e:
            print(f"Warning: Error processing trial {trial_idx}: {e}")
            for pair_key in all_plv_data.keys():
                all_plv_data[pair_key].append([np.nan] * len(freqs))

    # Convert to numpy arrays and calculate statistics
    mean_plv = {}
    error_plv = {}

    for pair_key in all_plv_data.keys():
        all_plv_data[pair_key] = np.array(all_plv_data[pair_key])  # Shape: (n_trials, n_freqs)

        # Use helper function to calculate trial statistics
        mean_plv[pair_key], error_plv[pair_key] = calculate_trial_statistics(
            all_plv_data[pair_key], error_type=error_type
        )

    # Create plot
    fig = plt.figure(figsize=figsize)

    # Determine subplot layout if many pairs
    n_pairs = len(pop_pairs)
    if n_pairs <= 3:
        # Single row for up to 3 pairs
        n_rows, n_cols = 1, n_pairs
    else:
        # Multiple rows for more pairs
        n_cols = min(n_pairs, 3)
        n_rows = (n_pairs + n_cols - 1) // n_cols

    # Define markers and colors for different pairs
    markers = ["o-", "s-", "^-", "D-", "v-", "<-", ">-", "p-"]
    colors = sns.color_palette(n_colors=n_pairs)

    # Plot each population pair
    if n_rows == 1 and n_cols == 1:
        axes = [plt.subplot(n_rows, n_cols, 1)]
    elif n_rows == 1:
        axes = [plt.subplot(n_rows, n_cols, i + 1) for i in range(n_cols)]
    else:
        axes = [plt.subplot(n_rows, n_cols, i + 1) for i in range(n_pairs)]

    for pair_idx, (pop1, pop2) in enumerate(pop_pairs):
        pair_key = f"{pop1}-{pop2}"
        marker = markers[pair_idx % len(markers)]
        color = colors[pair_idx]
        ax = axes[pair_idx] if isinstance(axes, list) else axes

        # Only plot if we have valid data
        valid_mask = ~np.isnan(mean_plv[pair_key])
        if np.any(valid_mask):
            ax.plot(
                freqs[valid_mask],
                mean_plv[pair_key][valid_mask],
                marker,
                linewidth=2,
                label=pair_key,
                color=color,
                markersize=6,
            )

            # Add error bars/shading if available
            if not np.all(np.isnan(error_plv[pair_key])):
                ax.fill_between(
                    freqs[valid_mask],
                    (mean_plv[pair_key] - error_plv[pair_key])[valid_mask],
                    (mean_plv[pair_key] + error_plv[pair_key])[valid_mask],
                    alpha=0.3,
                    color=color,
                )

        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("PLV", fontsize=11)
        ax.set_title(f"{pair_key}", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Add overall title
    error_labels = {"ci": "95% CI", "sem": "±SEM", "std": "±1 SD"}
    error_label = error_labels[error_type]
    fig.suptitle(f"Trial-Averaged Spike Rate PLV ({error_label})", fontsize=14, y=0.995)

    plt.tight_layout()
    return fig
