from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.gridspec import GridSpec
from scipy import stats

from bmtool.analysis import entrainment as bmentr
from bmtool.analysis import spikes as bmspikes
from bmtool.analysis.lfp import get_lfp_power


def plot_spike_power_correlation(
    spike_df: pd.DataFrame,
    lfp_data: xr.DataArray,
    firing_quantile: float,
    fs: float,
    pop_names: list,
    filter_method: str = "wavelet",
    bandwidth: float = 2.0,
    lowcut: float = None,
    highcut: float = None,
    freq_range: tuple = (10, 100),
    freq_step: float = 5,
    type_name: str = "raw",
    time_windows: list = None,
    error_type: str = "ci",  # New parameter: "ci" for confidence interval, "sem" for standard error, "std" for standard deviation
):
    """
    Calculate and plot correlation between population spike rates and LFP power across frequencies.
    Supports both single-signal and trial-based analysis with error bars.

    Parameters
    ----------
    spike_rate : xr.DataArray
        Population spike rates with dimensions (time, population[, type])
    lfp_data : xr.DataArray
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
    type_name : str, optional
        Which type of spike rate to use if 'type' dimension exists (default: 'raw')
    time_windows : list, optional
        List of (start, end) time tuples for trial-based analysis. If None, analyze entire signal
    error_type : str, optional
        Type of error bars to plot: "ci" for 95% confidence interval, "sem" for standard error, "std" for standard deviation
    """

    if not (0 <= firing_quantile < 1):
        raise ValueError("firing_quantile must be between 0 and 1")

    if error_type not in ["ci", "sem", "std"]:
        raise ValueError(
            "error_type must be 'ci' for confidence interval, 'sem' for standard error, or 'std' for standard deviation"
        )

    # Setup
    is_trial_based = time_windows is not None

    # Convert spike_df to spike rate with trial-based filtering of high firing cells
    if is_trial_based:
        # Initialize storage for trial-based spike rates
        trial_rates = []

        for start_time, end_time in time_windows:
            # Get spikes for this trial
            trial_spikes = spike_df[
                (spike_df["timestamps"] >= start_time) & (spike_df["timestamps"] <= end_time)
            ].copy()

            # Filter for high firing cells within this trial
            trial_spikes = bmspikes.find_highest_firing_cells(
                trial_spikes, upper_quantile=firing_quantile
            )
            # Calculate rate for this trial's filtered spikes
            trial_rate = bmspikes.get_population_spike_rate(
                trial_spikes, fs=fs, t_start=start_time, t_stop=end_time
            )
            trial_rates.append(trial_rate)

        # Combine all trial rates
        spike_rate = xr.concat(trial_rates, dim="trial")
    else:
        # For non-trial analysis, proceed as before
        spike_df = bmspikes.find_highest_firing_cells(spike_df, upper_quantile=firing_quantile)
        spike_rate = bmspikes.get_population_spike_rate(spike_df)

    # Setup frequencies for analysis
    frequencies = np.arange(freq_range[0], freq_range[1] + 1, freq_step)

    # Pre-calculate LFP power for all frequencies
    power_by_freq = {}
    for freq in frequencies:
        power_by_freq[freq] = get_lfp_power(
            lfp_data, freq, fs, filter_method, lowcut=lowcut, highcut=highcut, bandwidth=bandwidth
        )

    # Calculate correlations
    results = {}
    for pop in pop_names:
        pop_spike_rate = spike_rate.sel(population=pop, type=type_name)
        results[pop] = {}

        for freq in frequencies:
            lfp_power = power_by_freq[freq]

            if not is_trial_based:
                # Single signal analysis
                if len(pop_spike_rate) != len(lfp_power):
                    print(f"Warning: Length mismatch for {pop} at {freq} Hz")
                    continue

                corr, p_val = stats.spearmanr(pop_spike_rate, lfp_power)
                results[pop][freq] = {
                    "correlation": corr,
                    "p_value": p_val,
                }
            else:
                # Trial-based analysis using pre-filtered trial rates
                trial_correlations = []

                for trial_idx in range(len(time_windows)):
                    # Get time window first
                    start_time, end_time = time_windows[trial_idx]

                    # Get the pre-filtered spike rate for this trial
                    trial_spike_rate = pop_spike_rate.sel(trial=trial_idx)

                    # Get corresponding LFP power for this trial window
                    trial_lfp_power = lfp_power.sel(time=slice(start_time, end_time))

                    # Ensure both signals have same time points
                    common_times = np.intersect1d(trial_spike_rate.time, trial_lfp_power.time)

                    if len(common_times) > 0:
                        trial_sr = trial_spike_rate.sel(time=common_times).values
                        trial_lfp = trial_lfp_power.sel(time=common_times).values

                        if (
                            len(trial_sr) > 1 and len(trial_lfp) > 1
                        ):  # Need at least 2 points for correlation
                            corr, _ = stats.spearmanr(trial_sr, trial_lfp)
                            if not np.isnan(corr):
                                trial_correlations.append(corr)

                # Calculate trial statistics
                if len(trial_correlations) > 0:
                    trial_correlations = np.array(trial_correlations)
                    mean_corr = np.mean(trial_correlations)

                    if len(trial_correlations) > 1:
                        if error_type == "ci":
                            # Calculate 95% confidence interval using t-distribution
                            df = len(trial_correlations) - 1
                            sem = stats.sem(trial_correlations)
                            t_critical = stats.t.ppf(0.975, df)  # 95% CI, two-tailed
                            error_val = t_critical * sem
                            error_lower = mean_corr - error_val
                            error_upper = mean_corr + error_val
                        elif error_type == "sem":
                            # Calculate standard error of the mean
                            sem = stats.sem(trial_correlations)
                            error_lower = mean_corr - sem
                            error_upper = mean_corr + sem
                        elif error_type == "std":
                            # Calculate standard deviation
                            std = np.std(trial_correlations, ddof=1)
                            error_lower = mean_corr - std
                            error_upper = mean_corr + std
                    else:
                        error_lower = error_upper = mean_corr

                    results[pop][freq] = {
                        "correlation": mean_corr,
                        "error_lower": error_lower,
                        "error_upper": error_upper,
                        "n_trials": len(trial_correlations),
                        "trial_correlations": trial_correlations,
                    }
                else:
                    # No valid trials
                    results[pop][freq] = {
                        "correlation": np.nan,
                        "error_lower": np.nan,
                        "error_upper": np.nan,
                        "n_trials": 0,
                        "trial_correlations": np.array([]),
                    }

    # Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    for i, pop in enumerate(pop_names):
        # Extract data for plotting
        plot_freqs = []
        plot_corrs = []
        plot_ci_lower = []
        plot_ci_upper = []

        for freq in frequencies:
            if freq in results[pop] and not np.isnan(results[pop][freq]["correlation"]):
                plot_freqs.append(freq)
                plot_corrs.append(results[pop][freq]["correlation"])

                if is_trial_based:
                    plot_ci_lower.append(results[pop][freq]["error_lower"])
                    plot_ci_upper.append(results[pop][freq]["error_upper"])

        if len(plot_freqs) == 0:
            continue

        # Convert to arrays
        plot_freqs = np.array(plot_freqs)
        plot_corrs = np.array(plot_corrs)

        # Get color for this population
        colors = plt.get_cmap("tab10")
        color = colors(i)

        # Plot main line
        plt.plot(
            plot_freqs, plot_corrs, marker="o", label=pop, linewidth=2, markersize=6, color=color
        )

        # Plot error bands for trial-based analysis
        if is_trial_based and len(plot_ci_lower) > 0:
            plot_ci_lower = np.array(plot_ci_lower)
            plot_ci_upper = np.array(plot_ci_upper)
            plt.fill_between(plot_freqs, plot_ci_lower, plot_ci_upper, alpha=0.2, color=color)

    # Formatting
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Spike Rate-Power Correlation", fontsize=12)

    # Calculate percentage for title
    firing_percentage = round(float((1 - firing_quantile) * 100), 1)
    if is_trial_based:
        title = f"Trial-averaged Spike Rate-LFP Power Correlation\nTop {firing_percentage}% Firing Cells (95% CI)"
    else:
        title = f"Spike Rate-LFP Power Correlation\nTop {firing_percentage}% Firing Cells"

    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    # Legend
    # Create legend elements for each population
    from matplotlib.lines import Line2D

    colors = plt.get_cmap("tab10")
    legend_elements = [
        Line2D([0], [0], color=colors(i), marker="o", linestyle="-", label=pop)
        for i, pop in enumerate(pop_names)
    ]

    # Add error band legend element for trial-based analysis
    if is_trial_based:
        # Map error type to legend label
        error_labels = {"ci": "95% CI", "sem": "±SEM", "std": "±1 SD"}
        error_label = error_labels[error_type]

        legend_elements.append(
            Line2D([0], [0], color="gray", alpha=0.3, linewidth=10, label=error_label)
        )

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
    plt.show()


def plot_cycle_with_spike_histograms(phase_data, bins=36, pop_name=None):
    """
    Plot an idealized cycle with spike histograms for different neuron populations.

    Parameters:
    -----------
    phase_data : dict
        Dictionary containing phase values for each spike and neuron population
    fs : float
        Sampling frequency of LFP in Hz
    bins : int
        Number of bins for the phase histogram (default 36 gives 10-degree bins)
    pop_name : list
        List of population names to be plotted
    """
    sns.set_style("whitegrid")
    # Create a figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(len(pop_name) + 1, 1, height_ratios=[1.5] + [1] * len(pop_name))

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
    colors = plt.cm.tab10(np.linspace(0, 1, len(pop_name)))

    # Add histograms for each neuron population
    for i, pop_name in enumerate(pop_name):
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
    plt.show()


def plot_entrainment_by_population(ppc_dict, pop_names, freqs, figsize=(15, 8), title=None):
    """
    Plot PPC for all node populations on one graph with mean and standard error.

    Parameters:
    -----------
    ppc_dict : dict
        Dictionary containing PPC data organized by population, node, and frequency
    pop_names : list
        List of population names to plot data for
    freqs : list
        List of frequencies to plot
    figsize : tuple
        Figure size for the plot
    """
    # Set up the visualization style
    sns.set_style("whitegrid")
    plt.figure(figsize=figsize)

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
    plt.show()


def plot_entrainment_swarm_plot(ppc_dict, pop_names, freq, save_path=None, title=None):
    """
    Plot a swarm plot of the entrainment for different populations at a single frequency.

    Parameters:
    -----------
    ppc_dict : dict
        Dictionary containing PPC values organized by population, node, and frequency
    pop_names : list
        List of population names to include in the plot
    freq : float or int
        The specific frequency to plot
    save_path : str, optional
        Path to save the figure. If None, figure is just displayed.

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
    plt.figure(figsize=(max(8, len(pop_names) * 1.5), 8))

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

    plt.show()


def plot_trial_avg_entrainment(
    spike_df: pd.DataFrame,
    lfp: np.ndarray,
    time_windows: List[Tuple[float, float]],
    entrainment_method: str,
    pop_names: List[str],
    freqs: Union[List[float], np.ndarray],
    firing_quantile: float,
    spike_fs: float = 1000,
    error_type: str = "ci",  # New parameter: "ci" for confidence interval, "sem" for standard error, "std" for standard deviation
) -> None:
    """
    Plot trial-averaged entrainment for specified population names. Only supports wavelet filter current, could easily add other support

    Parameters:
    -----------
    spike_df : pd.DataFrame
        Spike data containing timestamps, node_ids, and pop_name columns
    spike_fs : float
        fs for spike data. Default is 1000
    lfp : xarray
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
    error_type : str
        Type of error bars to plot: "ci" for 95% confidence interval, "sem" for standard error, "std" for standard deviation

    Raises:
    -------
    ValueError
        If entrainment_method is not 'ppc', 'ppc2' or 'plv'
        If error_type is not 'ci', 'sem', or 'std'
        If no spikes found for a population in a trial

    Returns:
    --------
    None
        Displays plot and prints summary statistics
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

        # Calculate statistics across trials, ignoring NaN values
        with np.errstate(invalid="ignore"):  # Suppress warnings for all-NaN slices
            mean_plv[pop_name] = np.nanmean(all_plv_data[pop_name], axis=0)

            if error_type == "ci":
                # Calculate 95% confidence intervals using SEM
                valid_counts = np.sum(~np.isnan(all_plv_data[pop_name]), axis=0)
                sem_plv = np.nanstd(all_plv_data[pop_name], axis=0, ddof=1) / np.sqrt(valid_counts)

                # For 95% CI, multiply SEM by appropriate t-value
                # Use minimum valid count across frequencies for conservative t-value
                min_valid_trials = np.min(valid_counts[valid_counts > 1])  # Avoid division by zero
                if min_valid_trials > 1:
                    t_value = stats.t.ppf(0.975, min_valid_trials - 1)  # 95% CI, two-tailed
                    error_plv[pop_name] = t_value * sem_plv
                else:
                    error_plv[pop_name] = np.full_like(sem_plv, np.nan)

            elif error_type == "sem":
                # Calculate standard error of the mean
                valid_counts = np.sum(~np.isnan(all_plv_data[pop_name]), axis=0)
                error_plv[pop_name] = np.nanstd(all_plv_data[pop_name], axis=0, ddof=1) / np.sqrt(
                    valid_counts
                )

            elif error_type == "std":
                # Calculate standard deviation
                error_plv[pop_name] = np.nanstd(all_plv_data[pop_name], axis=0, ddof=1)

    # Create the combined plot
    plt.figure(figsize=(12, 8))

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
    plt.show()
