import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats


def plot_spike_power_correlation(correlation_results, frequencies, pop_names):
    """
    Plot the correlation between population spike rates and LFP power.

    Parameters:
    -----------
    correlation_results : dict
        Dictionary with correlation results for calculate_spike_rate_power_correlation
    frequencies : array
        Array of frequencies analyzed
    pop_names : list
        List of population names
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    for pop in pop_names:
        # Extract correlation values for each frequency
        corr_values = []
        valid_freqs = []

        for freq in frequencies:
            if freq in correlation_results[pop]:
                corr_values.append(correlation_results[pop][freq]["correlation"])
                valid_freqs.append(freq)

        # Plot correlation line
        plt.plot(valid_freqs, corr_values, marker="o", label=pop, linewidth=2, markersize=6)

    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Spike Rate-Power Correlation", fontsize=12)
    plt.title("Spike rate LFP power correlation during stimulus", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(frequencies[::2])  # Display every other frequency on x-axis

    # Add horizontal line at zero for reference
    plt.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    # Set y-axis limits to make zero visible
    y_min, y_max = plt.ylim()
    plt.ylim(min(y_min, -0.1), max(y_max, 0.1))

    plt.tight_layout()

    plt.show()


def plot_cycle_with_spike_histograms(phase_data, bins=36, pop_name=None):
    """
    Plot an idealized gamma cycle with spike histograms for different neuron populations.

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


def plot_ppc_by_population(ppc_dict, pop_names, freqs, figsize=(15, 8), title=None):
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
