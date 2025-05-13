import matplotlib.pyplot as plt
import seaborn as sns


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
