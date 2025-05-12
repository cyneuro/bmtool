from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ..util.util import load_nodes_from_config


def raster(
    spikes_df: Optional[pd.DataFrame] = None,
    config: Optional[str] = None,
    network_name: Optional[str] = None,
    groupby: Optional[str] = "pop_name",
    ax: Optional[Axes] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    color_map: Optional[Dict[str, str]] = None,
) -> Axes:
    """
    Plots a raster plot of neural spikes, with different colors for each population.

    Parameters:
    ----------
    spikes_df : pd.DataFrame, optional
        DataFrame containing spike data with columns 'timestamps', 'node_ids', and optional 'pop_name'.
    config : str, optional
        Path to the configuration file used to load node data.
    network_name : str, optional
        Specific network name to select from the configuration; if not provided, uses the first network.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the raster; if None, a new figure and axes are created.
    tstart : float, optional
        Start time for filtering spikes; only spikes with timestamps greater than `tstart` will be plotted.
    tstop : float, optional
        Stop time for filtering spikes; only spikes with timestamps less than `tstop` will be plotted.
    color_map : dict, optional
        Dictionary specifying colors for each population. Keys should be population names, and values should be color values.

    Returns:
    -------
    matplotlib.axes.Axes
        Axes with the raster plot.

    Notes:
    -----
    - If `config` is provided, the function merges population names from the node data with `spikes_df`.
    - Each unique population from groupby in `spikes_df` will be represented by a different color if `color_map` is not specified.
    - If `color_map` is provided, it should contain colors for all unique `pop_name` values in `spikes_df`.
    """
    # Initialize axes if none provided
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Filter spikes by time range if specified
    if tstart is not None:
        spikes_df = spikes_df[spikes_df["timestamps"] > tstart]
    if tstop is not None:
        spikes_df = spikes_df[spikes_df["timestamps"] < tstop]

    # Load and merge node population data if config is provided
    if config:
        nodes = load_nodes_from_config(config)
        if network_name:
            nodes = nodes.get(network_name, {})
        else:
            nodes = list(nodes.values())[0] if nodes else {}
            print(
                "Grabbing first network; specify a network name to ensure correct node population is selected."
            )

        # Find common columns, but exclude the join key from the list
        common_columns = spikes_df.columns.intersection(nodes.columns).tolist()
        common_columns = [
            col for col in common_columns if col != "node_ids"
        ]  # Remove our join key from the common list

        # Drop all intersecting columns except the join key column from df2
        spikes_df = spikes_df.drop(columns=common_columns)
        # merge nodes and spikes df
        spikes_df = spikes_df.merge(
            nodes[groupby], left_on="node_ids", right_index=True, how="left"
        )

    # Get unique population names
    unique_pop_names = spikes_df[groupby].unique()

    # Generate colors if no color_map is provided
    if color_map is None:
        cmap = plt.get_cmap("tab10")  # Default colormap
        color_map = {
            pop_name: cmap(i / len(unique_pop_names)) for i, pop_name in enumerate(unique_pop_names)
        }
    else:
        # Ensure color_map contains all population names
        missing_colors = [pop for pop in unique_pop_names if pop not in color_map]
        if missing_colors:
            raise ValueError(f"color_map is missing colors for populations: {missing_colors}")

    # Plot each population with its specified or generated color
    for pop_name, group in spikes_df.groupby(groupby):
        ax.scatter(
            group["timestamps"], group["node_ids"], label=pop_name, color=color_map[pop_name], s=0.5
        )

    # Label axes
    ax.set_xlabel("Time")
    ax.set_ylabel("Node ID")
    ax.legend(title="Population", loc="upper right", framealpha=0.9, markerfirst=False)

    return ax


# uses df from bmtool.analysis.spikes compute_firing_rate_stats
def plot_firing_rate_pop_stats(
    firing_stats: pd.DataFrame,
    groupby: Union[str, List[str]],
    ax: Optional[Axes] = None,
    color_map: Optional[Dict[str, str]] = None,
) -> Axes:
    """
    Plots a bar graph of mean firing rates with error bars (standard deviation).

    Parameters:
    ----------
    firing_stats : pd.DataFrame
        Dataframe containing 'firing_rate_mean' and 'firing_rate_std'.
    groupby : str or list of str
        Column(s) used for grouping.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the bar chart; if None, a new figure and axes are created.
    color_map : dict, optional
        Dictionary specifying colors for each group. Keys should be group names, and values should be color values.

    Returns:
    -------
    matplotlib.axes.Axes
        Axes with the bar plot.
    """
    # Ensure groupby is a list for consistent handling
    if isinstance(groupby, str):
        groupby = [groupby]

    # Create a categorical column for grouping
    firing_stats["group"] = firing_stats[groupby].astype(str).agg("_".join, axis=1)

    # Get unique group names
    unique_groups = firing_stats["group"].unique()

    # Generate colors if no color_map is provided
    if color_map is None:
        cmap = plt.get_cmap("viridis")
        color_map = {group: cmap(i / len(unique_groups)) for i, group in enumerate(unique_groups)}
    else:
        # Ensure color_map contains all groups
        missing_colors = [group for group in unique_groups if group not in color_map]
        if missing_colors:
            raise ValueError(f"color_map is missing colors for groups: {missing_colors}")

    # Create new figure and axes if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Sort data for consistent plotting
    firing_stats = firing_stats.sort_values(by="group")

    # Extract values for plotting
    x_labels = firing_stats["group"]
    means = firing_stats["firing_rate_mean"]
    std_devs = firing_stats["firing_rate_std"]

    # Get colors for each group
    colors = [color_map[group] for group in x_labels]

    # Create bar plot
    bars = ax.bar(x_labels, means, yerr=std_devs, capsize=5, color=colors, edgecolor="black")

    # Add error bars manually with caps
    _, caps, _ = ax.errorbar(
        x=np.arange(len(x_labels)),
        y=means,
        yerr=std_devs,
        fmt="none",
        capsize=5,
        capthick=2,
        color="black",
    )

    # Formatting
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel("Population Group")
    ax.set_ylabel("Mean Firing Rate (spikes/s)")
    ax.set_title("Firing Rate Statistics by Population")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return ax


# uses df from bmtool.analysis.spikes compute_firing_rate_stats
def plot_firing_rate_distribution(
    individual_stats: pd.DataFrame,
    groupby: Union[str, list],
    ax: Optional[Axes] = None,
    color_map: Optional[Dict[str, str]] = None,
    plot_type: Union[str, list] = "box",
    swarm_alpha: float = 0.6,
) -> Axes:
    """
    Plots a distribution of individual firing rates using one or more plot types
    (box plot, violin plot, or swarm plot), overlaying them on top of each other.

    Parameters:
    ----------
    individual_stats : pd.DataFrame
        Dataframe containing individual firing rates and corresponding group labels.
    groupby : str or list of str
        Column(s) used for grouping.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the graph; if None, a new figure and axes are created.
    color_map : dict, optional
        Dictionary specifying colors for each group. Keys should be group names, and values should be color values.
    plot_type : str or list of str, optional
        List of plot types to generate. Options: "box", "violin", "swarm". Default is "box".
    swarm_alpha : float, optional
        Transparency of swarm plot points. Default is 0.6.

    Returns:
    -------
    matplotlib.axes.Axes
        Axes with the selected plot type(s) overlayed.
    """
    # Ensure groupby is a list for consistent handling
    if isinstance(groupby, str):
        groupby = [groupby]

    # Create a categorical column for grouping
    individual_stats["group"] = individual_stats[groupby].astype(str).agg("_".join, axis=1)

    # Validate plot_type (it can be a list or a single type)
    if isinstance(plot_type, str):
        plot_type = [plot_type]

    for pt in plot_type:
        if pt not in ["box", "violin", "swarm"]:
            raise ValueError("plot_type must be one of: 'box', 'violin', 'swarm'.")

    # Get unique groups for coloring
    unique_groups = individual_stats["group"].unique()

    # Generate colors if no color_map is provided
    if color_map is None:
        cmap = plt.get_cmap("viridis")
        color_map = {group: cmap(i / len(unique_groups)) for i, group in enumerate(unique_groups)}

    # Ensure color_map contains all groups
    missing_colors = [group for group in unique_groups if group not in color_map]
    if missing_colors:
        raise ValueError(f"color_map is missing colors for groups: {missing_colors}")

    # Create new figure and axes if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Sort data for consistent plotting
    individual_stats = individual_stats.sort_values(by="group")

    # Loop over each plot type and overlay them
    for pt in plot_type:
        if pt == "box":
            sns.boxplot(
                data=individual_stats,
                x="group",
                y="firing_rate",
                ax=ax,
                palette=color_map,
                width=0.5,
            )
        elif pt == "violin":
            sns.violinplot(
                data=individual_stats,
                x="group",
                y="firing_rate",
                ax=ax,
                palette=color_map,
                inner="quartile",
                alpha=0.4,
            )
        elif pt == "swarm":
            sns.swarmplot(
                data=individual_stats,
                x="group",
                y="firing_rate",
                ax=ax,
                palette=color_map,
                alpha=swarm_alpha,
            )

    # Formatting
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Population Group")
    ax.set_ylabel("Firing Rate (spikes/s)")
    ax.set_title("Firing Rate Distribution for individual cells")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    return ax
