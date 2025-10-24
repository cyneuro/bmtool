"""Plotting functions for neural spikes and firing rates."""

from typing import Dict, List, Optional, Tuple, Union

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
    groupby: str = "pop_name",
    sortby: Optional[str] = None,
    ax: Optional[Axes] = None,
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    color_map: Optional[Dict[str, str]] = None,
    dot_size: float = 0.3,
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
    groupby : str, optional
        Column name to group spikes by for coloring. Default is 'pop_name'.
    sortby : str, optional
        Column name to sort node_ids within each group. If provided, nodes within each population will be sorted by this column.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot the raster; if None, a new figure and axes are created.
    tstart : float, optional
        Start time for filtering spikes; only spikes with timestamps greater than `tstart` will be plotted.
    tstop : float, optional
        Stop time for filtering spikes; only spikes with timestamps less than `tstop` will be plotted.
    color_map : dict, optional
        Dictionary specifying colors for each population. Keys should be population names, and values should be color values.
    dot_size: float, optional
        Size of the dot to display on the scatterplot

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
    sns.set_style("whitegrid")
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
    legend_handles = []
    y_offset = 0  # Track y-position offset for stacking populations
    
    for pop_name, group in spikes_df.groupby(groupby):
        if sortby:
            # Sort by the specified column, putting NaN values at the end
            group_sorted = group.sort_values(by=sortby, na_position='last')
            # Create a mapping from node_ids to consecutive y-positions based on sorted order
            # Use the sorted order to maintain the same sequence for all spikes from same node
            unique_nodes_sorted = group_sorted['node_ids'].drop_duplicates()
            node_to_y = {node_id: y_offset + i for i, node_id in enumerate(unique_nodes_sorted)}
            # Map node_ids to new y-positions for ALL spikes (not just the sorted group)
            y_positions = group['node_ids'].map(node_to_y)
            # Verify no data was lost
            assert len(y_positions) == len(group), f"Data loss detected in population {pop_name}"
            assert y_positions.isna().sum() == 0, f"Unmapped node_ids found in population {pop_name}"
        else:
            y_positions = group['node_ids']
            
        ax.scatter(group["timestamps"], y_positions, color=color_map[pop_name], s=dot_size)
        # Dummy scatter for consistent legend appearance
        handle = ax.scatter([], [], color=color_map[pop_name], label=pop_name, s=20)
        legend_handles.append(handle)
        
        # Update y_offset for next population if sortby is used
        if sortby:
            y_offset += len(unique_nodes_sorted)

    # Label axes
    ax.set_xlabel("Time")
    ax.set_ylabel("Node ID")
    ax.legend(handles=legend_handles, title="Population", loc="upper right", framealpha=0.9)

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
    sns.set_style("whitegrid")
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
    groupby: Union[str, List[str]],
    ax: Optional[Axes] = None,
    color_map: Optional[Dict[str, str]] = None,
    plot_type: Union[str, List[str]] = "box",
    swarm_alpha: float = 0.6,
    logscale: bool = False,
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
    logscale : bool, optional
        If True, use logarithmic scale for the y-axis (default is False).

    Returns:
    -------
    matplotlib.axes.Axes
        Axes with the selected plot type(s) overlayed.
    """
    sns.set_style("whitegrid")
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
                inner="box",
                alpha=0.4,
                cut=0,  # This prevents the KDE from extending beyond the data range
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

    if logscale:
        ax.set_yscale('log')

    return ax


def plot_firing_rate_vs_node_attribute(
    individual_stats: pd.DataFrame,
    groupby: str,
    attribute: str,
    config: Optional[str] = None,
    nodes: Optional[pd.DataFrame] = None,
    network_name: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    dot_size: float = 3,
    color_map: Optional[Dict[str, str]] = None,
) -> plt.Figure:
    """
    Plot firing rate vs node attribute for each group in separate subplots.

    Parameters
    ----------
    individual_stats : pd.DataFrame
        DataFrame containing individual cell firing rates from compute_firing_rate_stats
    groupby : str
        Column name in individual_stats to group plots by
    attribute : str
        Node attribute column name to plot against firing rate
    config : str, optional
        Path to configuration file for loading node data
    nodes : pd.DataFrame, optional
        Pre-loaded node data as alternative to loading from config
    network_name : str, optional
        Name of network to load from config file
    figsize : Tuple[float, float], optional
        Figure dimensions (width, height) in inches
    dot_size : float, optional
        Size of scatter plot points
    color_map : dict, optional
        Dictionary specifying colors for each group. Keys should be group names, and values should be color values.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the subplots

    Raises
    ------
    ValueError
        If neither config nor nodes is provided
        If network_name is missing when using config
        If attribute is not found in nodes DataFrame
        If node_ids column is missing
        If nodes index is not unique
    """
    # Input validation
    if config is None and nodes is None:
        raise ValueError("Must provide either config or nodes")
    if config is not None and nodes is None:
        if network_name is None:
            raise ValueError("network_name required when using config")
        nodes = load_nodes_from_config(config)
    if attribute not in nodes.columns:
        raise ValueError(f"Attribute '{attribute}' not found in nodes DataFrame")

    # Extract node attribute data
    node_attribute = nodes[attribute]

    # Validate data structure
    if "node_ids" not in individual_stats.columns:
        raise ValueError("individual_stats missing required 'node_ids' column")
    if not nodes.index.is_unique:
        raise ValueError("nodes DataFrame must have unique index for merging")

    # Merge firing rate data with node attributes
    merged_df = individual_stats.merge(
        node_attribute, left_on="node_ids", right_index=True, how="left"
    )

    # Setup subplot layout
    max_groups = 15  # Maximum number of subplots to avoid overcrowding
    unique_groups = merged_df[groupby].unique()
    n_groups = min(len(unique_groups), max_groups)

    if len(unique_groups) > max_groups:
        print(f"Warning: Limiting display to {max_groups} groups out of {len(unique_groups)}")
        unique_groups = unique_groups[:max_groups]

    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Generate colors if no color_map is provided
    if color_map is None:
        cmap = plt.get_cmap("tab10")
        color_map = {group: cmap(i / len(unique_groups)) for i, group in enumerate(unique_groups)}
    else:
        # Ensure color_map contains all groups
        missing_colors = [group for group in unique_groups if group not in color_map]
        if missing_colors:
            raise ValueError(f"color_map is missing colors for groups: {missing_colors}")

    # Plot each group
    for i, group in enumerate(unique_groups):
        group_df = merged_df[merged_df[groupby] == group]
        axes[i].scatter(group_df["firing_rate"], group_df[attribute], s=dot_size, color=color_map[group])
        axes[i].set_xlabel("Firing Rate (Hz)")
        axes[i].set_ylabel(attribute)
        
        # Calculate and display mean firing rate in legend
        mean_fr = group_df["firing_rate"].mean()
        axes[i].legend([f"Mean FR: {mean_fr:.2f} Hz"], loc="upper right")
        axes[i].set_title(f"{groupby}: {group}")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig


def plot_firing_rate_histogram(
    individual_stats: pd.DataFrame,
    groupby: str = "pop_name",
    ax: Optional[Axes] = None,
    color_map: Optional[Dict[str, str]] = None,
    bins: int = 30,
    alpha: float = 0.7,
    figsize: Tuple[float, float] = (12, 8),
    stacked: bool = False,
    logscale: bool = False,
    min_fr: Optional[float] = None,
) -> plt.Figure:
    """
    Plot histograms of firing rates for each population group.

    Parameters:
    ----------
    individual_stats : pd.DataFrame
        DataFrame containing individual firing rates with group labels.
    groupby : str, optional
        Column name to group by (default is "pop_name").
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot; if None, a new figure is created.
    color_map : dict, optional
        Dictionary specifying colors for each group. Keys should be group names, and values should be color values.
    bins : int, optional
        Number of bins for the histogram (default is 30).
    alpha : float, optional
        Transparency level for the histograms (default is 0.7).
    figsize : Tuple[float, float], optional
        Figure size if creating a new figure (default is (12, 8)).
    stacked : bool, optional
        If True, plot all histograms on a single axes stacked (default is False).
    logscale : bool, optional
        If True, use logarithmic scale for the x-axis (default is False).
    min_fr : float, optional
        Minimum firing rate for log scale bins (default is None).

    Returns:
    -------
    matplotlib.figure.Figure
        Figure containing the histogram subplots.
    """
    sns.set_style("whitegrid")

    # Get unique groups
    unique_groups = individual_stats[groupby].unique()

    # Generate colors if no color_map is provided
    if color_map is None:
        cmap = plt.get_cmap("tab10")
        color_map = {group: cmap(i / len(unique_groups)) for i, group in enumerate(unique_groups)}
    else:
        # Ensure color_map contains all groups
        missing_colors = [group for group in unique_groups if group not in color_map]
        if missing_colors:
            raise ValueError(f"color_map is missing colors for groups: {missing_colors}")

    # Group data by population
    pop_fr = {}
    for group in unique_groups:
        pop_fr[group] = individual_stats[individual_stats[groupby] == group]["firing_rate"].values

    if logscale and min_fr is not None:
        pop_fr = {p: np.fmax(fr, min_fr) for p, fr in pop_fr.items()}
    fr = np.concatenate(list(pop_fr.values()))
    if logscale:
        fr = fr[fr > 0]
        bins_array = np.geomspace(fr.min(), fr.max(), bins + 1)
    else:
        bins_array = np.linspace(fr.min(), fr.max(), bins + 1)

    # Setup subplot layout or single plot
    n_groups = len(unique_groups)
    if stacked or not stacked:  # Always use single ax for now, since stacked means overlaid
        fig, ax = plt.subplots(figsize=figsize)
    else:
        # If not stacked, but since overlaid is default, perhaps keep as is
        fig, ax = plt.subplots(figsize=figsize)

    if stacked:
        ax.hist(pop_fr.values(), bins=bins_array, label=list(pop_fr.keys()),
                color=[color_map[p] for p in pop_fr.keys()], stacked=True)
    else:
        for p, fr_vals in pop_fr.items():
            ax.hist(fr_vals, bins=bins_array, label=p, color=color_map[p], alpha=alpha)

    if logscale:
        ax.set_xscale('log')
        plt.draw()
        xt = ax.get_xticks()
        xtl = [f'{x:g}' for x in xt]
        if min_fr is not None:
            xt = np.append(xt, min_fr)
            xtl.append('0')
        ax.set_xticks(xt)
        ax.set_xticklabels(xtl)

    ax.set_xlim(bins_array[0], bins_array[-1])
    ax.legend(loc='upper right')
    ax.set_title('Firing Rate Histogram')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Count')
    return fig


