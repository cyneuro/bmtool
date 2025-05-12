"""
Want to be able to take multiple plot names in and plot them all at the same time, to save time
https://stackoverflow.com/questions/458209/is-there-a-way-to-detach-matplotlib-plots-so-that-the-computation-can-continue
"""
import re
import statistics

import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from neuron import h

from ..util import util

use_description = """

Plot BMTK models easily.

python -m bmtool.plot
"""


def is_notebook() -> bool:
    """
    Detect if code is running in a Jupyter notebook environment.

    Returns:
    --------
    bool
        True if running in a Jupyter notebook, False otherwise.

    Notes:
    ------
    This is used to determine whether to call plt.show() explicitly or
    rely on Jupyter's automatic display functionality.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def total_connection_matrix(
    config=None,
    title=None,
    sources=None,
    targets=None,
    sids=None,
    tids=None,
    no_prepend_pop=False,
    save_file=None,
    synaptic_info="0",
    include_gap=True,
):
    """
    Generate a plot displaying total connections or other synaptic statistics.

    Parameters:
    -----------
    config : str
        Path to a BMTK simulation config file.
    title : str, optional
        Title for the plot. If None, a default title will be used.
    sources : str
        Comma-separated string of network names to use as sources.
    targets : str
        Comma-separated string of network names to use as targets.
    sids : str, optional
        Comma-separated string of source node identifiers to filter.
    tids : str, optional
        Comma-separated string of target node identifiers to filter.
    no_prepend_pop : bool, optional
        If True, don't display population name before sid or tid in the plot.
    save_file : str, optional
        Path to save the plot. If None, plot is not saved.
    synaptic_info : str, optional
        Type of information to display:
        - '0': Total connections (default)
        - '1': Mean and standard deviation of connections
        - '2': All synapse .mod files used
        - '3': All synapse .json files used
    include_gap : bool, optional
        If True, include gap junctions and chemical synapses in the analysis.
        If False, only include chemical synapses.

    Returns:
    --------
    None
        The function generates and displays a plot.
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []
    text, num, source_labels, target_labels = util.connection_totals(
        config=config,
        nodes=None,
        edges=None,
        sources=sources,
        targets=targets,
        sids=sids,
        tids=tids,
        prepend_pop=not no_prepend_pop,
        synaptic_info=synaptic_info,
        include_gap=include_gap,
    )

    if title is None or title == "":
        title = "Total Connections"
    if synaptic_info == "1":
        title = "Mean and Stdev # of Conn on Target"
    if synaptic_info == "2":
        title = "All Synapse .mod Files Used"
    if synaptic_info == "3":
        title = "All Synapse .json Files Used"
    plot_connection_info(
        text, num, source_labels, target_labels, title, syn_info=synaptic_info, save_file=save_file
    )
    return


def percent_connection_matrix(
    config=None,
    nodes=None,
    edges=None,
    title=None,
    sources=None,
    targets=None,
    sids=None,
    tids=None,
    no_prepend_pop=False,
    save_file=None,
    method="total",
    include_gap=True,
):
    """
    Generates a plot showing the percent connectivity of a network
    config: A BMTK simulation config
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    method: what percent to displace on the graph 'total','uni',or 'bi' for total connections, unidirectional connections or bidirectional connections
    save_file: If plot should be saved
    include_gap: Determines if connectivity shown should include gap junctions + chemical synapses. False will only include chemical
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")

    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []
    text, num, source_labels, target_labels = util.percent_connections(
        config=config,
        nodes=None,
        edges=None,
        sources=sources,
        targets=targets,
        sids=sids,
        tids=tids,
        prepend_pop=not no_prepend_pop,
        method=method,
        include_gap=include_gap,
    )
    if title is None or title == "":
        title = "Percent Connectivity"

    plot_connection_info(text, num, source_labels, target_labels, title, save_file=save_file)
    return


def probability_connection_matrix(
    config=None,
    nodes=None,
    edges=None,
    title=None,
    sources=None,
    targets=None,
    sids=None,
    tids=None,
    no_prepend_pop=False,
    save_file=None,
    dist_X=True,
    dist_Y=True,
    dist_Z=True,
    bins=8,
    line_plot=False,
    verbose=False,
    include_gap=True,
):
    """
    Generates probability graphs
    need to look into this more to see what it does
    needs model_template to be defined to work
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []

    throwaway, data, source_labels, target_labels = util.connection_probabilities(
        config=config,
        nodes=None,
        edges=None,
        sources=sources,
        targets=targets,
        sids=sids,
        tids=tids,
        prepend_pop=not no_prepend_pop,
        dist_X=dist_X,
        dist_Y=dist_Y,
        dist_Z=dist_Z,
        num_bins=bins,
        include_gap=include_gap,
    )
    if not data.any():
        return
    if data[0][0] == -1:
        return
    # plot_connection_info(data,source_labels,target_labels,title, save_file=save_file)

    # plt.clf()# clears previous plots
    np.seterr(divide="ignore", invalid="ignore")
    num_src, num_tar = data.shape
    fig, axes = plt.subplots(nrows=num_src, ncols=num_tar, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for x in range(num_src):
        for y in range(num_tar):
            ns = data[x][y]["ns"]
            bins = data[x][y]["bins"]

            XX = bins[:-1]
            YY = ns[0] / ns[1]

            if line_plot:
                axes[x, y].plot(XX, YY)
            else:
                axes[x, y].bar(XX, YY)

            if x == num_src - 1:
                axes[x, y].set_xlabel(target_labels[y])
            if y == 0:
                axes[x, y].set_ylabel(source_labels[x])

            if verbose:
                print("Source: [" + source_labels[x] + "] | Target: [" + target_labels[y] + "]")
                print("X:")
                print(XX)
                print("Y:")
                print(YY)

    tt = "Distance Probability Matrix"
    if title:
        tt = title
    st = fig.suptitle(tt, fontsize=14)
    fig.text(0.5, 0.04, "Target", ha="center")
    fig.text(0.04, 0.5, "Source", va="center", rotation="vertical")
    notebook = is_notebook
    if not notebook:
        fig.show()

    return


def convergence_connection_matrix(
    config=None,
    title=None,
    sources=None,
    targets=None,
    sids=None,
    tids=None,
    no_prepend_pop=False,
    save_file=None,
    convergence=True,
    method="mean+std",
    include_gap=True,
    return_dict=None,
):
    """
    Generates connection plot displaying convergence data
    config: A BMTK simulation config
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    save_file: If plot should be saved
    method: 'mean','min','max','stdev' or 'mean+std' connvergence plot
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    return divergence_connection_matrix(
        config,
        title,
        sources,
        targets,
        sids,
        tids,
        no_prepend_pop,
        save_file,
        convergence,
        method,
        include_gap=include_gap,
        return_dict=return_dict,
    )


def divergence_connection_matrix(
    config=None,
    title=None,
    sources=None,
    targets=None,
    sids=None,
    tids=None,
    no_prepend_pop=False,
    save_file=None,
    convergence=False,
    method="mean+std",
    include_gap=True,
    return_dict=None,
):
    """
    Generates connection plot displaying divergence data
    config: A BMTK simulation config
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    save_file: If plot should be saved
    method: 'mean','min','max','stdev', and 'mean+std' for divergence plot
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []

    syn_info, data, source_labels, target_labels = util.connection_divergence(
        config=config,
        nodes=None,
        edges=None,
        sources=sources,
        targets=targets,
        sids=sids,
        tids=tids,
        prepend_pop=not no_prepend_pop,
        convergence=convergence,
        method=method,
        include_gap=include_gap,
    )

    # data, labels = util.connection_divergence_average(config=config,nodes=nodes,edges=edges,populations=populations)

    if title is None or title == "":
        if method == "min":
            title = "Minimum "
        elif method == "max":
            title = "Maximum "
        elif method == "std":
            title = "Standard Deviation "
        elif method == "mean":
            title = "Mean "
        else:
            title = "Mean + Std "

        if convergence:
            title = title + "Synaptic Convergence"
        else:
            title = title + "Synaptic Divergence"
    if return_dict:
        dict = plot_connection_info(
            syn_info,
            data,
            source_labels,
            target_labels,
            title,
            save_file=save_file,
            return_dict=return_dict,
        )
        return dict
    else:
        plot_connection_info(
            syn_info, data, source_labels, target_labels, title, save_file=save_file
        )
        return


def gap_junction_matrix(
    config=None,
    title=None,
    sources=None,
    targets=None,
    sids=None,
    tids=None,
    no_prepend_pop=False,
    save_file=None,
    method="convergence",
):
    """
    Generates connection plot displaying gap junction data.
    config: A BMTK simulation config
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    save_file: If plot should be saved
    type:'convergence' or 'percent' connections
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    if method != "convergence" and method != "percent":
        raise Exception("type must be 'convergence' or 'percent'")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []
    syn_info, data, source_labels, target_labels = util.gap_junction_connections(
        config=config,
        nodes=None,
        edges=None,
        sources=sources,
        targets=targets,
        sids=sids,
        tids=tids,
        prepend_pop=not no_prepend_pop,
        method=method,
    )

    def filter_rows(syn_info, data, source_labels, target_labels):
        """
        Filters out rows in a connectivity matrix that contain only NaN or zero values.

        This function is used to clean up connection matrices by removing rows that have no meaningful data,
        which helps create more informative visualizations of network connectivity.

        Parameters:
        -----------
        syn_info : numpy.ndarray
            Array containing synaptic information corresponding to the data matrix.
        data : numpy.ndarray
            2D matrix containing connectivity data with rows representing sources and columns representing targets.
        source_labels : list
            List of labels for the source populations corresponding to rows in the data matrix.
        target_labels : list
            List of labels for the target populations corresponding to columns in the data matrix.

        Returns:
        --------
        tuple
            A tuple containing the filtered (syn_info, data, source_labels, target_labels) with invalid rows removed.
        """
        # Identify rows with all NaN or all zeros
        valid_rows = ~np.all(np.isnan(data), axis=1) & ~np.all(data == 0, axis=1)

        # Filter rows based on valid_rows mask
        new_syn_info = syn_info[valid_rows]
        new_data = data[valid_rows]
        new_source_labels = np.array(source_labels)[valid_rows]

        return new_syn_info, new_data, new_source_labels, target_labels

    def filter_rows_and_columns(syn_info, data, source_labels, target_labels):
        """
        Filters out both rows and columns in a connectivity matrix that contain only NaN or zero values.

        This function performs a two-step filtering process: first removing rows with no data,
        then transposing the matrix and removing columns with no data (by treating them as rows).
        This creates a cleaner, more informative connectivity matrix visualization.

        Parameters:
        -----------
        syn_info : numpy.ndarray
            Array containing synaptic information corresponding to the data matrix.
        data : numpy.ndarray
            2D matrix containing connectivity data with rows representing sources and columns representing targets.
        source_labels : list
            List of labels for the source populations corresponding to rows in the data matrix.
        target_labels : list
            List of labels for the target populations corresponding to columns in the data matrix.

        Returns:
        --------
        tuple
            A tuple containing the filtered (syn_info, data, source_labels, target_labels) with both
            invalid rows and columns removed.
        """
        # Filter rows first
        syn_info, data, source_labels, target_labels = filter_rows(
            syn_info, data, source_labels, target_labels
        )

        # Transpose data to filter columns
        transposed_syn_info = np.transpose(syn_info)
        transposed_data = np.transpose(data)
        transposed_source_labels = target_labels
        transposed_target_labels = source_labels

        # Filter columns (by treating them as rows in transposed data)
        (
            transposed_syn_info,
            transposed_data,
            transposed_source_labels,
            transposed_target_labels,
        ) = filter_rows(
            transposed_syn_info, transposed_data, transposed_source_labels, transposed_target_labels
        )

        # Transpose back to original orientation
        filtered_syn_info = np.transpose(transposed_syn_info)
        filtered_data = np.transpose(transposed_data)
        filtered_source_labels = transposed_target_labels  # Back to original source_labels
        filtered_target_labels = transposed_source_labels  # Back to original target_labels

        return filtered_syn_info, filtered_data, filtered_source_labels, filtered_target_labels

    syn_info, data, source_labels, target_labels = filter_rows_and_columns(
        syn_info, data, source_labels, target_labels
    )

    if title is None or title == "":
        title = "Gap Junction"
        if method == "convergence":
            title += " Syn Convergence"
        elif method == "percent":
            title += " Percent Connectivity"
    plot_connection_info(syn_info, data, source_labels, target_labels, title, save_file=save_file)
    return


def connection_histogram(
    config=None,
    nodes=None,
    edges=None,
    sources=[],
    targets=[],
    sids=[],
    tids=[],
    no_prepend_pop=True,
    synaptic_info="0",
    source_cell=None,
    target_cell=None,
    include_gap=True,
):
    """
    Generates histogram of number of connections individual cells in a population receieve from another population
    config: A BMTK simulation config
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    source_cell: where connections are coming from
    target_cell: where connections on coming onto
    save_file: If plot should be saved
    """

    def connection_pair_histogram(**kwargs):
        """
        Creates a histogram showing the distribution of connection counts between a specific source and target cell type.

        This function is designed to be used with the relation_matrix utility and will only create histograms
        for the specified source and target cell types, ignoring all other combinations.

        Parameters:
        -----------
        kwargs : dict
            Dictionary containing the following keys:
            - edges: DataFrame containing edge information
            - sid: Column name for source ID type in the edges DataFrame
            - tid: Column name for target ID type in the edges DataFrame
            - source_id: Value to filter edges by source ID type
            - target_id: Value to filter edges by target ID type

        Global parameters used:
        ---------------------
        source_cell : str
            The source cell type to plot.
        target_cell : str
            The target cell type to plot.
        include_gap : bool
            Whether to include gap junctions in the analysis. If False, gap junctions are excluded.

        Returns:
        --------
        None
            Displays a histogram showing the distribution of connection counts.
        """
        edges = kwargs["edges"]
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]
        if source_id == source_cell and target_id == target_cell:
            temp = edges[
                (edges[source_id_type] == source_id) & (edges[target_id_type] == target_id)
            ]
            if not include_gap:
                temp = temp[~temp["is_gap_junction"]]
            node_pairs = temp.groupby("target_node_id")["source_node_id"].count()
            try:
                conn_mean = statistics.mean(node_pairs.values)
                conn_std = statistics.stdev(node_pairs.values)
                conn_median = statistics.median(node_pairs.values)
                label = "mean {:.2f} std {:.2f} median {:.2f}".format(
                    conn_mean, conn_std, conn_median
                )
            except:  # lazy fix for std not calculated with 1 node
                conn_mean = statistics.mean(node_pairs.values)
                conn_median = statistics.median(node_pairs.values)
                label = "mean {:.2f} median {:.2f}".format(conn_mean, conn_median)
            plt.hist(node_pairs.values, density=False, bins="auto", stacked=True, label=label)
            plt.legend()
            plt.xlabel("# of conns from {} to {}".format(source_cell, target_cell))
            plt.ylabel("# of cells")
            plt.show()
        else:  # dont care about other cell pairs so pass
            pass

    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []
    util.relation_matrix(
        config,
        nodes,
        edges,
        sources,
        targets,
        sids,
        tids,
        not no_prepend_pop,
        relation_func=connection_pair_histogram,
        synaptic_info=synaptic_info,
    )


def connection_distance(
    config: str,
    sources: str,
    targets: str,
    source_cell_id: int,
    target_id_type: str,
    ignore_z: bool = False,
) -> None:
    """
    Plots the 3D spatial distribution of target nodes relative to a source node
    and a histogram of distances from the source node to each target node.

    Parameters:
    ----------
    config: (str) A BMTK simulation config
    sources: (str) network name(s) to plot
    targets: (str) network name(s) to plot
    source_cell_id : (int) ID of the source cell for calculating distances to target nodes.
    target_id_type : (str) A string to filter target nodes based off the target_query.
    ignore_z : (bool) A bool to ignore_z axis or not for when calculating distance default is False

    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    # if source != target:
    # raise Exception("Code is setup for source and target to be the same! Look at source code for function to add feature")

    # Load nodes and edges based on config file
    nodes, edges = util.load_nodes_edges_from_config(config)

    edge_network = sources + "_to_" + targets
    node_network = sources

    # Filter edges to obtain connections originating from the source node
    edge = edges[edge_network]
    edge = edge[edge["source_node_id"] == source_cell_id]
    if target_id_type:
        edge = edge[edge["target_query"].str.contains(target_id_type, na=False)]

    target_node_ids = edge["target_node_id"]

    # Filter nodes to obtain only the target and source nodes
    node = nodes[node_network]
    target_nodes = node.loc[node.index.isin(target_node_ids)]
    source_node = node.loc[node.index == source_cell_id]

    # Calculate distances between source node and each target node
    if ignore_z:
        target_positions = target_nodes[["pos_x", "pos_y"]].values
        source_position = np.array(
            [source_node["pos_x"], source_node["pos_y"]]
        ).ravel()  # Ensure 1D shape
    else:
        target_positions = target_nodes[["pos_x", "pos_y", "pos_z"]].values
        source_position = np.array(
            [source_node["pos_x"], source_node["pos_y"], source_node["pos_z"]]
        ).ravel()  # Ensure 1D shape
    distances = np.linalg.norm(target_positions - source_position, axis=1)

    # Plot positions of source and target nodes in 3D space or 2D
    if ignore_z:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.scatter(target_nodes["pos_x"], target_nodes["pos_y"], c="blue", label="target cells")
        ax.scatter(source_node["pos_x"], source_node["pos_y"], c="red", label="source cell")
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            target_nodes["pos_x"],
            target_nodes["pos_y"],
            target_nodes["pos_z"],
            c="blue",
            label="target cells",
        )
        ax.scatter(
            source_node["pos_x"],
            source_node["pos_y"],
            source_node["pos_z"],
            c="red",
            label="source cell",
        )

    # Optional: Add text annotations for distances
    # for i, distance in enumerate(distances):
    #     ax.text(target_nodes['pos_x'].iloc[i], target_nodes['pos_y'].iloc[i], target_nodes['pos_z'].iloc[i],
    #             f'{distance:.2f}', color='black', fontsize=8, ha='center')

    plt.legend()
    plt.show()

    # Plot distances in a separate 2D plot
    plt.figure(figsize=(8, 6))
    plt.hist(distances, bins=20, color="blue", edgecolor="black")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.title("Distance from Source Node to Each Target Node")
    plt.grid(True)
    plt.show()


def edge_histogram_matrix(
    config=None,
    sources=None,
    targets=None,
    sids=None,
    tids=None,
    no_prepend_pop=None,
    edge_property=None,
    time=None,
    time_compare=None,
    report=None,
    title=None,
    save_file=None,
):
    """
    Generates a matrix of histograms showing the distribution of edge properties between different populations.

    This function creates a grid of histograms where each cell in the grid represents the distribution of a
    specific edge property (e.g., synaptic weights, delays) between a source population (row) and
    target population (column).

    Parameters:
    -----------
    config : str
        Path to a BMTK simulation config file.
    sources : str
        Comma-separated list of source network names.
    targets : str
        Comma-separated list of target network names.
    sids : str, optional
        Comma-separated list of source node identifiers to filter by.
    tids : str, optional
        Comma-separated list of target node identifiers to filter by.
    no_prepend_pop : bool, optional
        If True, population names are not prepended to node identifiers in the display.
    edge_property : str
        The edge property to analyze and display in the histograms (e.g., 'syn_weight', 'delay').
    time : int, optional
        Time point to analyze from a time series report.
    time_compare : int, optional
        Second time point for comparison with 'time'.
    report : str, optional
        Name of the report to analyze.
    title : str, optional
        Custom title for the plot.
    save_file : str, optional
        Path to save the generated plot.

    Returns:
    --------
    None
        Displays a matrix of histograms.
    """

    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []

    if time_compare:
        time_compare = int(time_compare)

    data, source_labels, target_labels = util.edge_property_matrix(
        edge_property,
        nodes=None,
        edges=None,
        config=config,
        sources=sources,
        targets=targets,
        sids=sids,
        tids=tids,
        prepend_pop=not no_prepend_pop,
        report=report,
        time=time,
        time_compare=time_compare,
    )

    # Fantastic resource
    # https://stackoverflow.com/questions/7941207/is-there-a-function-to-make-scatterplot-matrices-in-matplotlib
    num_src, num_tar = data.shape
    fig, axes = plt.subplots(nrows=num_src, ncols=num_tar, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for x in range(num_src):
        for y in range(num_tar):
            axes[x, y].hist(data[x][y])

            if x == num_src - 1:
                axes[x, y].set_xlabel(target_labels[y])
            if y == 0:
                axes[x, y].set_ylabel(source_labels[x])

    tt = edge_property + " Histogram Matrix"
    if title:
        tt = title
    st = fig.suptitle(tt, fontsize=14)
    fig.text(0.5, 0.04, "Target", ha="center")
    fig.text(0.04, 0.5, "Source", va="center", rotation="vertical")
    plt.draw()


def distance_delay_plot(
    simulation_config: str, source: str, target: str, group_by: str, sid: str, tid: str
) -> None:
    """
    Plots the relationship between the distance and delay of connections between nodes in a neural network simulation.

    This function loads the node and edge data from a simulation configuration file, filters nodes by population or group,
    identifies connections (edges) between source and target node populations, calculates the Euclidean distance between
    connected nodes, and plots the delay as a function of distance.

    Args:
        simulation_config (str): Path to the simulation config file
        source (str): The name of the source population in the edge data.
        target (str): The name of the target population in the edge data.
        group_by (str): Column name to group nodes by (e.g., population name).
        sid (str): Identifier for the source group (e.g., 'PN').
        tid (str): Identifier for the target group (e.g., 'PN').

    Returns:
        None: The function creates and displays a scatter plot of distance vs delay.
    """
    nodes, edges = util.load_nodes_edges_from_config(simulation_config)
    nodes = nodes[target]
    # node id is index of nodes df
    node_id_source = nodes[nodes[group_by] == sid].index
    node_id_target = nodes[nodes[group_by] == tid].index

    edges = edges[f"{source}_to_{target}"]
    edges = edges[
        edges["source_node_id"].isin(node_id_source) & edges["target_node_id"].isin(node_id_target)
    ]

    stuff_to_plot = []
    for index, row in edges.iterrows():
        try:
            source_node = row["source_node_id"]
            target_node = row["target_node_id"]

            source_pos = nodes.loc[[source_node], ["pos_x", "pos_y", "pos_z"]]
            target_pos = nodes.loc[[target_node], ["pos_x", "pos_y", "pos_z"]]

            distance = np.linalg.norm(source_pos.values - target_pos.values)

            delay = row["delay"]  # This line may raise KeyError
            stuff_to_plot.append([distance, delay])

        except KeyError as e:
            print(f"KeyError: Missing key {e} in either edge properties or node positions.")
        except IndexError as e:
            print(f"IndexError: Node ID {source_node} or {target_node} not found in nodes.")
        except Exception as e:
            print(f"Unexpected error at edge index {index}: {e}")

    plt.scatter([x[0] for x in stuff_to_plot], [x[1] for x in stuff_to_plot])
    plt.xlabel("Distance")
    plt.ylabel("Delay")
    plt.title(f"Distance vs Delay for edge between {sid} and {tid}")
    plt.show()


def plot_synapse_location_histograms(config, target_model, source=None, target=None):
    """
    generates a histogram of the positions of the synapses on a cell broken down by section
    config: a BMTK config
    target_model: the name of the model_template used when building the BMTK node
    source: The source BMTK network
    target: The target BMTK network
    """
    # Load mechanisms and template

    util.load_templates_from_config(config)

    # Load node and edge data
    nodes, edges = util.load_nodes_edges_from_config(config)
    nodes = nodes[source]
    edges = edges[f"{source}_to_{target}"]

    # Map target_node_id to model_template
    edges["target_model_template"] = edges["target_node_id"].map(nodes["model_template"])

    # Map source_node_id to pop_name
    edges["source_pop_name"] = edges["source_node_id"].map(nodes["pop_name"])

    edges = edges[edges["target_model_template"] == target_model]

    # Create the cell model from target model
    cell = getattr(h, target_model.split(":")[1])()

    # Create a mapping from section index to section name
    section_id_to_name = {}
    for idx, sec in enumerate(cell.all):
        section_id_to_name[idx] = sec.name()

    # Add a new column with section names based on afferent_section_id
    edges["afferent_section_name"] = edges["afferent_section_id"].map(section_id_to_name)

    # Get unique sections and source populations
    unique_pops = edges["source_pop_name"].unique()

    # Filter to only include sections with data
    section_counts = edges["afferent_section_name"].value_counts()
    sections_with_data = section_counts[section_counts > 0].index.tolist()

    # Create a figure with subplots for each section
    plt.figure(figsize=(8, 12))

    # Color map for source populations
    color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_pops)))
    pop_colors = {pop: color for pop, color in zip(unique_pops, color_map)}

    # Create a histogram for each section
    for i, section in enumerate(sections_with_data):
        ax = plt.subplot(len(sections_with_data), 1, i + 1)

        # Get data for this section
        section_data = edges[edges["afferent_section_name"] == section]

        # Group by source population
        for pop_name, pop_group in section_data.groupby("source_pop_name"):
            if len(pop_group) > 0:
                ax.hist(
                    pop_group["afferent_section_pos"],
                    bins=15,
                    alpha=0.7,
                    label=pop_name,
                    color=pop_colors[pop_name],
                )

        # Set title and labels
        ax.set_title(f"{section}", fontsize=10)
        ax.set_xlabel("Section Position", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        # Only add legend to the first plot
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.suptitle(
        "Connection Distribution by Cell Section and Source Population", fontsize=16, y=1.02
    )
    if is_notebook:
        plt.show()
    else:
        pass

    # Create a summary table
    print("Summary of connections by section and source population:")
    pivot_table = edges.pivot_table(
        values="afferent_section_id",
        index="afferent_section_name",
        columns="source_pop_name",
        aggfunc="count",
        fill_value=0,
    )
    print(pivot_table)


def plot_connection_info(
    text, num, source_labels, target_labels, title, syn_info="0", save_file=None, return_dict=None
):
    """
    Function to plot connection information as a heatmap, including handling missing source and target values.
    If there is no source or target, set the value to 0.
    """

    # Ensure text dimensions match num dimensions
    num_source = len(source_labels)
    num_target = len(target_labels)

    # Set color map
    matplotlib.rc("image", cmap="viridis")

    # Create figure and axis for the plot
    fig1, ax1 = plt.subplots(figsize=(num_source, num_target))
    num = np.nan_to_num(num, nan=0)  # replace NaN with 0
    im1 = ax1.imshow(num)

    # Set ticks and labels for source and target
    ax1.set_xticks(list(np.arange(len(target_labels))))
    ax1.set_yticks(list(np.arange(len(source_labels))))
    ax1.set_xticklabels(target_labels)
    ax1.set_yticklabels(source_labels, size=12, weight="semibold")

    # Rotate the tick labels for better visibility
    plt.setp(
        ax1.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        size=12,
        weight="semibold",
    )

    # Dictionary to store connection information
    graph_dict = {}

    # Loop over data dimensions and create text annotations
    for i in range(num_source):
        for j in range(num_target):
            # Get the edge info, or set it to '0' if it's missing
            edge_info = text[i, j] if text[i, j] is not None else 0

            # Initialize the dictionary for the source node if not already done
            if source_labels[i] not in graph_dict:
                graph_dict[source_labels[i]] = {}

            # Add edge info for the target node
            graph_dict[source_labels[i]][target_labels[j]] = edge_info

            # Set text annotations based on syn_info type
            if syn_info == "2" or syn_info == "3":
                if num_source > 8 and num_source < 20:
                    fig_text = ax1.text(
                        j,
                        i,
                        edge_info,
                        ha="center",
                        va="center",
                        color="w",
                        rotation=37.5,
                        size=8,
                        weight="semibold",
                    )
                elif num_source > 20:
                    fig_text = ax1.text(
                        j,
                        i,
                        edge_info,
                        ha="center",
                        va="center",
                        color="w",
                        rotation=37.5,
                        size=7,
                        weight="semibold",
                    )
                else:
                    fig_text = ax1.text(
                        j,
                        i,
                        edge_info,
                        ha="center",
                        va="center",
                        color="w",
                        rotation=37.5,
                        size=11,
                        weight="semibold",
                    )
            else:
                fig_text = ax1.text(
                    j, i, edge_info, ha="center", va="center", color="w", size=11, weight="semibold"
                )

    # Set labels and title for the plot
    ax1.set_ylabel("Source", size=11, weight="semibold")
    ax1.set_xlabel("Target", size=11, weight="semibold")
    ax1.set_title(title, size=20, weight="semibold")

    # Display the plot or save it based on the environment and arguments
    notebook = is_notebook()  # Check if running in a Jupyter notebook
    if not notebook:
        fig1.show()

    if save_file:
        plt.savefig(save_file)

    if return_dict:
        return graph_dict
    else:
        return


def connector_percent_matrix(
    csv_path: str = None,
    exclude_strings=None,
    assemb_key=None,
    title: str = "Percent connection matrix",
    pop_order=None,
) -> None:
    """
    Generates and plots a connection matrix based on connection probabilities from a CSV file produced by bmtool.connector.

    This function is useful for visualizing percent connectivity while factoring in population distance and other parameters.
    It processes the connection data by filtering the 'Source' and 'Target' columns in the CSV, and displays the percentage of
    connected pairs for each population combination in a matrix.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the connection data. The CSV should be an output from the bmtool.connector
        classes, specifically generated by the `save_connection_report()` function.
    exclude_strings : list of str, optional
        List of strings to exclude rows where 'Source' or 'Target' contain these strings.
    title : str, optional, default='Percent connection matrix'
        Title for the generated plot.
    pop_order : list of str, optional
        List of population labels to specify the order for the x- and y-ticks in the plot.

    Returns:
    --------
    None
        Displays a heatmap plot of the connection matrix, showing the percentage of connected pairs between populations.
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)

    # Choose the column to display
    selected_column = "Percent connectionivity within possible connections"

    # Filter the DataFrame based on exclude_strings
    def filter_dataframe(df, column_name, exclude_strings):
        def process_string(string):
            match = re.search(r"\[\'(.*?)\'\]", string)
            if exclude_strings and any(ex_string in string for ex_string in exclude_strings):
                return None
            elif match:
                filtered_string = match.group(1)
                if "Gap" in string:
                    filtered_string = filtered_string + "-Gap"

                if assemb_key:
                    if assemb_key in string:
                        filtered_string = filtered_string + assemb_key

                return filtered_string  # Return matched string

            return string  # If no match, return the original string

        df[column_name] = df[column_name].apply(process_string)
        df = df.dropna(subset=[column_name])

        return df

    df = filter_dataframe(df, "Source", exclude_strings)
    df = filter_dataframe(df, "Target", exclude_strings)

    # process assem rows and combine them into one prob per assem type
    if assemb_key:
        assems = df[df["Source"].str.contains(assemb_key)]
        unique_sources = assems["Source"].unique()

        for source in unique_sources:
            source_assems = assems[assems["Source"] == source]
            unique_targets = source_assems[
                "Target"
            ].unique()  # Filter targets for the current source

            for target in unique_targets:
                # Filter the assemblies with the current source and target
                unique_assems = source_assems[source_assems["Target"] == target]

                # find the prob of a conn
                forward_probs = []
                for _, row in unique_assems.iterrows():
                    selected_percentage = row[selected_column]
                    selected_percentage = [
                        float(p) for p in selected_percentage.strip("[]").split()
                    ]
                    if len(selected_percentage) == 1 or len(selected_percentage) == 2:
                        forward_probs.append(selected_percentage[0])
                    if len(selected_percentage) == 3:
                        forward_probs.append(selected_percentage[0])
                        forward_probs.append(selected_percentage[1])

                mean_probs = np.mean(forward_probs)
                source = source.replace(assemb_key, "")
                target = target.replace(assemb_key, "")
                new_row = pd.DataFrame(
                    {
                        "Source": [source],
                        "Target": [target],
                        "Percent connectionivity within possible connections": [mean_probs],
                        "Percent connectionivity within all connections": [0],
                    }
                )

                df = pd.concat([df, new_row], ignore_index=False)

    # Prepare connection data
    connection_data = {}
    for _, row in df.iterrows():
        source, target, selected_percentage = row["Source"], row["Target"], row[selected_column]
        if isinstance(selected_percentage, str):
            selected_percentage = [float(p) for p in selected_percentage.strip("[]").split()]
        connection_data[(source, target)] = selected_percentage

    # Determine population order
    populations = sorted(list(set(df["Source"].unique()) | set(df["Target"].unique())))
    if pop_order:
        populations = [
            pop for pop in pop_order if pop in populations
        ]  # Order according to pop_order, if provided
    num_populations = len(populations)

    # Create an empty matrix and populate it
    connection_matrix = np.zeros((num_populations, num_populations), dtype=float)
    for (source, target), probabilities in connection_data.items():
        if source in populations and target in populations:
            source_idx = populations.index(source)
            target_idx = populations.index(target)

            if isinstance(probabilities, float):
                connection_matrix[source_idx][target_idx] = probabilities
            elif len(probabilities) == 1:
                connection_matrix[source_idx][target_idx] = probabilities[0]
            elif len(probabilities) == 2:
                connection_matrix[source_idx][target_idx] = probabilities[0]
            elif len(probabilities) == 3:
                connection_matrix[source_idx][target_idx] = probabilities[0]
                connection_matrix[target_idx][source_idx] = probabilities[1]
            else:
                raise Exception("unsupported format")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(connection_matrix, cmap="viridis", interpolation="nearest")

    # Add annotations
    for i in range(num_populations):
        for j in range(num_populations):
            text = ax.text(
                j,
                i,
                f"{connection_matrix[i, j]:.2f}%",
                ha="center",
                va="center",
                color="w",
                size=10,
                weight="semibold",
            )

    # Add colorbar
    plt.colorbar(im, label=f"{selected_column}")

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel("Target Population")
    ax.set_ylabel("Source Population")

    # Set ticks and labels based on populations in specified order
    ax.set_xticks(np.arange(num_populations))
    ax.set_yticks(np.arange(num_populations))
    ax.set_xticklabels(populations, rotation=45, ha="right", size=12, weight="semibold")
    ax.set_yticklabels(populations, size=12, weight="semibold")

    plt.tight_layout()
    plt.show()


def plot_3d_positions(config=None, sources=None, sid=None, title=None, save_file=None, subset=None):
    """
    Plots a 3D graph of all cells with x, y, z location.

    Parameters:
    - config: A BMTK simulation config
    - sources: Which network(s) to plot
    - sid: How to name cell groups
    - title: Plot title
    - save_file: If plot should be saved
    - subset: Take every Nth row. This will make plotting large network graphs easier to see.
    """

    if not config:
        raise Exception("config not defined")

    if sources is None:
        sources = "all"

    # Set group keys (e.g., node types)
    group_keys = sid
    if title is None:
        title = "3D positions"

    # Load nodes from the configuration
    nodes = util.load_nodes_from_config(config)

    # Get the list of populations to plot
    if "all" in sources:
        populations = list(nodes)
    else:
        populations = sources.split(",")

    # Split group_by into list
    group_keys = group_keys.split(",")
    group_keys += (len(populations) - len(group_keys)) * [
        "node_type_id"
    ]  # Extend the array to default values if not enough given
    if len(group_keys) > 1:
        raise Exception("Only one group by is supported currently!")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    handles = []

    for pop in list(nodes):
        if "all" not in populations and pop not in populations:
            continue

        nodes_df = nodes[pop]
        group_key = group_keys[0]

        # If group_key is provided, ensure the column exists in the dataframe
        if group_key is not None:
            if group_key not in nodes_df:
                raise Exception(f"Could not find column '{group_key}' in {pop}")

            groupings = nodes_df.groupby(group_key)
            n_colors = nodes_df[group_key].nunique()
            color_norm = colors.Normalize(vmin=0, vmax=(n_colors - 1))
            scalar_map = cmx.ScalarMappable(norm=color_norm, cmap="hsv")
            color_map = [scalar_map.to_rgba(i) for i in range(n_colors)]
        else:
            groupings = [(None, nodes_df)]
            color_map = ["blue"]

        # Loop over groupings and plot
        for color, (group_name, group_df) in zip(color_map, groupings):
            if "pos_x" not in group_df or "pos_y" not in group_df or "pos_z" not in group_df:
                print(
                    f"Warning: Missing position columns in group '{group_name}' for {pop}. Skipping this group."
                )
                continue  # Skip if position columns are missing

            # Subset the dataframe by taking every Nth row if subset is provided
            if subset is not None:
                group_df = group_df.iloc[::subset]

            h = ax.scatter(
                group_df["pos_x"],
                group_df["pos_y"],
                group_df["pos_z"],
                color=color,
                label=group_name,
            )
            handles.append(h)

    if not handles:
        print("No data to plot.")
        return

    # Set plot title and legend
    plt.title(title)
    plt.legend(handles=handles)

    # Draw the plot
    plt.draw()

    # Save the plot if save_file is provided
    if save_file:
        plt.savefig(save_file)

    # Show the plot if running outside of a notebook
    if not is_notebook:
        plt.show()

    return ax


def plot_3d_cell_rotation(
    config=None,
    sources=None,
    sids=None,
    title=None,
    save_file=None,
    quiver_length=None,
    arrow_length_ratio=None,
    group=None,
    subset=None,
):
    from scipy.spatial.transform import Rotation as R

    if not config:
        raise Exception("config not defined")

    if sources is None:
        sources = ["all"]

    group_keys = sids.split(",") if sids else []

    if title is None:
        title = "Cell rotations"

    nodes = util.load_nodes_from_config(config)

    if "all" in sources:
        populations = list(nodes)
    else:
        populations = sources

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    handles = []

    for nodes_key, group_key in zip(list(nodes), group_keys):
        if "all" not in populations and nodes_key not in populations:
            continue

        nodes_df = nodes[nodes_key]

        if group_key is not None:
            if group_key not in nodes_df.columns:
                raise Exception(f"Could not find column {group_key}")
            groupings = nodes_df.groupby(group_key)

            n_colors = nodes_df[group_key].nunique()
            color_norm = colors.Normalize(vmin=0, vmax=(n_colors - 1))
            scalar_map = cmx.ScalarMappable(norm=color_norm, cmap="hsv")
            color_map = [scalar_map.to_rgba(i) for i in range(n_colors)]
        else:
            groupings = [(None, nodes_df)]
            color_map = ["blue"]

        for color, (group_name, group_df) in zip(color_map, groupings):
            if subset is not None:
                group_df = group_df.iloc[::subset]

            if group and group_name not in group.split(","):
                continue

            if "pos_x" not in group_df or "rotation_angle_xaxis" not in group_df:
                continue

            X = group_df["pos_x"]
            Y = group_df["pos_y"]
            Z = group_df["pos_z"]
            U = group_df["rotation_angle_xaxis"].values
            V = group_df["rotation_angle_yaxis"].values
            W = group_df["rotation_angle_zaxis"].values

            if U is None:
                U = np.zeros(len(X))
            if V is None:
                V = np.zeros(len(Y))
            if W is None:
                W = np.zeros(len(Z))

            # Create rotation matrices from Euler angles
            rotations = R.from_euler("xyz", np.column_stack((U, V, W)), degrees=False)

            # Define initial vectors
            init_vectors = np.column_stack((np.ones(len(X)), np.zeros(len(Y)), np.zeros(len(Z))))

            # Apply rotations to initial vectors
            rots = np.dot(rotations.as_matrix(), init_vectors.T).T

            # Extract x, y, and z components of the rotated vectors
            rot_x = rots[:, 0]
            rot_y = rots[:, 1]
            rot_z = rots[:, 2]

            h = ax.quiver(
                X,
                Y,
                Z,
                rot_x,
                rot_y,
                rot_z,
                color=color,
                label=group_name,
                arrow_length_ratio=arrow_length_ratio,
                length=quiver_length,
            )
            ax.scatter(X, Y, Z, color=color, label=group_name)
            ax.set_xlim([min(X), max(X)])
            ax.set_ylim([min(Y), max(Y)])
            ax.set_zlim([min(Z), max(Z)])
            handles.append(h)

    if not handles:
        return

    plt.title(title)
    plt.legend(handles=handles)
    plt.draw()

    if save_file:
        plt.savefig(save_file)
    notebook = is_notebook
    if not notebook:
        plt.show()
