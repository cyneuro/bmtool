"""
Module for processing BMTK spikes output.
"""

import os
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import mannwhitneyu

from bmtool.util.util import load_nodes_from_config


def load_spikes_to_df(
    spike_file: str,
    network_name: str,
    sort: bool = True,
    config: str = None,
    groupby: Union[str, List[str]] = "pop_name",
) -> pd.DataFrame:
    """
    Load spike data from an HDF5 file into a pandas DataFrame.

    Parameters
    ----------
    spike_file : str
        Path to the HDF5 file containing spike data
    network_name : str
        The name of the network within the HDF5 file from which to load spike data
    sort : bool, optional
        Whether to sort the DataFrame by 'timestamps' (default: True)
    config : str, optional
        Path to configuration file to label the cell type of each spike (default: None)
    groupby : Union[str, List[str]], optional
        The column(s) to group by (default: 'pop_name')

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing 'node_ids' and 'timestamps' columns from the spike data,
        with additional columns if a config file is provided

    Examples
    --------
    >>> df = load_spikes_to_df("spikes.h5", "cortex")
    >>> df = load_spikes_to_df("spikes.h5", "cortex", config="config.json", groupby=["pop_name", "model_type"])
    """
    with h5py.File(spike_file) as f:
        spikes_df = pd.DataFrame(
            {
                "node_ids": f["spikes"][network_name]["node_ids"],
                "timestamps": f["spikes"][network_name]["timestamps"],
            }
        )

        if sort:
            spikes_df.sort_values(by="timestamps", inplace=True, ignore_index=True)

        if config:
            nodes = load_nodes_from_config(config)
            nodes = nodes[network_name]

            # Convert single string to a list for uniform handling
            if isinstance(groupby, str):
                groupby = [groupby]

            # Ensure all requested columns exist
            missing_cols = [col for col in groupby if col not in nodes.columns]
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in nodes DataFrame.")

            spikes_df = spikes_df.merge(
                nodes[groupby], left_on="node_ids", right_index=True, how="left"
            )

    return spikes_df


def compute_firing_rate_stats(
    df: pd.DataFrame,
    groupby: Union[str, List[str]] = "pop_name",
    start_time: float = None,
    stop_time: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes the firing rates of individual nodes and the mean and standard deviation of firing rates per group.

    Args:
        df (pd.DataFrame): Dataframe containing spike timestamps and node IDs.
        groupby (str or list of str, optional): Column(s) to group by (e.g., 'pop_name' or ['pop_name', 'layer']).
        start_time (float, optional): Start time for the analysis window. Defaults to the minimum timestamp in the data.
        stop_time (float, optional): Stop time for the analysis window. Defaults to the maximum timestamp in the data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - The first DataFrame (`pop_stats`) contains the mean and standard deviation of firing rates per group.
            - The second DataFrame (`individual_stats`) contains the firing rate of each individual node.
    """

    # Ensure groupby is a list
    if isinstance(groupby, str):
        groupby = [groupby]

    # Ensure all columns exist in the dataframe
    for col in groupby:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")

    # Filter dataframe based on start/stop time
    if start_time is not None:
        df = df[df["timestamps"] >= start_time]
    if stop_time is not None:
        df = df[df["timestamps"] <= stop_time]

    # Compute total duration for firing rate calculation
    if start_time is None:
        min_time = df["timestamps"].min()
    else:
        min_time = start_time

    if stop_time is None:
        max_time = df["timestamps"].max()
    else:
        max_time = stop_time

    duration = max_time - min_time  # Avoid division by zero

    if duration <= 0:
        raise ValueError("Invalid time window: Stop time must be greater than start time.")

    # Compute firing rate for each node

    # Compute spike counts per node
    spike_counts = df["node_ids"].value_counts().reset_index()
    spike_counts.columns = ["node_ids", "spike_count"]  # Rename columns

    # Merge with original dataframe to get corresponding labels (e.g., 'pop_name')
    spike_counts = spike_counts.merge(
        df[["node_ids"] + groupby].drop_duplicates(), on="node_ids", how="left"
    )

    # Compute firing rate
    spike_counts["firing_rate"] = spike_counts["spike_count"] / duration * 1000  # scale to Hz
    indivdual_stats = spike_counts

    # Compute mean and standard deviation per group
    pop_stats = spike_counts.groupby(groupby)["firing_rate"].agg(["mean", "std"]).reset_index()

    # Rename columns
    pop_stats.rename(columns={"mean": "firing_rate_mean", "std": "firing_rate_std"}, inplace=True)

    return pop_stats, indivdual_stats


def _pop_spike_rate(
    spike_times: Union[np.ndarray, list],
    time: Optional[Tuple[float, float, float]] = None,
    time_points: Optional[Union[np.ndarray, list]] = None,
    frequency: bool = False,
) -> np.ndarray:
    """
    Calculate the spike count or frequency histogram over specified time intervals.

    Parameters
    ----------
    spike_times : Union[np.ndarray, list]
        Array or list of spike times in milliseconds
    time : Optional[Tuple[float, float, float]], optional
        Tuple specifying (start, stop, step) in milliseconds. Used to create evenly spaced time points
        if `time_points` is not provided. Default is None.
    time_points : Optional[Union[np.ndarray, list]], optional
        Array or list of specific time points for binning. If provided, `time` is ignored. Default is None.
    frequency : bool, optional
        If True, returns spike frequency in Hz; otherwise, returns spike count. Default is False.

    Returns
    -------
    np.ndarray
        Array of spike counts or frequencies, depending on the `frequency` flag.

    Raises
    ------
    ValueError
        If both `time` and `time_points` are None.
    """
    if time_points is None:
        if time is None:
            raise ValueError("Either `time` or `time_points` must be provided.")
        time_points = np.arange(*time)
        dt = time[2]
    else:
        time_points = np.asarray(time_points).ravel()
        dt = (time_points[-1] - time_points[0]) / (time_points.size - 1)

    bins = np.append(time_points, time_points[-1] + dt)
    spike_rate, _ = np.histogram(np.asarray(spike_times), bins)

    if frequency:
        spike_rate = 1000 / dt * spike_rate

    return spike_rate


def get_population_spike_rate(
    spike_data: pd.DataFrame,
    fs: float = 400.0,
    t_start: float = 0,
    t_stop: Optional[float] = None,
    config: Optional[str] = None,
    network_name: Optional[str] = None,
    save: bool = False,
    save_path: Optional[str] = None,
    normalize: bool = False,
    smooth_window: int = 50,  # Window size for smoothing (in time bins)
    smooth_method: str = "gaussian",  # Smoothing method: 'gaussian', 'boxcar', or 'exponential'
) -> xr.DataArray:
    """
    Calculate the population spike rate for each population in the given spike data.

    Parameters
    ----------
    spike_data : pd.DataFrame
        A DataFrame containing spike data with columns 'pop_name', 'timestamps', and 'node_ids'
    fs : float, optional
        Sampling frequency in Hz, which determines the time bin size for calculating the spike rate (default: 400.0)
    t_start : float, optional
        Start time (in milliseconds) for spike rate calculation (default: 0)
    t_stop : Optional[float], optional
        Stop time (in milliseconds) for spike rate calculation. If None, defaults to the maximum timestamp in the data
    config : Optional[str], optional
        Path to a configuration file containing node information, used to determine the correct number of nodes per population.
        If None, node count is estimated from unique node spikes (default: None)
    network_name : Optional[str], optional
        Name of the network used in the configuration file, allowing selection of nodes for that network.
        Required if `config` is provided (default: None)
    save : bool, optional
        Whether to save the calculated population spike rate to a file (default: False)
    save_path : Optional[str], optional
        Directory path where the file should be saved if `save` is True (default: None)
    normalize : bool, optional
        Whether to normalize the spike rates for each population to a range of [0, 1] (default: False)
    smooth_window : int, optional
        Window size for smoothing in number of time bins (default: 50)
    smooth_method : str, optional
        Smoothing method to use: 'gaussian', 'boxcar', or 'exponential' (default: 'gaussian')

    Returns
    -------
    xr.DataArray
        An xarray DataArray containing the spike rates with dimensions of time, population, and type.
        The 'type' dimension includes 'raw' and 'smoothed' values.
        The DataArray includes sampling frequency (fs) as an attribute.
        If normalize is True, each population's spike rate is scaled to [0, 1].

    Raises
    ------
    ValueError
        If `save` is True but `save_path` is not provided.
        If an invalid smooth_method is specified.

    Notes
    -----
    - If `config` is None, the function assumes all cells in each population have fired at least once;
      otherwise, the node count may be inaccurate.
    - If normalization is enabled, each population's spike rate is scaled using Min-Max normalization.
    - Smoothing is applied using scipy.ndimage's filters based on the specified method.
    """
    import numpy as np
    from scipy import ndimage

    # Validate smoothing method
    if smooth_method not in ["gaussian", "boxcar", "exponential"]:
        raise ValueError(
            f"Invalid smooth_method: {smooth_method}. Choose from 'gaussian', 'boxcar', or 'exponential'."
        )

    pop_spikes = {}
    node_number = {}

    if config is None:
        pass
        # print(
        #     "Note: Node number is obtained by counting unique node spikes in the network.\nIf the network did not run for a sufficient duration, or not all cells fired,\nthen this count will not include all nodes so the firing rate will not be of the whole population!"
        # )
        # print(
        #     "You can provide a config to calculate the correct amount of nodes! for a true population rate."
        # )

    if config:
        if not network_name:
            print(
                "Grabbing first network; specify a network name to ensure correct node population is selected."
            )

    # Get t_stop if not provided
    if t_stop is None:
        t_stop = spike_data["timestamps"].max()

    # Get population names and prepare data
    populations = spike_data["pop_name"].unique()
    for pop_name in populations:
        ps = spike_data[spike_data["pop_name"] == pop_name]

        if config:
            nodes = load_nodes_from_config(config)
            if network_name:
                nodes = nodes[network_name]
            else:
                nodes = list(nodes.values())[0] if nodes else {}
            nodes = nodes[nodes["pop_name"] == pop_name]
            node_number[pop_name] = nodes.index.nunique()

        else:
            node_number[pop_name] = ps["node_ids"].nunique()

        filtered_spikes = spike_data[
            (spike_data["pop_name"] == pop_name)
            & (spike_data["timestamps"] > t_start)
            & (spike_data["timestamps"] < t_stop)
        ]
        pop_spikes[pop_name] = filtered_spikes

    # Calculate time points
    time = np.arange(t_start, t_stop, 1000 / fs)  # Convert sampling frequency to time steps

    # Calculate spike rates for each population
    spike_rates = []
    for p in populations:
        raw_rate = _pop_spike_rate(pop_spikes[p]["timestamps"], (t_start, t_stop, 1000 / fs))
        rate = fs / node_number[p] * raw_rate
        spike_rates.append(rate)

    spike_rates_array = np.array(spike_rates).T  # Transpose to have time as first dimension

    # Calculate smoothed version for each population
    smoothed_rates = []

    for i in range(spike_rates_array.shape[1]):
        pop_rate = spike_rates_array[:, i]

        if smooth_method == "gaussian":
            # Gaussian smoothing (sigma is approximately window/6 for a Gaussian filter)
            sigma = smooth_window / 6
            smoothed_pop_rate = ndimage.gaussian_filter1d(pop_rate, sigma=sigma)
        elif smooth_method == "boxcar":
            # Boxcar/uniform smoothing
            kernel = np.ones(smooth_window) / smooth_window
            smoothed_pop_rate = ndimage.convolve1d(pop_rate, kernel, mode="nearest")
        elif smooth_method == "exponential":
            # Exponential smoothing
            alpha = 2 / (smooth_window + 1)  # Equivalent to window size in exponential smoothing
            smoothed_pop_rate = np.zeros_like(pop_rate)
            smoothed_pop_rate[0] = pop_rate[0]
            for t in range(1, len(pop_rate)):
                smoothed_pop_rate[t] = alpha * pop_rate[t] + (1 - alpha) * smoothed_pop_rate[t - 1]

        smoothed_rates.append(smoothed_pop_rate)

    smoothed_rates_array = np.array(smoothed_rates).T  # Transpose to have time as first dimension

    # Stack raw and smoothed data
    combined_data = np.stack([spike_rates_array, smoothed_rates_array], axis=2)

    # Create DataArray with the additional 'type' dimension
    spike_rate_array = xr.DataArray(
        combined_data,
        coords={"time": time, "population": populations, "type": ["raw", "smoothed"]},
        dims=["time", "population", "type"],
        attrs={
            "fs": fs,
            "normalized": False,
            "smooth_method": smooth_method,
            "smooth_window": smooth_window,
        },
    )

    # Normalize if requested
    if normalize:
        # Apply normalization for each population and each type (raw/smoothed)
        for pop_idx in range(len(populations)):
            for type_idx, type_name in enumerate(["raw", "smoothed"]):
                pop_data = spike_rate_array.sel(population=populations[pop_idx], type=type_name)
                min_val = pop_data.min(dim="time")
                max_val = pop_data.max(dim="time")

                # Handle case where min == max (constant signal)
                if max_val != min_val:
                    spike_rate_array.loc[:, populations[pop_idx], type_name] = (
                        pop_data - min_val
                    ) / (max_val - min_val)

        spike_rate_array.attrs["normalized"] = True

    # Save if requested
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save is True.")

        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "spike_rate.h5")
        spike_rate_array.to_netcdf(save_file)

    return spike_rate_array


def average_spike_rate_over_windows(
    spike_rate: xr.DataArray, windows: List[Tuple[float, float]]
) -> xr.DataArray:
    """
    Calculate the average spike rate over multiple time windows.

    Parameters
    ----------
    spike_rate : xr.DataArray
        The spike rate data array with dimensions (time, population, type)
        where 'type' can be 'raw' or 'smoothed'
    windows : List[Tuple[float, float]]
        List of (start, end) times in milliseconds defining the windows to average over

    Returns
    -------
    xr.DataArray
        Averaged spike rate with time normalized to start at 0,
        preserving all original dimensions (time, population, type)
    """
    # Check if the DataArray has a 'type' dimension (compatible with new format)
    has_type_dim = "type" in spike_rate.dims

    # Initialize list to store data from each window
    window_data = []

    # Get data for each window
    for start, end in windows:
        # Select data points within the window
        window = spike_rate.sel(time=slice(start, end))

        # Normalize time to start at 0 for this window
        window = window.assign_coords(time=window.time - start)
        window_data.append(window)

    # Align and average windows
    # First window determines the time coordinates
    aligned_data = xr.concat(window_data, dim="window")
    averaged_data = aligned_data.mean(dim="window")

    # Create new DataArray with the averaged data
    if has_type_dim:
        # Create result with time, population, and type dimensions
        result = xr.DataArray(
            averaged_data.values,
            coords={
                "time": averaged_data.time.values,
                "population": averaged_data.population,
                "type": averaged_data.type,
            },
            dims=["time", "population", "type"],
        )
    else:
        # Handle older format without 'type' dimension (for backward compatibility)
        result = xr.DataArray(
            averaged_data.values,
            coords={"time": averaged_data.time.values, "population": averaged_data.population},
            dims=["time", "population"],
        )

    # Preserve attributes
    result.attrs = spike_rate.attrs

    return result


def compare_firing_over_times(
    spike_df: pd.DataFrame, group_by: str, time_window_1: List[float], time_window_2: List[float]
) -> None:
    """
    Compares the firing rates of a population during two different time windows and performs
    a statistical test to determine if there is a significant difference.

    Parameters
    ----------
    spike_df : pd.DataFrame
        DataFrame containing spike data with columns for timestamps, node_ids, and grouping variable
    group_by : str
        Column name to group spikes by (e.g., 'pop_name')
    time_window_1 : List[float]
        First time window as [start, stop] in milliseconds
    time_window_2 : List[float]
        Second time window as [start, stop] in milliseconds

    Returns
    -------
    None
        Results are printed to the console

    Notes
    -----
    Uses Mann-Whitney U test (non-parametric) to compare firing rates between the two windows
    """
    # Filter spikes for the population of interest
    for pop_name in spike_df[group_by].unique():
        print(f"Population: {pop_name}")
        pop_spikes = spike_df[spike_df[group_by] == pop_name]

        # Filter by time windows
        pop_spikes_1 = pop_spikes[
            (pop_spikes["timestamps"] >= time_window_1[0])
            & (pop_spikes["timestamps"] <= time_window_1[1])
        ]
        pop_spikes_2 = pop_spikes[
            (pop_spikes["timestamps"] >= time_window_2[0])
            & (pop_spikes["timestamps"] <= time_window_2[1])
        ]

        # Get unique neuron IDs
        unique_neurons = pop_spikes["node_ids"].unique()

        # Calculate firing rates per neuron for each time window in Hz
        neuron_rates_1 = []
        neuron_rates_2 = []

        for neuron in unique_neurons:
            # Count spikes for this neuron in each window
            n_spikes_1 = len(pop_spikes_1[pop_spikes_1["node_ids"] == neuron])
            n_spikes_2 = len(pop_spikes_2[pop_spikes_2["node_ids"] == neuron])

            # Calculate firing rate in Hz (convert ms to seconds by dividing by 1000)
            rate_1 = n_spikes_1 / ((time_window_1[1] - time_window_1[0]) / 1000)
            rate_2 = n_spikes_2 / ((time_window_2[1] - time_window_2[0]) / 1000)

            neuron_rates_1.append(rate_1)
            neuron_rates_2.append(rate_2)

        # Calculate average firing rates
        avg_firing_rate_1 = np.mean(neuron_rates_1) if neuron_rates_1 else 0
        avg_firing_rate_2 = np.mean(neuron_rates_2) if neuron_rates_2 else 0

        # Perform Mann-Whitney U test
        # Handle the case when one or both arrays are empty
        if len(neuron_rates_1) > 0 and len(neuron_rates_2) > 0:
            u_stat, p_val = mannwhitneyu(neuron_rates_1, neuron_rates_2, alternative="two-sided")
        else:
            u_stat, p_val = np.nan, np.nan

        print(f"    Average firing rate in window 1: {avg_firing_rate_1:.2f} Hz")
        print(f"    Average firing rate in window 2: {avg_firing_rate_2:.2f} Hz")
        print(f"    U-statistic: {u_stat:.2f}")
        print(f"    p-value: {p_val}")
        print(f"    Significant difference (p<0.05): {'Yes' if p_val < 0.05 else 'No'}")
    return


def find_bursting_cells(
    df: pd.DataFrame, isi_threshold: float = 10, burst_count_threshold: int = 1
) -> pd.DataFrame:
    """
    Finds bursting cells in a population based on a time difference threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spike data with columns for timestamps, node_ids, and pop_name
    isi_threshold : float, optional
        Time difference threshold in milliseconds to identify bursts
    burst_count_threshold : int, optional
        Number of bursts required to identify a bursting cell

    Returns
    -------
    pd.DataFrame
        DataFrame with bursting cells renamed in their pop_name column
    """
    # Create a new DataFrame with the time differences
    diff_df = df.copy()
    diff_df["time_diff"] = df.groupby("node_ids")["timestamps"].diff()

    # Create a column indicating whether each time difference is a burst
    diff_df["is_burst_instance"] = diff_df["time_diff"] < isi_threshold

    # Group by node_ids and check if any row has a burst instance
    # check if there are enough bursts
    burst_summary = diff_df.groupby("node_ids")["is_burst_instance"].sum() >= burst_count_threshold

    # Convert to a DataFrame with reset index
    burst_cells = burst_summary.reset_index(name="is_burst")

    # merge with original df to get timestamps
    burst_cells = pd.merge(burst_cells, df, on="node_ids")

    # Create a mask for burst cells that don't already have "_bursters" in their name
    burst_mask = burst_cells["is_burst"] & ~burst_cells["pop_name"].str.contains(
        "_bursters", na=False
    )

    # Add "_bursters" suffix only to those cells
    burst_cells.loc[burst_mask, "pop_name"] = burst_cells.loc[burst_mask, "pop_name"] + "_bursters"

    for pop in sorted(burst_cells["pop_name"].unique()):
        print(
            f"Number of cells in {pop}: {burst_cells[burst_cells['pop_name'] == pop]['node_ids'].nunique()}"
        )

    return burst_cells


def find_highest_firing_cells(
    df: pd.DataFrame, upper_quantile: float, groupby: str = "pop_name"
) -> pd.DataFrame:
    """
    Identifies and returns spikes from cells with firing rates above a specified upper quantile,
    grouped by a population label.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spike data with at least the following columns:
        - 'timestamps': Time of each spike event
        - 'node_ids': Identifier for each neuron
        - groupby (e.g., 'pop_name'): Population labels or grouping identifiers for neurons

    upper_quantile : float
        The upper quantile threshold (between 0 and 1).
        Cells with firing rates in the top (1 - upper_quantile) fraction are selected.
        For example, upper_quantile=0.8 selects the top 20% of high-firing cells.

    groupby : str, optional
        The column name used to group neurons by population. Default is 'pop_name'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the spikes from the high-firing cells across all groupbys.
    """
    if upper_quantile == 0:
        return df
    df_list = []
    for pop in df[groupby].unique():
        pop_df = df[df[groupby] == pop]
        _, pop_fr = compute_firing_rate_stats(pop_df, groupby=groupby)

        # Identify high firing cells
        threshold = pop_fr["firing_rate"].quantile(upper_quantile)
        high_firing_cells = pop_fr[pop_fr["firing_rate"] >= threshold]["node_ids"]

        # Filter spikes for high firing cells
        pop_spikes = pop_df[pop_df["node_ids"].isin(high_firing_cells)]
        df_list.append(pop_spikes)

    # Combine all high firing spikes into one DataFrame
    result_df = pd.concat(df_list, ignore_index=True)
    return result_df
