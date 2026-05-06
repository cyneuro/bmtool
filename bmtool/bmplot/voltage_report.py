"""Plotting functions for voltage reports."""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes


def plot_voltage_traces(
    v_da: xr.DataArray,
    coordinate: str,
    values: Union[List, np.ndarray],
    tstart: Optional[float] = None,
    tstop: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot voltage traces from an xarray DataArray, filtered by a coordinate.

    Parameters
    ----------
    v_da : xr.DataArray
        A DataArray with dimensions ('time', 'node_id') and labeled coordinates.
        Typically returned from load_voltage_to_xarray().
    coordinate : str
        The coordinate name to filter by (e.g., 'node_id', 'pop_name').
    values : list or array-like
        The values of the coordinate to include in the plot.
    tstart : float, optional
        Start time for filtering; only time points >= tstart will be plotted.
    tstop : float, optional
        Stop time for filtering; only time points <= tstop will be plotted.
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot; if None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the voltage traces plotted.

    Notes
    -----
    - Each trace is labeled with its coordinate value in the legend.
    - If 'pop_name' is available as a coordinate, it will be included in the label for better readability.
    - Time filtering is applied after coordinate filtering.

    Examples
    --------
    Plot voltage traces for node_ids 0, 1, 2, 3:

    >>> plot_voltage_traces(v_da, 'node_id', [0, 1, 2, 3])

    Plot with time window filtering:

    >>> plot_voltage_traces(v_da, 'node_id', [0, 1], tstart=100, tstop=500)

    Plot multiple populations on the same axes:

    >>> fig, ax = plt.subplots()
    >>> plot_voltage_traces(v_da, 'pop_name', ['L23_PC'], ax=ax)
    >>> plot_voltage_traces(v_da, 'pop_name', ['L4_SS'], ax=ax)
    """
    # Initialize axes if none provided
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    # Filter by the specified coordinate
    try:
        filtered_da = v_da.sel({coordinate: values})
    except KeyError:
        raise ValueError(
            f"Coordinate '{coordinate}' not found in DataArray. "
            f"Available coordinates: {list(v_da.coords)}"
        )

    # Filter by time range if specified
    if tstart is not None or tstop is not None:
        time_slice = slice(tstart, tstop)
        filtered_da = filtered_da.sel(time=time_slice)

    # Extract time array (1D)
    time = filtered_da['time'].values

    # Iterate over the filtered node_ids and plot each trace
    for node_id in filtered_da.node_id.values:
        voltage = filtered_da.sel(node_id=node_id).values

        # Build label: include pop_name if available
        if 'pop_name' in filtered_da.coords:
            pop_name = filtered_da.sel(node_id=node_id)['pop_name'].values.item()
            label = f"node_id {node_id} ({pop_name})"
        else:
            label = f"node_id {node_id}"

        ax.plot(time, voltage, label=label)

    # Add labels and formatting
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax
