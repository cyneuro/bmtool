import os
import xarray as xr
import numpy as np
from ..util.util import CellVarsFile

def load_voltage_to_xarray(config=None, report_path=None, population=None, variable='v'):
    """
    Loads voltage traces into an xarray.DataArray with labeled dimensions.
    
    Parameters
    ----------
    config : str, optional
        Path to the Sonata config file. If provided, allows more automated lookup.
    report_path : str, optional
        Direct path to the v_traces.h5 file.
    population : str, optional
        The node population name.
    variable : str, default 'v'
        The variable name in the report.
        
    Returns
    -------
    xr.DataArray
        A DataArray with dimensions ('time', 'node_id')
    """
    if report_path is None:
        raise ValueError("report_path must be provided.")

    # Initialize the utility class from bmtool
    cv = CellVarsFile(report_path, population=population)

    # Check if the variable exists in the report
    if variable not in cv.variables:
        raise ValueError(f"Variable '{variable}' not found in the report. Available variables: {cv.variables}")

    # Extract gids
    gids = cv.gids
    if len(gids) == 0:
        raise ValueError("No gids found in the voltage report.")

    # Optimization: Read all data at once if it's a simple 1-compartment-per-cell report.
    # This is significantly faster than individual HDF5 reads.
    try:
        h5_dataset = cv._var_data[variable]
        if h5_dataset.shape[1] == len(gids):
            data = h5_dataset[:]
        else:
            # Fallback for multi-compartment or complex mapping
            data = np.column_stack([cv.data(gid, var_name=variable) for gid in gids])
    except Exception:
        # Final fallback
        data = np.column_stack([cv.data(gid, var_name=variable) for gid in gids])

    # Handle the time axis
    time_axis = cv.time_trace

    # Prepare coordinates dict
    coords = {
        "time": time_axis,
        "node_id": gids
    }

    # If config is provided, add pop_name and other metadata as coordinates
    if config is not None:
        from bmtool.util.util import load_nodes_from_config
        nodes = load_nodes_from_config(config)
        # If population is specified, use only that population's nodes
        if population is not None and population in nodes:
            node_df = nodes[population]
        else:
            # If not, try to find the population by matching gids
            # This fallback is not perfect but helps for single-population cases
            for pop, df in nodes.items():
                if set(gids).issubset(set(df.index)):
                    node_df = df
                    break
            else:
                node_df = None
        if node_df is not None:
            # Only keep rows for gids in this report
            node_df = node_df.loc[node_df.index.intersection(gids)]
            # Ensure order matches gids
            node_df = node_df.reindex(gids)
            # Add pop_name and any other columns as coordinates
            for col in node_df.columns:
                coords[col] = ("node_id", node_df[col].values)

    # Create the DataArray
    da = xr.DataArray(
        data,
        coords=coords,
        dims=("time", "node_id"),
        name=variable,
        attrs={"units": "mV"}
    )

    return da
