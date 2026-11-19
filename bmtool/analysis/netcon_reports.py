from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from ..util.util import load_nodes_from_config


def load_synapse_report(
    h5_file_path: str,
    config_path: str,
    edge_name: str,
    source_groupby: Union[str, List[str]],
    target_groupby: Union[str, List[str]],
) -> xr.Dataset:
    """
    Load and process a synapse report from a bmtk simulation into an xarray.

    Parameters:
    -----------
    h5_file_path : str
        Path to the h5 file containing the synapse report
    config_path : str
        Path to the simulation configuration file
    edge_name : str
        Edge name in format 'source_to_target' (e.g., 'thalamic_tone_to_LA')
        This determines which source and target networks to load for population mapping
    source_groupby : str or List[str]
        Node property column name(s) to load as coordinates for filtering.
        Examples: 'pop_name', ['pop_name', 'model_type']
    target_groupby : str or List[str]
        Node property column name(s) to load as coordinates for filtering.
        Examples: 'pop_name', ['pop_name', 'model_type']

    Returns:
    --------
    xarray.Dataset
        An xarray containing the synapse report data with proper population labeling.
        For each column in source_groupby/target_groupby, separate coordinates are created:
        'source_{column}', 'target_{column}', etc.
        The 'connection_label' coordinate uses only pop_name for simple labeling
        (e.g., 'Pyr->PV'), while all groupby columns are available as coordinates for filtering.

    Examples:
    ---------
    # Group by single column (default behavior):
    ds = load_synapse_report(
        h5_file_path='output/synapse_report.h5',
        config_path='simulation_config.json',
        edge_name='LA_to_LA',
        source_groupby='pop_name',
        target_groupby='pop_name'
    )

    # Group by multiple columns for filtering (connection_label still uses only pop_name):
    ds = load_synapse_report(
        h5_file_path='output/synapse_report.h5',
        config_path='simulation_config.json',
        edge_name='LA_to_LA',
        source_groupby=['pop_name', 'model_type'],
        target_groupby=['pop_name', 'model_type']
    )
    # Returns dataset with coordinates:
    # source_pop_name, source_model_type, target_pop_name, target_model_type
    # connection_label (based on pop_name only)
    """
    # Normalize groupby parameters to lists
    if isinstance(source_groupby, str):
        source_groupby = [source_groupby]
    if isinstance(target_groupby, str):
        target_groupby = [target_groupby]

    # Parse edge_name to extract source and target networks
    if "_to_" not in edge_name:
        raise ValueError(
            f"Invalid edge_name format: '{edge_name}'. Expected format: 'source_to_target' "
            "(e.g., 'thalamic_tone_to_LA')"
        )
    
    source_network, target_network = edge_name.split("_to_")
    
    # Load the h5 file to get synapse mapping data
    with h5py.File(h5_file_path, "r") as file:
        # Get the first (and typically only) network key in the report
        report_networks = list(file["report"].keys())
        if not report_networks:
            raise ValueError(f"No report networks found in {h5_file_path}")
        
        # Use the first available network in the h5 file
        report_network = report_networks[0]
        report = file["report"][report_network]
        mapping = report["mapping"]

        # Get the data - shape is (n_timesteps, n_synapses)
        data = report["data"][:]

        # Get time information
        time_info = mapping["time"][:]  # [start_time, end_time, dt]
        start_time = time_info[0]
        end_time = time_info[1]
        dt = time_info[2]

        # Create time array
        n_steps = data.shape[0]
        time = np.linspace(start_time, start_time + (n_steps - 1) * dt, n_steps)

        # Get mapping information
        src_ids = mapping["src_ids"][:]
        trg_ids = mapping["trg_ids"][:]
        sec_id = mapping["element_ids"][:]
        sec_x = mapping["element_pos"][:]

    # Load node information for both source and target networks
    all_nodes = load_nodes_from_config(config_path)
    
    # Get the source and target node dataframes
    if source_network not in all_nodes:
        raise ValueError(
            f"Source network '{source_network}' not found in config. "
            f"Available networks: {list(all_nodes.keys())}"
        )
    if target_network not in all_nodes:
        raise ValueError(
            f"Target network '{target_network}' not found in config. "
            f"Available networks: {list(all_nodes.keys())}"
        )
    
    source_nodes = all_nodes[source_network]
    target_nodes = all_nodes[target_network]

    # Validate that requested groupby columns exist in node dataframes
    missing_src_cols = [col for col in source_groupby if col not in source_nodes.columns]
    if missing_src_cols:
        raise KeyError(
            f"Columns {missing_src_cols} not found in source network '{source_network}'. "
            f"Available columns: {list(source_nodes.columns)}"
        )
    
    missing_trg_cols = [col for col in target_groupby if col not in target_nodes.columns]
    if missing_trg_cols:
        raise KeyError(
            f"Columns {missing_trg_cols} not found in target network '{target_network}'. "
            f"Available columns: {list(target_nodes.columns)}"
        )

    # Create mappings from node IDs to groupby column values
    # source_mappings[col] = {node_id: column_value}
    source_mappings: Dict[str, Dict[int, Any]] = {}
    for col in source_groupby:
        source_mappings[col] = dict(zip(source_nodes.index, source_nodes[col]))
    
    target_mappings: Dict[str, Dict[int, Any]] = {}
    for col in target_groupby:
        target_mappings[col] = dict(zip(target_nodes.index, target_nodes[col]))

    # Determine default values for external inputs (src_id = -1)
    # Get the most common value or "unknown" if heterogeneous
    src_external_values = {}
    for col in source_groupby:
        unique_vals = source_nodes[col].unique()
        if len(unique_vals) == 1:
            src_external_values[col] = unique_vals[0]
        else:
            src_external_values[col] = "unknown"

    # Get the number of synapses
    n_synapses = data.shape[1]

    # Create arrays to hold the groupby values for each synapse
    # synapse_values[col] = [val_for_synapse_0, val_for_synapse_1, ...]
    source_values: Dict[str, List[Any]] = {col: [] for col in source_groupby}
    target_values: Dict[str, List[Any]] = {col: [] for col in target_groupby}
    connection_labels = []

    # Process each synapse
    for i in range(n_synapses):
        src_id = src_ids[i]
        trg_id = trg_ids[i]

        # Get source groupby values
        src_label_parts = []
        for col in source_groupby:
            if src_id == -1:
                # External input: use default value for this column
                val = src_external_values[col]
            else:
                val = source_mappings[col].get(src_id, f"unknown_{src_id}")
            source_values[col].append(val)
            src_label_parts.append(str(val))
        
        # Get target groupby values
        trg_label_parts = []
        for col in target_groupby:
            val = target_mappings[col].get(trg_id, f"unknown_{trg_id}")
            target_values[col].append(val)
            trg_label_parts.append(str(val))
        
        # Create connection label using only pop_name (not all groupby columns)
        if src_id == -1:
            src_pop = src_external_values.get('pop_name', 'unknown')
        else:
            src_pop = source_mappings['pop_name'].get(src_id, f"unknown_{src_id}")
        
        trg_pop = target_mappings['pop_name'].get(trg_id, f"unknown_{trg_id}")
        connection_labels.append(f"{src_pop}->{trg_pop}")

    # Create coordinates dictionary dynamically based on groupby columns
    coords = {
        "time": time,
        "synapse": np.arange(n_synapses),
        "source_id": ("synapse", src_ids),
        "target_id": ("synapse", trg_ids),
        "sec_id": ("synapse", sec_id),
        "sec_x": ("synapse", sec_x),
        "connection_label": ("synapse", connection_labels),
    }
    
    # Add source groupby coordinates
    for col in source_groupby:
        coords[f"source_{col}"] = ("synapse", source_values[col])
    
    # Add target groupby coordinates
    for col in target_groupby:
        coords[f"target_{col}"] = ("synapse", target_values[col])

    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={"synapse_value": (["time", "synapse"], data)},
        coords=coords,
        attrs={"description": "Synapse report data from bmtk simulation"},
    )

    return ds


# ----------------------------------------------------------------------------- array-level report access
class NetconReport(NamedTuple):
    """A BMTK netcon (synapse variable) report: ``data`` is the lazy h5 dataset (time x synapse)."""

    times: np.ndarray
    data: object
    target_ids: np.ndarray
    source_ids: np.ndarray


def read_netcon_report(h5_file_path, population: str) -> Optional[NetconReport]:
    """
    Open a BMTK netcon report (e.g. a weight or calcium variable) without loading the data.

    Returns ``NetconReport(times_ms, data, target_ids, source_ids)`` or None if the file does not exist. ``data`` is
    the h5 dataset (rows = time samples), so slices are read on demand.
    """
    from pathlib import Path

    if not Path(h5_file_path).exists():
        return None
    f = h5py.File(h5_file_path, "r")
    g = f["report"][population]
    m = g["mapping"]
    t0, _t1, dt = m["time"][()]
    data = g["data"]
    times = t0 + dt * np.arange(data.shape[0])
    return NetconReport(times, data, m["trg_ids"][()].astype(int), m["src_ids"][()].astype(int))


def sample_index(times, t):
    """
    Index of the first report sample at or after ``t`` (same units as ``times``), or the last sample if ``t`` is
    past the end. ``t`` may be a scalar or an array.
    """
    times = np.asarray(times)
    k = np.minimum(np.searchsorted(times, t), len(times) - 1)
    return int(k) if np.ndim(k) == 0 else k


def relative_to_initial(data, zero_initial: str = "one", dtype=np.float64) -> np.ndarray:
    """
    Report values divided by their first sample (e.g. W / W0), shape (time, synapse).

    ``zero_initial``: what to divide by where the first sample is 0 -- ``'one'`` (leave the column unscaled) or
    ``'nan'`` (the column becomes NaN).
    """
    a = np.asarray(data[:, :], dtype=dtype)
    a0 = a[0].copy()
    if zero_initial == "one":
        a0[a0 == 0] = 1
    elif zero_initial == "nan":
        a0 = np.where(a0 == 0, np.nan, a0)
    else:
        raise ValueError(f"zero_initial must be 'one' or 'nan', not {zero_initial!r}")
    return a / a0


def weight_change_by_group(
    report: NetconReport,
    labels: np.ndarray,
    t_marks: Dict[str, float],
    ltp_ratio: float = 1.1,
    ltd_ratio: float = 0.9,
) -> pd.DataFrame:
    """
    Relative weight change W(t)/W(0) - 1 per synapse group at named times (ms).

    The sample used for each mark is the first one at or after the mark (or the last sample). Synapses with
    W(0) = 0 are ignored. Also returns the fraction of synapses with W(t)/W(0) > ``ltp_ratio`` and < ``ltd_ratio``.

    Returns
    -------
    pd.DataFrame
        Columns: group, mark, time_ms, mean_change, frac_ltp, frac_ltd, n.
    """
    labels = np.asarray(labels)
    w0 = np.asarray(report.data[0, :], dtype=float)
    rows = []
    for name, t_ms in t_marks.items():
        k = sample_index(report.times, t_ms)
        rel = np.asarray(report.data[k, :], dtype=float) / np.where(w0 == 0, np.nan, w0)
        for lab in np.unique(labels):
            msk = labels == lab
            rows.append({"group": lab, "mark": name, "time_ms": float(report.times[k]),
                         "mean_change": float(np.nanmean(rel[msk]) - 1.0),
                         "frac_ltp": float(np.nanmean(rel[msk] > ltp_ratio)),
                         "frac_ltd": float(np.nanmean(rel[msk] < ltd_ratio)), "n": int(msk.sum())})
    return pd.DataFrame(rows)


def band_fractions(
    values: np.ndarray,
    times: np.ndarray,
    columns: np.ndarray,
    windows: Sequence[Tuple[float, float]],
    low: float,
    high: float,
) -> Optional[Dict[str, float]]:
    """
    Fractions of samples of the selected synapse ``columns`` that fall in ``[low, high)`` and at or above ``high``,
    and the 99th percentile divided by ``high``, over all samples inside ``windows`` (ms, half-open). Typical use:
    time spent by a calcium variable in the LTD and LTP zones of a calcium-threshold plasticity rule.

    ``values`` is a (time x synapse) array (already scaled to the units of ``low``/``high``). Returns None if no
    sample falls inside the windows.
    """
    msk = np.zeros(len(times), bool)
    for a, b in windows:
        msk |= (times >= a) & (times < b)
    if not msk.any():
        return None
    x = values[np.ix_(msk, np.asarray(columns))]
    return {"frac_between": float(np.mean((x >= low) & (x < high))), "frac_above": float(np.mean(x >= high)),
            "p99_over_high": float(np.percentile(x, 99) / high)}
