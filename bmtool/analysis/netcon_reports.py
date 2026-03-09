from typing import Any, Dict, List, Union

import h5py
import numpy as np
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
