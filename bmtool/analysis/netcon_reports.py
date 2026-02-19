import h5py
import numpy as np
import xarray as xr

from ..util.util import load_nodes_from_config


def load_synapse_report(h5_file_path, config_path, edge_name):
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

    Returns:
    --------
    xarray.Dataset
        An xarray containing the synapse report data with proper population labeling
        including source_pop, target_pop, and connection_label coordinates
    """
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

    # Create mappings from node IDs to population names
    src_id_to_pop = dict(zip(source_nodes.index, source_nodes["pop_name"]))
    trg_id_to_pop = dict(zip(target_nodes.index, target_nodes["pop_name"]))

    # Get the number of synapses
    n_synapses = data.shape[1]

    # Create arrays to hold the source and target populations for each synapse
    source_pops = []
    target_pops = []
    connection_labels = []

    # Determine the source population for external inputs (src_id = -1)
    # Check if all nodes in source network have the same pop_name
    unique_src_pops = source_nodes["pop_name"].unique()
    if len(unique_src_pops) == 1:
        external_src_pop = unique_src_pops[0]
    else:
        external_src_pop = "unknown"

    # Process each synapse
    for i in range(n_synapses):
        src_id = src_ids[i]
        trg_id = trg_ids[i]

        # Get population names
        # For external inputs, src_id is typically -1
        if src_id == -1:
            src_pop = external_src_pop
        else:
            src_pop = src_id_to_pop.get(src_id, f"unknown_{src_id}")
        
        trg_pop = trg_id_to_pop.get(trg_id, f"unknown_{trg_id}")

        source_pops.append(src_pop)
        target_pops.append(trg_pop)
        connection_labels.append(f"{src_pop}->{trg_pop}")

    # Create xarray dataset
    ds = xr.Dataset(
        data_vars={"synapse_value": (["time", "synapse"], data)},
        coords={
            "time": time,
            "synapse": np.arange(n_synapses),
            "source_pop": ("synapse", source_pops),
            "target_pop": ("synapse", target_pops),
            "source_id": ("synapse", src_ids),
            "target_id": ("synapse", trg_ids),
            "sec_id": ("synapse", sec_id),
            "sec_x": ("synapse", sec_x),
            "connection_label": ("synapse", connection_labels),
        },
        attrs={"description": "Synapse report data from bmtk simulation"},
    )

    return ds
