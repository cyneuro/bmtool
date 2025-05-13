import h5py
import numpy as np
import xarray as xr

from ..util.util import load_nodes_from_config


def load_synapse_report(h5_file_path, config_path, network):
    """
    Load and process a synapse report from a bmtk simulation into an xarray.

    Parameters:
    -----------
    h5_file_path : str
        Path to the h5 file containing the synapse report
    config_path : str
        Path to the simulation configuration file

    Returns:
    --------
    xarray.Dataset
        An xarray containing the synapse report data with proper population labeling
    """
    # Load the h5 file
    with h5py.File(h5_file_path, "r") as file:
        # Get the report data
        report = file["report"][network]
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

    # Load node information
    nodes = load_nodes_from_config(config_path)
    nodes = nodes[network]

    # Create a mapping from node IDs to population names
    node_to_pop = dict(zip(nodes.index, nodes["pop_name"]))

    # Get the number of synapses
    n_synapses = data.shape[1]

    # Create arrays to hold the source and target populations for each synapse
    source_pops = []
    target_pops = []
    connection_labels = []

    # Process each synapse
    for i in range(n_synapses):
        src_id = src_ids[i]
        trg_id = trg_ids[i]

        # Get population names (with fallback for unknown IDs)
        src_pop = node_to_pop.get(src_id, f"unknown_{src_id}")
        trg_pop = node_to_pop.get(trg_id, f"unknown_{trg_id}")

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
