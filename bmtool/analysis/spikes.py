"""
Module for processing BMTK spikes output.
"""

import h5py
import pandas as pd
from bmtool.util.util import load_nodes_from_config

def load_spikes_to_df(spike_file: str, network_name: str, sort: bool = True,config: str = None) -> pd.DataFrame:
    """
    Load spike data from an HDF5 file into a pandas DataFrame.

    Args:
        spike_file (str): Path to the HDF5 file containing spike data.
        network_name (str): The name of the network within the HDF5 file from which to load spike data.
        sort (bool, optional): Whether to sort the DataFrame by 'timestamps'. Defaults to True.
        config(str, optional): Will label the cell type of each spike 

    Returns:
        pd.DataFrame: A pandas DataFrame containing 'node_ids' and 'timestamps' columns from the spike data.
    
    Example:
        df = load_spikes_to_df("spikes.h5", "cortex")
    """
    with h5py.File(spike_file) as f:
        spikes_df = pd.DataFrame({
            'node_ids': f['spikes'][network_name]['node_ids'],
            'timestamps': f['spikes'][network_name]['timestamps']
        })
        if sort:
            spikes_df.sort_values(by='timestamps', inplace=True, ignore_index=True)
        if config:
            nodes = load_nodes_from_config(config)
            nodes = nodes[network_name]
            spikes_df = spikes_df.merge(nodes['pop_name'], left_on='node_ids', right_index=True, how='left')

    return spikes_df
