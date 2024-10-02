"""
Module for processing BMTK spikes output.
"""
# will add more to this at some point for now just putting this here to the file can exist

import h5py
import pandas as pd

def load_spikes_to_df(spike_file: str, network_name: str, sort: bool = True) -> pd.DataFrame:
    """
    Load spike data from an HDF5 file into a pandas DataFrame.

    Args:
        spike_file (str): Path to the HDF5 file containing spike data.
        network_name (str): The name of the network within the HDF5 file from which to load spike data.
        sort (bool, optional): Whether to sort the DataFrame by 'timestamps'. Defaults to True.

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
    return spikes_df
