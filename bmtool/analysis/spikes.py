"""
Module for processing BMTK spikes output.
"""

import h5py
import pandas as pd
from bmtool.util.util import load_nodes_from_config
from typing import Dict, Optional,Tuple, Union
import numpy as np
import os


def load_spikes_to_df(spike_file: str, network_name: str, sort: bool = True, config: str = None, groupby: str = 'pop_name') -> pd.DataFrame:
    """
    Load spike data from an HDF5 file into a pandas DataFrame.

    Args:
        spike_file (str): Path to the HDF5 file containing spike data.
        network_name (str): The name of the network within the HDF5 file from which to load spike data.
        sort (bool, optional): Whether to sort the DataFrame by 'timestamps'. Defaults to True.
        config (str, optional): Will label the cell type of each spike.
        groupby (str or list of str, optional): The column(s) to group by. Defaults to 'pop_name'.

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

            # Check if 'groupby' is a string or a list of strings and handle accordingly
            if isinstance(groupby, str):
                spikes_df = spikes_df.merge(nodes[groupby], left_on='node_ids', right_index=True, how='left')
            elif isinstance(groupby, list):
                for group in groupby:
                    spikes_df = spikes_df.merge(nodes[group], left_on='node_ids', right_index=True, how='left')

    return spikes_df



def _pop_spike_rate(spike_times: Union[np.ndarray, list], time: Optional[Tuple[float, float, float]] = None, 
                   time_points: Optional[Union[np.ndarray, list]] = None, frequeny: bool = False) -> np.ndarray:
    """
    Calculate the spike count or frequency histogram over specified time intervals.

    Args:
        spike_times (Union[np.ndarray, list]): Array or list of spike times in milliseconds.
        time (Optional[Tuple[float, float, float]], optional): Tuple specifying (start, stop, step) in milliseconds. 
            Used to create evenly spaced time points if `time_points` is not provided. Default is None.
        time_points (Optional[Union[np.ndarray, list]], optional): Array or list of specific time points for binning. 
            If provided, `time` is ignored. Default is None.
        frequeny (bool, optional): If True, returns spike frequency in Hz; otherwise, returns spike count. Default is False.

    Returns:
        np.ndarray: Array of spike counts or frequencies, depending on the `frequeny` flag.

    Raises:
        ValueError: If both `time` and `time_points` are None.
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
    
    if frequeny:
        spike_rate = 1000 / dt * spike_rate
    
    return spike_rate


def get_population_spike_rate(spikes: pd.DataFrame, fs: float = 400.0, t_start: float = 0, t_stop: Optional[float] = None, 
                              config: Optional[str] = None, network_name: Optional[str] = None,
                              save: bool = False, save_path: Optional[str] = None,
                              normalize: bool = False) -> Dict[str, np.ndarray]:
    """
    Calculate the population spike rate for each population in the given spike data, with an option to normalize.

    Args:
        spikes (pd.DataFrame): A DataFrame containing spike data with columns 'pop_name', 'timestamps', and 'node_ids'.
        fs (float, optional): Sampling frequency in Hz, which determines the time bin size for calculating the spike rate. Default is 400.
        t_start (float, optional): Start time (in milliseconds) for spike rate calculation. Default is 0.
        t_stop (Optional[float], optional): Stop time (in milliseconds) for spike rate calculation. If None, defaults to the maximum timestamp in the data.
        config (Optional[str], optional): Path to a configuration file containing node information, used to determine the correct number of nodes per population. 
            If None, node count is estimated from unique node spikes. Default is None.
        network_name (Optional[str], optional): Name of the network used in the configuration file, allowing selection of nodes for that network. 
            Required if `config` is provided. Default is None.
        save (bool, optional): Whether to save the calculated population spike rate to a file. Default is False.
        save_path (Optional[str], optional): Directory path where the file should be saved if `save` is True. If `save` is True and `save_path` is None, a ValueError is raised.
        normalize (bool, optional): Whether to normalize the spike rates for each population to a range of [0, 1]. Default is False.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are population names, and values are arrays representing the spike rate over time for each population. 
            If `normalize` is True, each population's spike rate is scaled to [0, 1].

    Raises:
        ValueError: If `save` is True but `save_path` is not provided.

    Notes:
        - If `config` is None, the function assumes all cells in each population have fired at least once; otherwise, the node count may be inaccurate.
        - If normalization is enabled, each population's spike rate is scaled using Min-Max normalization based on its own minimum and maximum values.

    """
    pop_spikes = {}
    node_number = {}

    if config is None:
        print("Note: Node number is obtained by counting unique node spikes in the network.\nIf the network did not run for a sufficient duration, and not all cells fired, this count might be incorrect.")
        print("You can provide a config to calculate the correct amount of nodes!")
        
    if config:
        if not network_name:
            print("Grabbing first network; specify a network name to ensure correct node population is selected.")

    for pop_name in spikes['pop_name'].unique():
        ps = spikes[spikes['pop_name'] == pop_name]
        
        if config:
            nodes = load_nodes_from_config(config)
            if network_name:
                nodes = nodes[network_name]
            else:
                nodes = list(nodes.values())[0] if nodes else {}
            nodes = nodes[nodes['pop_name'] == pop_name]
            node_number[pop_name] = nodes.index.nunique()
        else:
            node_number[pop_name] = ps['node_ids'].nunique()

        if t_stop is None:
            t_stop = spikes['timestamps'].max()

        filtered_spikes = spikes[
            (spikes['pop_name'] == pop_name) & 
            (spikes['timestamps'] > t_start) & 
            (spikes['timestamps'] < t_stop)
        ]
        pop_spikes[pop_name] = filtered_spikes

    time = np.array([t_start, t_stop, 1000 / fs])
    pop_rspk = {p: _pop_spike_rate(spk['timestamps'], time) for p, spk in pop_spikes.items()}
    spike_rate = {p: fs / node_number[p] * pop_rspk[p] for p in pop_rspk}

    # Normalize each spike rate series if normalize=True
    if normalize:
        spike_rate = {p: (sr - sr.min()) / (sr.max() - sr.min()) for p, sr in spike_rate.items()}

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided if save is True.")
        
        os.makedirs(save_path, exist_ok=True)
        
        save_file = os.path.join(save_path, 'spike_rate.h5')
        with h5py.File(save_file, 'w') as f:
            f.create_dataset('time', data=time)
            grp = f.create_group('populations')
            for p, rspk in spike_rate.items():
                pop_grp = grp.create_group(p)
                pop_grp.create_dataset('data', data=rspk)

    return spike_rate

