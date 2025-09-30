import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

# widgets
import ipywidgets as widgets
import matplotlib.pyplot as plt
import neuron
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from ipywidgets import HBox, VBox
from neuron import h
from neuron.units import ms, mV
from scipy.optimize import curve_fit, minimize, minimize_scalar

# scipy
from scipy.signal import find_peaks
from tqdm.notebook import tqdm

from bmtool.util.util import load_templates_from_config, load_nodes_from_config, load_edges_from_config, load_config

DEFAULT_GENERAL_SETTINGS = {
    "vclamp": False,
    "rise_interval": (0.1, 0.9),
    "tstart": 500.0,
    "tdur": 100.0,
    "threshold": -15.0,
    "delay": 1.3,
    "weight": 1.0,
    "dt": 0.025,
    "celsius": 20,
}

DEFAULT_GAP_JUNCTION_GENERAL_SETTINGS = {
    "tstart": 500.0,
    "tdur": 500.0,
    "dt": 0.025,
    "celsius": 20,
    "iclamp_amp": -0.01, # nA
}


class SynapseTuner:
    def __init__(
        self,
        conn_type_settings: Optional[Dict[str, dict]] = None,
        connection: Optional[str] = None,
        current_name: str = "i",
        mechanisms_dir: Optional[str] = None,
        templates_dir: Optional[str] = None,
        config: Optional[str] = None,
        general_settings: Optional[dict] = None,
        json_folder_path: Optional[str] = None,
        other_vars_to_record: Optional[List[str]] = None,
        slider_vars: Optional[List[str]] = None,
        hoc_cell: Optional[object] = None,
        network: Optional[str] = None,
    ) -> None:
        """
        Initialize the SynapseTuner class with connection type settings, mechanisms, and template directories.

        Parameters:
        -----------
        mechanisms_dir : Optional[str]
            Directory path containing the compiled mod files needed for NEURON mechanisms.
        templates_dir : Optional[str]
            Directory path containing cell template files (.hoc or .py) loaded into NEURON.
        conn_type_settings : Optional[dict]
            A dictionary containing connection-specific settings, such as synaptic properties and details.
        connection : Optional[str]
            Name of the connection type to be used from the conn_type_settings dictionary.
        general_settings : Optional[dict]
            General settings dictionary including parameters like simulation time step, duration, and temperature.
        json_folder_path : Optional[str]
            Path to folder containing JSON files with additional synaptic properties to update settings.
        current_name : str, optional
            Name of the synaptic current variable to be recorded (default is 'i').
        other_vars_to_record : Optional[list]
            List of additional synaptic variables to record during the simulation (e.g., 'Pr', 'Use').
        slider_vars : Optional[list]
            List of synaptic variables you would like sliders set up for the STP sliders method by default will use all parameters in spec_syn_param.
        hoc_cell : Optional[object]
            An already loaded NEURON cell object. If provided, template loading and cell setup will be skipped.
        network : Optional[str]
            Name of the specific network dataset to access from the loaded edges data (e.g., 'network_to_network').
            If not provided, will use all available networks. When a config file is provided, this enables
            the network dropdown feature in InteractiveTuner for switching between different networks.

        Network Dropdown Feature:
        -------------------------
        When initialized with a BMTK config file, the tuner automatically:
        1. Loads all available network datasets from the config
        2. Creates a network dropdown in InteractiveTuner (if multiple networks exist)
        3. Allows dynamic switching between networks, which rebuilds connection types
        4. Updates connection dropdown options when network is changed
        5. Preserves current connection if it exists in the new network, otherwise selects the first available
        """
        self.hoc_cell = hoc_cell
        # Store config and network information for network dropdown functionality
        self.config = config  # Store config path for network dropdown functionality
        self.available_networks = []  # Store available networks from config file
        self.current_network = network  # Store current network selection
        # Cache for loaded dynamics params JSON by filename to avoid repeated disk reads
        self._syn_params_cache = {}
        h.load_file('stdrun.hoc')

        if hoc_cell is None:
            if config is None and (mechanisms_dir is None or templates_dir is None):
                raise ValueError(
                    "Either a config file, both mechanisms_dir and templates_dir, or a hoc_cell must be provided."
                )

            if config is None:
                neuron.load_mechanisms(mechanisms_dir)
                h.load_file(templates_dir)
            else:
                # loads both mech and templates
                load_templates_from_config(config)
                # Load available networks from config for network dropdown feature
                self._load_available_networks()
                # Prebuild connection type settings for each available network to
                # make network switching in the UI fast. This will make __init__ slower
                # but dramatically speed up response when changing the network dropdown.
                self._prebuilt_conn_type_settings = {}
                try:
                    for net in self.available_networks:
                        self._prebuilt_conn_type_settings[net] = self._build_conn_type_settings_from_config(config, network=net)
                except Exception as e:
                    print(f"Warning: error prebuilding conn_type_settings for networks: {e}")

        if conn_type_settings is None:
            if config is not None:
                print("Building conn_type_settings from BMTK config files...")
                # If we prebuilt per-network settings, use the one for the requested network
                if hasattr(self, '_prebuilt_conn_type_settings') and network in getattr(self, '_prebuilt_conn_type_settings', {}):
                    conn_type_settings = self._prebuilt_conn_type_settings[network]
                else:
                    conn_type_settings = self._build_conn_type_settings_from_config(config, network=network)
                print(f"Found {len(conn_type_settings)} connection types: {list(conn_type_settings.keys())}")
                
                # If connection is not specified, use the first available connection
                if connection is None and conn_type_settings:
                    connection = list(conn_type_settings.keys())[0]
                    print(f"No connection specified, using first available: {connection}")
            else:
                raise ValueError("conn_type_settings must be provided if config is not specified.")
                
        if connection is None:
            raise ValueError("connection must be provided or inferable from conn_type_settings.")
        if connection not in conn_type_settings:
            raise ValueError(f"connection '{connection}' not found in conn_type_settings.")

        self.conn_type_settings: dict = conn_type_settings
        if json_folder_path:
            print(f"updating settings from json path {json_folder_path}")
            self._update_spec_syn_param(json_folder_path)
        # Use default general settings if not provided
        if general_settings is None:
            self.general_settings: dict = DEFAULT_GENERAL_SETTINGS.copy()
        else:
            # Merge defaults with user-provided
            self.general_settings = {**DEFAULT_GENERAL_SETTINGS, **general_settings}
        
        # Store the initial connection name and set up connection
        self.current_connection = connection
        self.conn = self.conn_type_settings[connection]
        self._current_cell_type = self.conn["spec_settings"]["post_cell"]
        self.synaptic_props = self.conn["spec_syn_param"]
        self.vclamp = self.general_settings["vclamp"]
        self.current_name = current_name
        self.other_vars_to_record = other_vars_to_record or []
        self.ispk = None
        self.input_mode = False  # Add input_mode attribute

        # Store original slider_vars for connection switching
        self.original_slider_vars = slider_vars or list(self.synaptic_props.keys())

        if slider_vars:
            # Start by filtering based on keys in slider_vars
            self.slider_vars = {
                key: value for key, value in self.synaptic_props.items() if key in slider_vars
            }
            # Iterate over slider_vars and check for missing keys in self.synaptic_props
            for key in slider_vars:
                # If the key is missing from synaptic_props, get the value using getattr
                if key not in self.synaptic_props:
                    try:
                        self._set_up_cell()
                        self._set_up_synapse()
                        value = getattr(self.syn, key)
                        self.slider_vars[key] = value
                    except AttributeError as e:
                        print(f"Error accessing '{key}' in syn {self.syn}: {e}")
        else:
            self.slider_vars = self.synaptic_props

        h.tstop = self.general_settings["tstart"] + self.general_settings["tdur"]
        h.dt = self.general_settings["dt"]  # Time step (resolution) of the simulation in ms
        h.steps_per_ms = 1 / h.dt
        h.celsius = self.general_settings["celsius"]

        # get some stuff set up we need for both SingleEvent and Interactive Tuner
        # Only set up cell if hoc_cell was not provided
        if self.hoc_cell is None:
            self._set_up_cell()
        else:
            self.cell = self.hoc_cell
        self._set_up_synapse()

        self.nstim = h.NetStim()
        self.nstim2 = h.NetStim()

        self.vcl = h.VClamp(self.cell.soma[0](0.5))

        self.nc = h.NetCon(
            self.nstim,
            self.syn,
            self.general_settings["threshold"],
            self.general_settings["delay"],
            self.general_settings["weight"],
        )
        self.nc2 = h.NetCon(
            self.nstim2,
            self.syn,
            self.general_settings["threshold"],
            self.general_settings["delay"],
            self.general_settings["weight"],
        )

        self._set_up_recorders()

    def _build_conn_type_settings_from_config(self, config_path: str, node_set: Optional[str] = None, network: Optional[str] = None) -> Dict[str, dict]:
        """
        Build conn_type_settings from BMTK simulation and circuit config files using the method used by relation matrix function in util.
        
        Parameters:
        -----------
        config_path : str
            Path to the simulation config JSON file.
        node_set : Optional[str]
            Specific node set to filter connections for. If None, processes all connections.
        network : Optional[str]
            Name of the specific network dataset to access (e.g., 'network_to_network').
            If None, processes all available networks.
            
        Returns:
        --------
        Dict[str, dict]
            Dictionary with connection names as keys and connection settings as values.

        NOTE: a lot of this code could probs be made a bit more simple or just removed i kinda tried a bunch of things and it works now
        but is kinda complex and some code is probs note needed 
            
        """
        # Load configuration and get nodes and edges using util.py methods
        config = load_config(config_path)
        # Ensure the config dict knows its source path so path substitutions can be resolved
        try:
            # load_config may return a dict; store path used so callers can resolve $COMPONENTS_DIR
            config['config_path'] = config_path
        except Exception:
            pass
        nodes = load_nodes_from_config(config_path)
        edges = load_edges_from_config(config_path)
        
        conn_type_settings = {}
        
        # If a specific network is requested, only process that one
        if network:
            if network not in edges:
                print(f"Warning: Network '{network}' not found in edges. Available networks: {list(edges.keys())}")
                return conn_type_settings
            edge_datasets = {network: edges[network]}
        else:
            edge_datasets = edges
        
        # Process each edge dataset using the util.py approach
        for edge_dataset_name, edge_df in edge_datasets.items():
            if edge_df.empty:
                continue
            
            # Create merged DataFrames with source and target node information like util.py does
            source_node_df = None
            target_node_df = None

            # First, try to deterministically parse the edge_dataset_name for patterns like '<src>_to_<tgt>'
            # e.g., 'network_to_network', 'extnet_to_network'
            if '_to_' in edge_dataset_name:
                parts = edge_dataset_name.split('_to_')
                if len(parts) == 2:
                    src_name, tgt_name = parts
                    if src_name in nodes:
                        source_node_df = nodes[src_name].add_prefix('source_')
                    if tgt_name in nodes:
                        target_node_df = nodes[tgt_name].add_prefix('target_')

            # If not found by parsing name, fall back to inspecting a sample edge row which contains
            # explicit 'source_population' and 'target_population' fields (this avoids reversing source/target)
            if source_node_df is None or target_node_df is None:
                sample_edge = edge_df.iloc[0] if len(edge_df) > 0 else None
                if sample_edge is not None:
                    # Use explicit population names from the edge entry
                    source_pop_name = sample_edge.get('source_population', '')
                    target_pop_name = sample_edge.get('target_population', '')
                    if source_pop_name in nodes:
                        source_node_df = nodes[source_pop_name].add_prefix('source_')
                    if target_pop_name in nodes:
                        target_node_df = nodes[target_pop_name].add_prefix('target_')

            # As a last resort, attempt to heuristically match by prefix/suffix of the dataset name
            if source_node_df is None or target_node_df is None:
                for pop_name, node_df in nodes.items():
                    if source_node_df is None and (edge_dataset_name.startswith(pop_name) or edge_dataset_name.endswith(pop_name)):
                        source_node_df = node_df.add_prefix('source_')
                    elif target_node_df is None and (edge_dataset_name.startswith(pop_name) or edge_dataset_name.endswith(pop_name)):
                        target_node_df = node_df.add_prefix('target_')
            
            # If we still don't have the node data, skip this edge dataset
            if source_node_df is None or target_node_df is None:
                print(f"Warning: Could not find node data for edge dataset {edge_dataset_name}")
                continue
            
            # Merge edge data with source node info
            edges_with_source = pd.merge(
                edge_df.reset_index(), 
                source_node_df, 
                how='left', 
                left_on='source_node_id', 
                right_index=True
            )
            
            # Merge with target node info
            edges_with_nodes = pd.merge(
                edges_with_source, 
                target_node_df, 
                how='left', 
                left_on='target_node_id', 
                right_index=True
            )
            
            # Get unique edge types from the merged dataset
            if 'edge_type_id' in edges_with_nodes.columns:
                edge_types = edges_with_nodes['edge_type_id'].unique()
            else:
                edge_types = [0]  # Single edge type
            
            # Process each edge type
            for edge_type_id in edge_types:
                # Filter edges for this type
                if 'edge_type_id' in edges_with_nodes.columns:
                    edge_type_data = edges_with_nodes[edges_with_nodes['edge_type_id'] == edge_type_id]
                else:
                    edge_type_data = edges_with_nodes
                
                if len(edge_type_data) == 0:
                    continue
                
                # Get representative edge for this type
                edge_info = edge_type_data.iloc[0]
                
                # Skip gap junctions
                if 'is_gap_junction' in edge_info and pd.notna(edge_info['is_gap_junction']) and edge_info['is_gap_junction']:
                    continue
                
                # Get population names from the merged data (this is the key improvement!)
                source_pop = edge_info.get('source_pop_name', '')
                target_pop = edge_info.get('target_pop_name', '')
                
                # Get target cell template from the merged data
                target_model_template = edge_info.get('target_model_template', '')
                if target_model_template.startswith('hoc:'):
                    target_cell_type = target_model_template.replace('hoc:', '')
                else:
                    target_cell_type = target_model_template
                
                # Create connection name using the actual population names
                if source_pop and target_pop:
                    conn_name = f"{source_pop}2{target_pop}"
                else:
                    conn_name = f"{edge_dataset_name}_type_{edge_type_id}"
                
                # Get synaptic model template
                model_template = edge_info.get('model_template', 'exp2syn')

                # Build connection settings early so we can attach metadata like dynamics file name
                conn_settings = {
                    'spec_settings': {
                        'post_cell': target_cell_type,
                        'vclamp_amp': -70.0,  # Default voltage clamp amplitude
                        'sec_x': 0.5,  # Default location on section
                        'sec_id': 0,   # Default to soma
                        # level_of_detail may be overridden by dynamics params below
                        'level_of_detail': model_template,
                    },
                    'spec_syn_param': {}
                }

                # Load synaptic parameters from dynamics_params file if available.
                # NOTE: the edge DataFrame produced by load_edges_from_config/load_sonata_edges_to_dataframe
                # already contains the 'dynamics_params' column (from the CSV) or the
                # flattened H5 dynamics_params attributes (prefixed with 'dynamics_params/').
                # Prefer the direct 'dynamics_params' column value from the merged DataFrame
                # rather than performing ad-hoc string parsing here.
                syn_params = {}
                dynamics_file_name = None
                # Prefer a top-level 'dynamics_params' column if present
                if 'dynamics_params' in edge_info and pd.notna(edge_info.get('dynamics_params')):
                    val = edge_info.get('dynamics_params')
                    # Some CSV loaders can produce bytes or numpy types; coerce to str
                    try:
                        dynamics_file_name = str(val).strip()
                    except Exception:
                        dynamics_file_name = None

                # If we found a dynamics file name, use it directly (skip token parsing)
                if dynamics_file_name and dynamics_file_name.upper() != 'NULL':
                    try:
                        conn_settings['spec_settings']['dynamics_params_file'] = dynamics_file_name
                        # use a cache to avoid re-reading the same JSON multiple times
                        if dynamics_file_name in self._syn_params_cache:
                            syn_params = self._syn_params_cache[dynamics_file_name]
                        else:
                            syn_params = self._load_synaptic_params_from_config(config, dynamics_file_name)
                            # cache result (even if empty dict) to avoid repeated lookups
                            self._syn_params_cache[dynamics_file_name] = syn_params
                    except Exception as e:
                        print(f"Warning: could not load dynamics_params file '{dynamics_file_name}' for edge {edge_dataset_name}: {e}")

                # If a dynamics params JSON filename was provided, prefer using its basename
                # as the connection name so that the UI matches the JSON definitions.
                if dynamics_file_name:
                    try:
                        json_base = os.path.splitext(os.path.basename(dynamics_file_name))[0]
                        # Ensure uniqueness in conn_type_settings
                        if json_base in conn_type_settings:
                            # Append edge_type_id to disambiguate
                            json_base = f"{json_base}_type_{edge_type_id}"
                        conn_name = json_base
                    except Exception:
                        pass

                # If the dynamics params defined a level_of_detail, override the default
                if isinstance(syn_params, dict) and 'level_of_detail' in syn_params:
                    conn_settings['spec_settings']['level_of_detail'] = syn_params.get('level_of_detail', model_template)

                # Add synaptic parameters, excluding level_of_detail
                for key, value in syn_params.items():
                    if key != 'level_of_detail':
                        conn_settings['spec_syn_param'][key] = value
                else:
                    # Fallback: some SONATA/H5 edge files expose dynamics params as flattened
                    # columns named like 'dynamics_params/<param>'. If no filename was given,
                    # gather any such columns from edge_info and use them as spec_syn_param.
                    for col in edge_info.index:
                        if isinstance(col, str) and col.startswith('dynamics_params/'):
                            param_key = col.split('/', 1)[1]
                            try:
                                val = edge_info[col]
                                if pd.notna(val):
                                    conn_settings['spec_syn_param'][param_key] = val
                            except Exception:
                                # Ignore malformed entries
                                pass
                
                # Add weight from edge info if available
                if 'syn_weight' in edge_info and pd.notna(edge_info['syn_weight']):
                    conn_settings['spec_syn_param']['initW'] = float(edge_info['syn_weight'])
                
                # Handle afferent section information
                if 'afferent_section_id' in edge_info and pd.notna(edge_info['afferent_section_id']):
                    conn_settings['spec_settings']['sec_id'] = int(edge_info['afferent_section_id'])
                
                if 'afferent_section_pos' in edge_info and pd.notna(edge_info['afferent_section_pos']):
                    conn_settings['spec_settings']['sec_x'] = float(edge_info['afferent_section_pos'])
                
                # Store in connection settings
                conn_type_settings[conn_name] = conn_settings

        return conn_type_settings
    
    def _load_available_networks(self) -> None:
        """
        Load available network names from the config file for the network dropdown feature.
        
        This method is automatically called during initialization when a config file is provided.
        It populates the available_networks list which enables the network dropdown in 
        InteractiveTuner when multiple networks are available.
        
        Network Dropdown Behavior:
        -------------------------
        - If only one network exists: No network dropdown is shown
        - If multiple networks exist: Network dropdown appears next to connection dropdown
        - Networks are loaded from the edges data in the config file
        - Current network defaults to the first available if not specified during init
        """
        if self.config is None:
            self.available_networks = []
            return
            
        try:
            edges = load_edges_from_config(self.config)
            self.available_networks = list(edges.keys())
            
            # Set current network to first available if not specified
            if self.current_network is None and self.available_networks:
                self.current_network = self.available_networks[0]
        except Exception as e:
            print(f"Warning: Could not load networks from config: {e}")
            self.available_networks = []
    
    def _load_synaptic_params_from_config(self, config: dict, dynamics_params: str) -> dict:
        """
        Load synaptic parameters from dynamics params file using config information.
        
        Parameters:
        -----------
        config : dict
            BMTK configuration dictionary
        dynamics_params : str
            Dynamics parameters filename
            
        Returns:
        --------
        dict
            Synaptic parameters dictionary
        """
        try:
            # Get the synaptic models directory from config
            synaptic_models_dir = config.get('components', {}).get('synaptic_models_dir', '')
            if synaptic_models_dir:
                # Handle path variables
                if synaptic_models_dir.startswith('$'):
                    # This is a placeholder, try to resolve it
                    config_dir = os.path.dirname(config.get('config_path', ''))
                    synaptic_models_dir = synaptic_models_dir.replace('$COMPONENTS_DIR', 
                                                                    os.path.join(config_dir, 'components'))
                    synaptic_models_dir = synaptic_models_dir.replace('$BASE_DIR', config_dir)
                
                dynamics_file = os.path.join(synaptic_models_dir, dynamics_params)
                
                if os.path.exists(dynamics_file):
                    with open(dynamics_file, 'r') as f:
                        return json.load(f)
                else:
                    print(f"Warning: Dynamics params file not found: {dynamics_file}")
        except Exception as e:
            print(f"Warning: Error loading synaptic parameters: {e}")
        
        return {}
    
    @classmethod
    def list_connections_from_config(cls, config_path: str, network: Optional[str] = None) -> Dict[str, dict]:
        """
        Class method to list all available connections from a BMTK config file without creating a tuner.
        
        Parameters:
        -----------
        config_path : str
            Path to the simulation config JSON file.
        network : Optional[str]
            Name of the specific network dataset to access (e.g., 'network_to_network').
            If None, processes all available networks.
            
        Returns:
        --------
        Dict[str, dict]
            Dictionary with connection names as keys and connection info as values.
        """
        # Create a temporary instance just to use the parsing methods
        temp_tuner = cls.__new__(cls)  # Create without calling __init__
        conn_type_settings = temp_tuner._build_conn_type_settings_from_config(config_path, network=network)
        
        # Create a summary of connections with key info
        connections_summary = {}
        for conn_name, settings in conn_type_settings.items():
            connections_summary[conn_name] = {
                'post_cell': settings['spec_settings']['post_cell'],
                'synapse_type': settings['spec_settings']['level_of_detail'],
                'parameters': list(settings['spec_syn_param'].keys())
            }
        
        return connections_summary

    def _switch_connection(self, new_connection: str) -> None:
        """
        Switch to a different connection type and update all related properties.
        
        Parameters:
        -----------
        new_connection : str
            Name of the new connection type to switch to.
        """
        if new_connection not in self.conn_type_settings:
            raise ValueError(f"Connection '{new_connection}' not found in conn_type_settings.")
        
        # Update current connection
        self.current_connection = new_connection
        self.conn = self.conn_type_settings[new_connection]
        self.synaptic_props = self.conn["spec_syn_param"]
        
        # Update slider vars for new connection
        if hasattr(self, 'original_slider_vars'):
            # Filter slider vars based on new connection's parameters
            self.slider_vars = {
                key: value for key, value in self.synaptic_props.items() 
                if key in self.original_slider_vars
            }
            
            # Check for missing keys and try to get them from the synapse
            for key in self.original_slider_vars:
                if key not in self.synaptic_props:
                    try:
                        # We'll get this after recreating the synapse
                        pass
                    except AttributeError as e:
                        print(f"Warning: Could not access '{key}' for connection '{new_connection}': {e}")
        else:
            self.slider_vars = self.synaptic_props
        
        # Need to recreate the cell if it's different
        if self.hoc_cell is None:
            # Check if we need a different cell type
            new_cell_type = self.conn["spec_settings"]["post_cell"]
            if not hasattr(self, '_current_cell_type') or self._current_cell_type != new_cell_type:
                self._current_cell_type = new_cell_type
                self._set_up_cell()
        
        # Recreate synapse for new connection
        self._set_up_synapse()
        
        # Update any missing slider vars from the new synapse
        if hasattr(self, 'original_slider_vars'):
            for key in self.original_slider_vars:
                if key not in self.synaptic_props:
                    try:
                        value = getattr(self.syn, key)
                        self.slider_vars[key] = value
                    except AttributeError as e:
                        print(f"Warning: Could not access '{key}' for connection '{new_connection}': {e}")
        
        # Recreate NetCon connections with new synapse
        self.nc = h.NetCon(
            self.nstim,
            self.syn,
            self.general_settings["threshold"],
            self.general_settings["delay"],
            self.general_settings["weight"],
        )
        self.nc2 = h.NetCon(
            self.nstim2,
            self.syn,
            self.general_settings["threshold"],
            self.general_settings["delay"],
            self.general_settings["weight"],
        )
        
        # Recreate voltage clamp with potentially new cell
        self.vcl = h.VClamp(self.cell.soma[0](0.5))
        
        # Recreate recorders for new synapse
        self._set_up_recorders()
        
        # Reset NEURON state
        h.finitialize()
        
        print(f"Successfully switched to connection: {new_connection}")

    def _switch_network(self, new_network: str) -> None:
        """
        Switch to a different network and rebuild conn_type_settings for the new network.
        
        This method is called when the user selects a different network from the network 
        dropdown in InteractiveTuner. It performs a complete rebuild of the connection 
        types available for the new network.
        
        Parameters:
        -----------
        new_network : str
            Name of the new network to switch to.
            
        Network Switching Process:
        -------------------------
        1. Validates the new network exists in available_networks
        2. Rebuilds conn_type_settings using the new network's edge data
        3. Updates the connection dropdown with new network's available connections
        4. Preserves current connection if it exists in new network
        5. Falls back to first available connection if current doesn't exist
        6. Recreates synapses and NEURON objects for the new connection
        7. Updates UI components to reflect the changes
        """
        if new_network not in self.available_networks:
            print(f"Warning: Network '{new_network}' not found in available networks: {self.available_networks}")
            return
        
        if new_network == self.current_network:
            return  # No change needed
        
        # Update current network
        self.current_network = new_network
        
        # Switch conn_type_settings using prebuilt data if available, otherwise build on-demand
        if self.config:
            print(f"Switching connections for network: {new_network}")
            if hasattr(self, '_prebuilt_conn_type_settings') and new_network in self._prebuilt_conn_type_settings:
                self.conn_type_settings = self._prebuilt_conn_type_settings[new_network]
            else:
                # Fallback: build on-demand (slower)
                self.conn_type_settings = self._build_conn_type_settings_from_config(self.config, network=new_network)
            
            # Update available connections and select first one if current doesn't exist
            available_connections = list(self.conn_type_settings.keys())
            if self.current_connection not in available_connections and available_connections:
                self.current_connection = available_connections[0]
                print(f"Connection '{self.current_connection}' not available in new network. Switched to: {available_connections[0]}")
            
            # Switch to the (potentially new) connection
            if self.current_connection in self.conn_type_settings:
                self._switch_connection(self.current_connection)
            
            print(f"Successfully switched to network: {new_network}")
            print(f"Available connections: {available_connections}")

    def _update_spec_syn_param(self, json_folder_path: str) -> None:
        """
        Update specific synaptic parameters using JSON files located in the specified folder.

        Parameters:
        -----------
        json_folder_path : str
            Path to folder containing JSON files, where each JSON file corresponds to a connection type.
        """
        if not self.conn_type_settings:
            return
        for conn_type, settings in self.conn_type_settings.items():
            json_file_path = os.path.join(json_folder_path, f"{conn_type}.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as json_file:
                    json_data = json.load(json_file)
                    settings["spec_syn_param"].update(json_data)
            else:
                print(f"JSON file for {conn_type} not found.")

    def _set_up_cell(self) -> None:
        """
        Set up the neuron cell based on the specified connection settings.
        This method is only called when hoc_cell is not provided.
        """
        if self.hoc_cell is None:
            self.cell = getattr(h, self.conn["spec_settings"]["post_cell"])()
        else:
            self.cell = self.hoc_cell

    def _set_up_synapse(self) -> None:
        """
        Set up the synapse on the target cell according to the synaptic parameters in `conn_type_settings`.

        Notes:
        ------
        - `_set_up_cell()` should be called before setting up the synapse.
        - Synapse location, type, and properties are specified within `spec_syn_param` and `spec_settings`.
        """
        try:
            self.syn = getattr(h, self.conn["spec_settings"]["level_of_detail"])(
                list(self.cell.all)[self.conn["spec_settings"]["sec_id"]](
                    self.conn["spec_settings"]["sec_x"]
                )
            )
        except:
            raise Exception("Make sure the mod file exist you are trying to load check spelling!")
        for key, value in self.conn["spec_syn_param"].items():
            if isinstance(value, (int, float)):
                if hasattr(self.syn, key):
                    setattr(self.syn, key, value)
                else:
                    print(
                        f"Warning: {key} cannot be assigned as it does not exist in the synapse. Check your mod file or spec_syn_param."
                    )

    def _set_up_recorders(self) -> None:
        """
        Set up recording vectors to capture simulation data.

        The method sets up recorders for:
        - Synaptic current specified by `current_name`
        - Other specified synaptic variables (`other_vars_to_record`)
        - Time, soma voltage, and voltage clamp current for all simulations.
        """
        self.rec_vectors = {}
        for var in self.other_vars_to_record:
            self.rec_vectors[var] = h.Vector()
            ref_attr = f"_ref_{var}"
            if hasattr(self.syn, ref_attr):
                self.rec_vectors[var].record(getattr(self.syn, ref_attr))
            else:
                print(
                    f"Warning: {ref_attr} not found in the syn object. Use vars() to inspect available attributes."
                )

        # Record synaptic current
        self.rec_vectors[self.current_name] = h.Vector()
        ref_attr = f"_ref_{self.current_name}"
        if hasattr(self.syn, ref_attr):
            self.rec_vectors[self.current_name].record(getattr(self.syn, ref_attr))
        else:
            print("Warning: Synaptic current recorder not set up correctly.")

        # Record time, synaptic events, soma voltage, and voltage clamp current
        self.t = h.Vector()
        self.tspk = h.Vector()
        self.soma_v = h.Vector()
        self.ivcl = h.Vector()

        self.t.record(h._ref_t)
        self.nc.record(self.tspk)
        self.nc2.record(self.tspk)
        self.soma_v.record(self.cell.soma[0](0.5)._ref_v)
        self.ivcl.record(self.vcl._ref_i)

    def SingleEvent(self, plot_and_print=True):
        """
        Simulate a single synaptic event by delivering an input stimulus to the synapse.

        The method sets up the neuron cell, synapse, stimulus, and voltage clamp,
        and then runs the NEURON simulation for a single event. The single synaptic event will occur at general_settings['tstart']
        Will display graphs and synaptic properies works best with a jupyter notebook
        """
        self.ispk = None

        # user slider values if the sliders are set up
        if hasattr(self, "dynamic_sliders"):
            syn_props = {var: slider.value for var, slider in self.dynamic_sliders.items()}
            self._set_syn_prop(**syn_props)

        # sets values based off optimizer
        if hasattr(self, "using_optimizer"):
            for name, value in zip(self.param_names, self.params):
                setattr(self.syn, name, value)

        # Set up the stimulus
        self.nstim.start = self.general_settings["tstart"]
        self.nstim.noise = 0
        self.nstim2.start = h.tstop
        self.nstim2.noise = 0

        # Set up voltage clamp
        vcldur = [[0, 0, 0], [self.general_settings["tstart"], h.tstop, 1e9]]
        for i in range(3):
            self.vcl.amp[i] = self.conn["spec_settings"]["vclamp_amp"]
            self.vcl.dur[i] = vcldur[1][i]

        # Run simulation
        h.tstop = self.general_settings["tstart"] + self.general_settings["tdur"]
        self.nstim.interval = self.general_settings["tdur"]
        self.nstim.number = 1
        self.nstim2.start = h.tstop
        h.run()

        current = np.array(self.rec_vectors[self.current_name])
        syn_props = self._get_syn_prop(
            rise_interval=self.general_settings["rise_interval"], dt=h.dt
        )
        current = (current - syn_props["baseline"]) * 1000  # Convert to pA
        current_integral = np.trapz(current, dx=h.dt)  # pA·ms

        if plot_and_print:
            self._plot_model(
                [
                    self.general_settings["tstart"] - 5,
                    self.general_settings["tstart"] + self.general_settings["tdur"],
                ]
            )
            for prop in syn_props.items():
                print(prop)
            print(f"Current Integral in pA*ms: {current_integral:.2f}")

        self.rise_time = syn_props["rise_time"]
        self.decay_time = syn_props["decay_time"]

    def _find_first(self, x):
        """
        Find the index of the first non-zero element in a given array.

        Parameters:
        -----------
        x : np.array
            The input array to search.

        Returns:
        --------
        int
            Index of the first non-zero element, or None if none exist.
        """
        x = np.asarray(x)
        idx = np.nonzero(x)[0]
        return idx[0] if idx.size else None

    def _get_syn_prop(self, rise_interval=(0.2, 0.8), dt=h.dt, short=False):
        """
        Calculate synaptic properties such as peak amplitude, latency, rise time, decay time, and half-width.

        Parameters:
        -----------
        rise_interval : tuple of floats, optional
            Fractional rise time interval to calculate (default is (0.2, 0.8)).
        dt : float, optional
            Time step of the simulation (default is NEURON's `h.dt`).
        short : bool, optional
            If True, only return baseline and sign without calculating full properties.

        Returns:
        --------
        dict
            A dictionary containing the synaptic properties: baseline, sign, peak amplitude, latency, rise time,
            decay time, and half-width.
        """
        if self.vclamp:
            isyn = self.ivcl
        else:
            isyn = self.rec_vectors[self.current_name]
        isyn = np.asarray(isyn)
        tspk = np.asarray(self.tspk)
        if tspk.size:
            tspk = tspk[0]

        ispk = int(np.floor(tspk / dt))
        baseline = isyn[ispk]
        isyn = isyn[ispk:] - baseline
        # print(np.abs(isyn))
        # print(np.argmax(np.abs(isyn)))
        # print(isyn[np.argmax(np.abs(isyn))])
        # print(np.sign(isyn[np.argmax(np.abs(isyn))]))
        sign = np.sign(isyn[np.argmax(np.abs(isyn))])
        if short:
            return {"baseline": baseline, "sign": sign}
        isyn *= sign
        # print(isyn)
        # peak amplitude
        ipk, _ = find_peaks(isyn)
        ipk = ipk[0]
        peak = isyn[ipk]
        # latency
        istart = self._find_first(np.diff(isyn[: ipk + 1]) > 0)
        latency = dt * (istart + 1)
        # rise time
        rt1 = self._find_first(isyn[istart : ipk + 1] > rise_interval[0] * peak)
        rt2 = self._find_first(isyn[istart : ipk + 1] > rise_interval[1] * peak)
        rise_time = (rt2 - rt1) * dt
        # decay time
        iend = self._find_first(np.diff(isyn[ipk:]) > 0)
        iend = isyn.size - 1 if iend is None else iend + ipk
        decay_len = iend - ipk + 1
        popt, _ = curve_fit(
            lambda t, a, tau: a * np.exp(-t / tau),
            dt * np.arange(decay_len),
            isyn[ipk : iend + 1],
            p0=(peak, dt * decay_len / 2),
        )
        decay_time = popt[1]
        # half-width
        hw1 = self._find_first(isyn[istart : ipk + 1] > 0.5 * peak)
        hw2 = self._find_first(isyn[ipk:] < 0.5 * peak)
        hw2 = isyn.size if hw2 is None else hw2 + ipk
        half_width = dt * (hw2 - hw1)
        output = {
            "baseline": baseline,
            "sign": sign,
            "latency": latency,
            "amp": peak,
            "rise_time": rise_time,
            "decay_time": decay_time,
            "half_width": half_width,
        }
        return output

    def _plot_model(self, xlim):
        """
        Plots the results of the simulation, including synaptic current, soma voltage,
        and any additional recorded variables.

        Parameters:
        -----------
        xlim : tuple
            A tuple specifying the limits of the x-axis for the plot (start_time, end_time).

        Notes:
        ------
        - The function determines how many plots to generate based on the number of variables recorded.
        - Synaptic current and either voltage clamp or soma voltage will always be plotted.
        - If other variables are provided in `other_vars_to_record`, they are also plotted.
        - The function adjusts the plot layout and removes any extra subplots that are not needed.
        """
        # Determine the number of plots to generate (at least 2: current and voltage)
        num_vars_to_plot = 2 + (len(self.other_vars_to_record) if self.other_vars_to_record else 0)

        # Set up figure based on number of plots (2x2 grid max)
        num_rows = (num_vars_to_plot + 1) // 2  # This ensures we have enough rows
        fig, axs = plt.subplots(num_rows, 2, figsize=(12, 7))
        axs = axs.ravel()

        # Plot synaptic current (always included)
        current = self.rec_vectors[self.current_name]
        syn_prop = self._get_syn_prop(short=True, dt=h.dt)
        current = current - syn_prop["baseline"]
        current = current * 1000

        axs[0].plot(self.t, current)
        if self.ispk is not None:
            for num in range(len(self.ispk)):
                axs[0].text(self.t[self.ispk[num]], current[self.ispk[num]], f"{str(num+1)}")

        axs[0].set_ylabel("Synaptic Current (pA)")

        # Plot voltage clamp or soma voltage (always included)
        ispk = int(np.round(self.tspk[0] / h.dt))
        if self.vclamp:
            baseline = self.ivcl[ispk]
            ivcl_plt = np.array(self.ivcl) - baseline
            ivcl_plt[:ispk] = 0
            axs[1].plot(self.t, 1000 * ivcl_plt)
            axs[1].set_ylabel("VClamp Current (pA)")
        else:
            soma_v_plt = np.array(self.soma_v)
            soma_v_plt[:ispk] = soma_v_plt[ispk]

            axs[1].plot(self.t, soma_v_plt)
            axs[1].set_ylabel("Soma Voltage (mV)")

        # Plot any other variables from other_vars_to_record, if provided
        if self.other_vars_to_record:
            for i, var in enumerate(self.other_vars_to_record, start=2):
                if var in self.rec_vectors:
                    axs[i].plot(self.t, self.rec_vectors[var])
                    axs[i].set_ylabel(f"{var.capitalize()}")

        # Adjust the layout
        for i, ax in enumerate(axs[:num_vars_to_plot]):
            ax.set_xlim(*xlim)
            if i >= num_vars_to_plot - 2:  # Add x-label to the last row
                ax.set_xlabel("Time (ms)")

        # Remove extra subplots if less than 4 plots are present
        if num_vars_to_plot < len(axs):
            for j in range(num_vars_to_plot, len(axs)):
                fig.delaxes(axs[j])

        # plt.tight_layout()
        plt.show()

    def _set_drive_train(self, freq=50.0, delay=250.0):
        """
        Configures trains of 12 action potentials at a specified frequency and delay period
        between pulses 8 and 9.

        Parameters:
        -----------
        freq : float, optional
            Frequency of the pulse train in Hz. Default is 50 Hz.
        delay : float, optional
            Delay period in milliseconds between the 8th and 9th pulses. Default is 250 ms.

        Returns:
        --------
        tstop : float
            The time at which the last pulse stops.

        Notes:
        ------
        - This function is based on experiments from the Allen Database.
        """
        # lets also set the train drive and delay here
        self.train_freq = freq
        self.train_delay = delay

        n_init_pulse = 8
        n_ending_pulse = 4
        self.nstim.start = self.general_settings["tstart"]
        self.nstim.interval = 1000 / freq
        self.nstim2.interval = 1000 / freq
        self.nstim.number = n_init_pulse
        self.nstim2.number = n_ending_pulse
        self.nstim2.start = self.nstim.start + (n_init_pulse - 1) * self.nstim.interval + delay
        tstop = self.nstim2.start + n_ending_pulse * self.nstim2.interval
        return tstop

    def _response_amplitude(self):
        """
        Calculates the amplitude of synaptic responses for each pulse in a train.

        Returns:
        --------
        amp : list
            A list containing the peak amplitudes for each pulse in the recorded synaptic current.

        Notes:
        ------
        This method:
        1. Extracts and normalizes the synaptic current
        2. Identifies spike times and segments the current accordingly
        3. Calculates the peak response amplitude for each segment
        4. Records the indices of peak amplitudes for visualization

        The amplitude values are returned in the original current units (before pA conversion).
        """
        isyn = np.array(self.rec_vectors[self.current_name].to_python())
        tspk = np.append(np.asarray(self.tspk), h.tstop)
        syn_prop = self._get_syn_prop(short=True, dt=h.dt)
        # print("syn_prp[sign] = " + str(syn_prop['sign']))
        isyn = isyn - syn_prop["baseline"]
        isyn *= syn_prop["sign"]
        ispk = np.floor((tspk + self.general_settings["delay"]) / h.dt).astype(int)

        try:
            amp = [isyn[ispk[i] : ispk[i + 1]].max() for i in range(ispk.size - 1)]
            # indexs of where the max of the synaptic current is at. This is then plotted
            self.ispk = [
                np.argmax(isyn[ispk[i] : ispk[i + 1]]) + ispk[i] for i in range(ispk.size - 1)
            ]
        # Sometimes the sim can cutoff at the peak of synaptic activity. So we just reduce the range by 1 and ingore that point
        except:
            amp = [isyn[ispk[i] : ispk[i + 1]].max() for i in range(ispk.size - 2)]
            self.ispk = [
                np.argmax(isyn[ispk[i] : ispk[i + 1]]) + ispk[i] for i in range(ispk.size - 2)
            ]

        return amp

    def _find_max_amp(self, amp):
        """
        Determines the maximum amplitude from the response data and returns the max in pA

        Parameters:
        -----------
        amp : array-like
            Array containing the amplitudes of synaptic responses.

        Returns:
        --------
        max_amp : float
            The maximum or minimum amplitude based on the sign of the response.
        """
        max_amp = max(amp)
        min_amp = min(amp)
        if abs(min_amp) > max_amp:
            return min_amp * 1000  # scale unit
        return max_amp * 1000  # scale unit

    def _calc_ppr_induction_recovery(self, amp, normalize_by_trial=True, print_math=True):
        """
        Calculates paired-pulse ratio, induction, and recovery metrics from response amplitudes.

        Parameters:
        -----------
        amp : array-like
            Array containing the amplitudes of synaptic responses to a pulse train.
        normalize_by_trial : bool, optional
            If True, normalize the amplitudes within each trial. Default is True.
        print_math : bool, optional
            If True, print detailed calculation steps and explanations. Default is True.

        Returns:
        --------
        tuple
            A tuple containing:
            - ppr: Paired-pulse ratio (2nd pulse / 1st pulse)
            - induction: Measure of facilitation/depression during initial pulses
            - recovery: Measure of recovery after the delay period

        Notes:
        ------
        - PPR > 1 indicates facilitation, PPR < 1 indicates depression
        - Induction > 0 indicates facilitation, Induction < 0 indicates depression
        - Recovery compares the response after delay to the initial pulses
        """
        amp = np.array(amp)
        amp = amp * 1000  # scale up
        amp = amp.reshape(-1, amp.shape[-1])
        
        # Calculate 90th percentile amplitude for normalization
        percentile_90 = np.percentile(amp, 90)

        def format_array(arr):
            """Format an array to 2 significant figures for cleaner output."""
            return np.array2string(arr, precision=2, separator=", ", suppress_small=True)

        if print_math:
            print("\n" + "=" * 40)
            print(
                f"Short Term Plasticity Results for {self.train_freq}Hz with {self.train_delay} Delay"
            )
            print("=" * 40)
            print("PPR: Above 0 is facilitating, below 0 is depressing.")
            print("Induction: Above 0 is facilitating, below 0 is depressing.")
            print("Recovery: A measure of how fast STP decays.\n")

            # PPR Calculation: (Avg 2nd pulse - Avg 1st pulse) / 90th percentile amplitude
            ppr = (np.mean(amp[:, 1:2]) - np.mean(amp[:, 0:1])) / percentile_90
            print("Paired Pulse Response (PPR)")
            print("Calculation: (Avg 2nd pulse - Avg 1st pulse) / 90th percentile amplitude")
            print(
                f"Values: ({np.mean(amp[:, 1:2]):.3f} - {np.mean(amp[:, 0:1]):.3f}) / {percentile_90:.3f} = {ppr:.3f}\n"
            )

            # Induction Calculation: (Avg (6th, 7th, 8th pulses) - Avg 1st pulse) / 90th percentile amplitude
            induction = (np.mean(amp[:, 5:8]) - np.mean(amp[:, :1])) / percentile_90
            print("Induction")
            print("Calculation: (Avg(6th, 7th, 8th pulses) - Avg 1st pulse) / 90th percentile amplitude")
            print(
                f"Values: {np.mean(amp[:, 5:8]):.3f} - {np.mean(amp[:, :1]):.3f} / {percentile_90:.3f} = {induction:.3f}\n"
            )

            # Recovery Calculation: (Avg (9th, 10th, 11th, 12th pulses) - Avg (1st, 2nd, 3rd, 4th pulses)) / 90th percentile amplitude
            recovery = (np.mean(amp[:, 8:12]) - np.mean(amp[:, :4])) / percentile_90
            print("Recovery")
            print(
                "Calculation: (Avg(9th, 10th, 11th, 12th pulses) - Avg(1st to 4th pulses)) / 90th percentile amplitude"
            )
            print(
                f"Values: {np.mean(amp[:, 8:12]):.3f} - {np.mean(amp[:, :4]):.3f} / {percentile_90:.3f} = {recovery:.3f}\n"
            )

            print("=" * 40 + "\n")

        # Calculate final metrics
        ppr = (np.mean(amp[:, 1:2]) - np.mean(amp[:, 0:1])) / percentile_90
        induction = (np.mean(amp[:, 5:8]) - np.mean(amp[:, :1])) / percentile_90
        recovery = (np.mean(amp[:, 8:12]) - np.mean(amp[:, :4])) / percentile_90

        return ppr, induction, recovery

    def _set_syn_prop(self, **kwargs):
        """
        Sets the synaptic parameters based on user inputs from sliders.

        Parameters:
        -----------
        **kwargs : dict
            Synaptic properties (such as weight, Use, tau_f, tau_d) as keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self.syn, key, value)

    def _simulate_model(self, input_frequency, delay, vclamp=None):
        """
        Runs the simulation with the specified input frequency, delay, and voltage clamp settings.

        Parameters:
        -----------
        input_frequency : float
            Frequency of the input drive train in Hz.
        delay : float
            Delay period in milliseconds between the 8th and 9th pulses.
        vclamp : bool or None, optional
            Whether to use voltage clamp. If None, the current setting is used. Default is None.

        Notes:
        ------
        This method handles two different input modes:
        - Standard train mode with 8 initial pulses followed by a delay and 4 additional pulses
        - Continuous input mode where stimulation continues for a specified duration
        """
        if not self.input_mode:
            self.tstop = self._set_drive_train(input_frequency, delay)
            h.tstop = self.tstop

            vcldur = [[0, 0, 0], [self.general_settings["tstart"], self.tstop, 1e9]]
            for i in range(3):
                self.vcl.amp[i] = self.conn["spec_settings"]["vclamp_amp"]
                self.vcl.dur[i] = vcldur[1][i]
            #h.finitialize(self.cell.Vinit * mV)
            #h.continuerun(self.tstop * ms)
            h.run()
        else:
            self.tstop = self.general_settings["tstart"] + self.general_settings["tdur"]
            self.nstim.interval = 1000 / input_frequency
            self.nstim.number = np.ceil(self.w_duration.value / 1000 * input_frequency + 1)
            self.nstim2.number = 0
            self.tstop = self.w_duration.value + self.general_settings["tstart"]

            #h.finitialize(self.cell.Vinit * mV)
            #h.continuerun(self.tstop * ms)
            h.run()

    def InteractiveTuner(self):
        """
        Sets up interactive sliders for tuning short-term plasticity (STP) parameters in a Jupyter Notebook.

        This method creates an interactive UI with sliders for:
        - Network selection dropdown (if multiple networks available and config provided)
        - Connection type selection dropdown
        - Input frequency
        - Delay between pulse trains
        - Duration of stimulation (for continuous input mode)
        - Synaptic parameters (e.g., Use, tau_f, tau_d) based on the syn model

        It also provides buttons for:
        - Running a single event simulation
        - Running a train input simulation
        - Toggling voltage clamp mode
        - Switching between standard and continuous input modes

        Network Dropdown Feature:
        ------------------------
        When the SynapseTuner is initialized with a BMTK config file containing multiple networks:
        - A network dropdown appears next to the connection dropdown
        - Users can dynamically switch between networks (e.g., 'network_to_network', 'external_to_network')
        - Switching networks rebuilds available connections and updates the connection dropdown
        - The current connection is preserved if it exists in the new network
        - If multiple networks exist but only one is specified during init, that network is used as default

        Notes:
        ------
        Ideal for exploratory parameter tuning and interactive visualization of
        synapse behavior with different parameter values and stimulation protocols.
        The network dropdown feature enables comprehensive exploration of multi-network
        BMTK simulations without needing to reinitialize the tuner.
        """
        # Widgets setup (Sliders)
        freqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 35, 50, 100, 200]
        delays = [125, 250, 500, 1000, 2000, 4000]
        durations = [100, 300, 500, 1000, 2000, 5000, 10000]
        freq0 = 50
        delay0 = 250
        duration0 = 300
        vlamp_status = self.vclamp

        # Connection dropdown
        connection_options = sorted(list(self.conn_type_settings.keys()))
        w_connection = widgets.Dropdown(
            options=connection_options,
            value=self.current_connection,
            description="Connection:",
            style={'description_width': 'initial'}
        )

        # Network dropdown - only shown if config was provided and multiple networks are available
        # This enables users to switch between different network datasets dynamically
        w_network = None
        if self.config is not None and len(self.available_networks) > 1:
            w_network = widgets.Dropdown(
                options=self.available_networks,
                value=self.current_network,
                description="Network:",
                style={'description_width': 'initial'}
            )

        w_run = widgets.Button(description="Run Train", icon="history", button_style="primary")
        w_single = widgets.Button(description="Single Event", icon="check", button_style="success")
        w_vclamp = widgets.ToggleButton(
            value=vlamp_status,
            description="Voltage Clamp",
            icon="fast-backward",
            button_style="warning",
        )
        
        # Voltage clamp amplitude input
        default_vclamp_amp = getattr(self.conn['spec_settings'], 'vclamp_amp', -70.0)
        w_vclamp_amp = widgets.FloatText(
            value=default_vclamp_amp,
            description="V_clamp (mV):",
            step=5.0,
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='150px')
        )
        
        w_input_mode = widgets.ToggleButton(
            value=False, description="Continuous input", icon="eject", button_style="info"
        )
        w_input_freq = widgets.SelectionSlider(options=freqs, value=freq0, description="Input Freq")

        # Sliders for delay and duration
        self.w_delay = widgets.SelectionSlider(options=delays, value=delay0, description="Delay")
        self.w_duration = widgets.SelectionSlider(
            options=durations, value=duration0, description="Duration"
        )

        def create_dynamic_sliders():
            """Create sliders based on current connection's parameters"""
            sliders = {}
            for key, value in self.slider_vars.items():
                if isinstance(value, (int, float)):  # Only create sliders for numeric values
                    if hasattr(self.syn, key):
                        if value == 0:
                            print(
                                f"{key} was set to zero, going to try to set a range of values, try settings the {key} to a nonzero value if you dont like the range!"
                            )
                            slider = widgets.FloatSlider(
                                value=value, min=0, max=1000, step=1, description=key
                            )
                        else:
                            slider = widgets.FloatSlider(
                                value=value, min=0, max=value * 20, step=value / 5, description=key
                            )
                        sliders[key] = slider
                    else:
                        print(f"skipping slider for {key} due to not being a synaptic variable")
            return sliders

        # Generate sliders dynamically based on valid numeric entries in self.slider_vars
        self.dynamic_sliders = create_dynamic_sliders()
        print(
            "Setting up slider! The sliders ranges are set by their init value so try changing that if you dont like the slider range!"
        )

        # Create output widget for displaying results
        output_widget = widgets.Output()
        
        def run_single_event(*args):
            clear_output()
            display(ui)
            display(output_widget)
            
            self.vclamp = w_vclamp.value
            # Update voltage clamp amplitude if voltage clamp is enabled
            if self.vclamp:
                # Update the voltage clamp amplitude settings
                self.conn['spec_settings']['vclamp_amp'] = w_vclamp_amp.value
                # Update general settings if they exist
                if hasattr(self, 'general_settings'):
                    self.general_settings['vclamp_amp'] = w_vclamp_amp.value
            # Update synaptic properties based on slider values
            self.ispk = None
            
            # Clear previous results and run simulation
            output_widget.clear_output()
            with output_widget:
                self.SingleEvent()

        def on_connection_change(*args):
            """Handle connection dropdown change"""
            try:
                new_connection = w_connection.value
                if new_connection != self.current_connection:
                    # Switch to new connection
                    self._switch_connection(new_connection)
                    
                    # Recreate dynamic sliders for new connection
                    self.dynamic_sliders = create_dynamic_sliders()
                    
                    # Update UI
                    update_ui_layout()
                    update_ui()
                    
            except Exception as e:
                print(f"Error switching connection: {e}")

        def on_network_change(*args):
            """
            Handle network dropdown change events.
            
            This callback is triggered when the user selects a different network from 
            the network dropdown. It coordinates the complete switching process:
            1. Calls _switch_network() to rebuild connections for the new network
            2. Updates the connection dropdown options with new network's connections
            3. Recreates dynamic sliders for the new connection parameters
            4. Refreshes the entire UI to reflect all changes
            """
            if w_network is None:
                return
            try:
                new_network = w_network.value
                if new_network != self.current_network:
                    # Switch to new network
                    self._switch_network(new_network)
                    
                    # Update connection dropdown options with new network's connections
                    connection_options = list(self.conn_type_settings.keys())
                    w_connection.options = connection_options
                    if connection_options:
                        w_connection.value = self.current_connection
                    
                    # Recreate dynamic sliders for new connection
                    self.dynamic_sliders = create_dynamic_sliders()
                    
                    # Update UI
                    update_ui_layout()
                    update_ui()
                    
            except Exception as e:
                print(f"Error switching network: {e}")

        def update_ui_layout():
            """
            Update the UI layout with new sliders and network dropdown.
            
            This function reconstructs the entire UI layout including:
            - Network dropdown (if available) and connection dropdown in the top row
            - Button controls and input mode toggles
            - Parameter sliders arranged in columns
            """
            nonlocal ui, slider_columns
            
            # Add the dynamic sliders to the UI
            slider_widgets = [slider for slider in self.dynamic_sliders.values()]
            
            if slider_widgets:
                half = len(slider_widgets) // 2
                col1 = VBox(slider_widgets[:half])
                col2 = VBox(slider_widgets[half:])
                slider_columns = HBox([col1, col2])
            else:
                slider_columns = VBox([])
            
            # Create button row with voltage clamp controls
            if w_vclamp.value:  # Show voltage clamp amplitude input when toggle is on
                button_row = HBox([w_run, w_single, w_vclamp, w_vclamp_amp, w_input_mode])
            else:  # Hide voltage clamp amplitude input when toggle is off
                button_row = HBox([w_run, w_single, w_vclamp, w_input_mode])
            
            # Construct the top row - include network dropdown if available
            # This creates a horizontal layout with network dropdown (if present) and connection dropdown
            if w_network is not None:
                connection_row = HBox([w_network, w_connection])
            else:
                connection_row = HBox([w_connection])
            slider_row = HBox([w_input_freq, self.w_delay, self.w_duration])
            
            ui = VBox([connection_row, button_row, slider_row, slider_columns])

        # Function to update UI based on input mode
        def update_ui(*args):
            clear_output()
            display(ui)
            display(output_widget)
            
            self.vclamp = w_vclamp.value
            # Update voltage clamp amplitude if voltage clamp is enabled
            if self.vclamp:
                self.conn['spec_settings']['vclamp_amp'] = w_vclamp_amp.value
                if hasattr(self, 'general_settings'):
                    self.general_settings['vclamp_amp'] = w_vclamp_amp.value
            
            self.input_mode = w_input_mode.value
            syn_props = {var: slider.value for var, slider in self.dynamic_sliders.items()}
            self._set_syn_prop(**syn_props)
            
            # Clear previous results and run simulation
            output_widget.clear_output()
            with output_widget:
                if not self.input_mode:
                    self._simulate_model(w_input_freq.value, self.w_delay.value, w_vclamp.value)
                else:
                    self._simulate_model(w_input_freq.value, self.w_duration.value, w_vclamp.value)
                amp = self._response_amplitude()
                self._plot_model(
                    [self.general_settings["tstart"] - self.nstim.interval / 3, self.tstop]
                )
                _ = self._calc_ppr_induction_recovery(amp)

        # Function to switch between delay and duration sliders
        def switch_slider(*args):
            if w_input_mode.value:
                self.w_delay.layout.display = "none"  # Hide delay slider
                self.w_duration.layout.display = ""  # Show duration slider
            else:
                self.w_delay.layout.display = ""  # Show delay slider
                self.w_duration.layout.display = "none"  # Hide duration slider

        # Function to handle voltage clamp toggle
        def on_vclamp_toggle(*args):
            """Handle voltage clamp toggle changes to show/hide amplitude input"""
            update_ui_layout()
            clear_output()
            display(ui)
            display(output_widget)

        # Link widgets to their callback functions
        w_connection.observe(on_connection_change, names="value")
        # Link network dropdown callback only if network dropdown was created
        if w_network is not None:
            w_network.observe(on_network_change, names="value")
        w_input_mode.observe(switch_slider, names="value")
        w_vclamp.observe(on_vclamp_toggle, names="value")

        # Hide the duration slider initially until the user selects it
        self.w_duration.layout.display = "none"  # Hide duration slider

        w_single.on_click(run_single_event)
        w_run.on_click(update_ui)

        # Initial UI setup
        slider_columns = VBox([])
        ui = VBox([])
        update_ui_layout()

        display(ui)
        update_ui()

    def stp_frequency_response(
        self,
        freqs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 35, 50, 100, 200],
        delay=250,
        plot=True,
        log_plot=True,
    ):
        """
        Analyze synaptic response across different stimulation frequencies.

        This method systematically tests how the synapse model responds to different
        stimulation frequencies, calculating key short-term plasticity (STP) metrics
        for each frequency.

        Parameters:
        -----------
        freqs : list, optional
            List of frequencies to analyze (in Hz). Default covers a wide range from 1-200 Hz.
        delay : float, optional
            Delay between pulse trains in ms. Default is 250 ms.
        plot : bool, optional
            Whether to plot the results. Default is True.
        log_plot : bool, optional
            Whether to use logarithmic scale for frequency axis. Default is True.

        Returns:
        --------
        dict
            Dictionary containing frequency-dependent metrics with keys:
            - 'frequencies': List of tested frequencies
            - 'ppr': Paired-pulse ratios at each frequency
            - 'induction': Induction values at each frequency
            - 'recovery': Recovery values at each frequency

        Notes:
        ------
        This method is particularly useful for characterizing the frequency-dependent
        behavior of synapses, such as identifying facilitating vs. depressing regimes
        or the frequency at which a synapse transitions between these behaviors.
        """
        results = {"frequencies": freqs, "ppr": [], "induction": [], "recovery": []}

        # Store original state
        original_ispk = self.ispk

        for freq in tqdm(freqs, desc="Analyzing frequencies"):
            self._simulate_model(freq, delay)
            amp = self._response_amplitude()
            ppr, induction, recovery = self._calc_ppr_induction_recovery(amp, print_math=False)

            results["ppr"].append(float(ppr))
            results["induction"].append(float(induction))
            results["recovery"].append(float(recovery))

        # Restore original state
        self.ispk = original_ispk

        if plot:
            self._plot_frequency_analysis(results, log_plot=log_plot)

        return results

    def _plot_frequency_analysis(self, results, log_plot):
        """
        Plot the frequency-dependent synaptic properties.

        Parameters:
        -----------
        results : dict
            Dictionary containing frequency analysis results with keys:
            - 'frequencies': List of tested frequencies
            - 'ppr': Paired-pulse ratios at each frequency
            - 'induction': Induction values at each frequency
            - 'recovery': Recovery values at each frequency
        log_plot : bool
            Whether to use logarithmic scale for frequency axis

        Notes:
        ------
        Creates a figure with three subplots showing:
        1. Paired-pulse ratio vs. frequency
        2. Induction vs. frequency
        3. Recovery vs. frequency

        Each plot includes a horizontal reference line at y=0 or y=1 to indicate
        the boundary between facilitation and depression.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot PPR
        if log_plot:
            ax1.semilogx(results["frequencies"], results["ppr"], "o-")
        else:
            ax1.plot(results["frequencies"], results["ppr"], "o-")
        ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Paired Pulse Ratio")
        ax1.set_title("PPR vs Frequency")
        ax1.grid(True)

        # Plot Induction
        if log_plot:
            ax2.semilogx(results["frequencies"], results["induction"], "o-")
        else:
            ax2.plot(results["frequencies"], results["induction"], "o-")
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Induction")
        ax2.set_title("Induction vs Frequency")
        ax2.grid(True)

        # Plot Recovery
        if log_plot:
            ax3.semilogx(results["frequencies"], results["recovery"], "o-")
        else:
            ax3.plot(results["frequencies"], results["recovery"], "o-")
        ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Recovery")
        ax3.set_title("Recovery vs Frequency")
        ax3.grid(True)

        plt.tight_layout()
        plt.show()


class GapJunctionTuner:
    def __init__(
        self,
        mechanisms_dir: Optional[str] = None,
        templates_dir: Optional[str] = None,
        config: Optional[str] = None,
        general_settings: Optional[dict] = None,
        conn_type_settings: Optional[dict] = None,
        hoc_cell: Optional[object] = None,
    ):
        """
        Initialize the GapJunctionTuner class.

        Parameters:
        -----------
        mechanisms_dir : str
            Directory path containing the compiled mod files needed for NEURON mechanisms.
        templates_dir : str
            Directory path containing cell template files (.hoc or .py) loaded into NEURON.
        config : str
            Path to a BMTK config.json file. Can be used to load mechanisms, templates, and other settings.
        general_settings : dict
            General settings dictionary including parameters like simulation time step, duration, and temperature.
        conn_type_settings : dict
            A dictionary containing connection-specific settings for gap junctions.
        hoc_cell : object, optional
            An already loaded NEURON cell object. If provided, template loading and cell creation will be skipped.
        """
        self.hoc_cell = hoc_cell

        if hoc_cell is None:
            if config is None and (mechanisms_dir is None or templates_dir is None):
                raise ValueError(
                    "Either a config file, both mechanisms_dir and templates_dir, or a hoc_cell must be provided."
                )

            if config is None:
                neuron.load_mechanisms(mechanisms_dir)
                h.load_file(templates_dir)
            else:
                # this will load both mechs and templates
                load_templates_from_config(config)

        # Use default general settings if not provided, merge with user-provided
        if general_settings is None:
            self.general_settings: dict = DEFAULT_GAP_JUNCTION_GENERAL_SETTINGS.copy()
        else:
            self.general_settings = {**DEFAULT_GAP_JUNCTION_GENERAL_SETTINGS, **general_settings}
        self.conn_type_settings = conn_type_settings

        self._syn_params_cache = {}
        self.config = config
        self.available_networks = []
        self.current_network = None
        if self.conn_type_settings is None and self.config is not None:
            self.conn_type_settings = self._build_conn_type_settings_from_config(self.config)
        if self.conn_type_settings is None or len(self.conn_type_settings) == 0:
            raise ValueError("conn_type_settings must be provided or config must be given to load gap junction connections from")
        self.current_connection = list(self.conn_type_settings.keys())[0]
        self.conn = self.conn_type_settings[self.current_connection]

        h.tstop = self.general_settings["tstart"] + self.general_settings["tdur"] + 100.0
        h.dt = self.general_settings["dt"]  # Time step (resolution) of the simulation in ms
        h.steps_per_ms = 1 / h.dt
        h.celsius = self.general_settings["celsius"]

        # Clean up any existing parallel context before setting up gap junctions
        try:
            pc_temp = h.ParallelContext()
            pc_temp.done()  # Clean up any existing parallel context
        except:
            pass  # Ignore errors if no existing context
        
        # Force cleanup
        import gc
        gc.collect()

        # set up gap junctions
        self.pc = h.ParallelContext()

        # Use provided hoc_cell or create new cells
        if self.hoc_cell is not None:
            self.cell1 = self.hoc_cell
            # For gap junctions, we need two cells, so create a second one if using hoc_cell
            self.cell_name = self.conn['cell']
            self.cell2 = getattr(h, self.cell_name)()
        else:
            print(self.conn)
            self.cell_name = self.conn['cell']
            self.cell1 = getattr(h, self.cell_name)()
            self.cell2 = getattr(h, self.cell_name)()

        self.icl = h.IClamp(self.cell1.soma[0](0.5))
        self.icl.delay = self.general_settings["tstart"]
        self.icl.dur = self.general_settings["tdur"]
        self.icl.amp = self.general_settings["iclamp_amp"]  # nA

        sec1 = list(self.cell1.all)[self.conn["sec_id"]]
        sec2 = list(self.cell2.all)[self.conn["sec_id"]]

        # Use unique IDs to avoid conflicts with existing parallel context setups
        import time
        unique_id = int(time.time() * 1000) % 10000  # Use timestamp as unique base ID
        
        self.pc.source_var(sec1(self.conn["sec_x"])._ref_v, unique_id, sec=sec1)
        self.gap_junc_1 = h.Gap(sec1(0.5))
        self.pc.target_var(self.gap_junc_1._ref_vgap, unique_id + 1)

        self.pc.source_var(sec2(self.conn["sec_x"])._ref_v, unique_id + 1, sec=sec2)
        self.gap_junc_2 = h.Gap(sec2(0.5))
        self.pc.target_var(self.gap_junc_2._ref_vgap, unique_id)

        self.pc.setup_transfer()
        
        # Now it's safe to initialize NEURON
        h.finitialize()

    def _load_synaptic_params_from_config(self, config: dict, dynamics_params: str) -> dict:
        try:
            # Get the synaptic models directory from config
            synaptic_models_dir = config.get('components', {}).get('synaptic_models_dir', '')
            if synaptic_models_dir:
                # Handle path variables
                if synaptic_models_dir.startswith('$'):
                    # This is a placeholder, try to resolve it
                    config_dir = os.path.dirname(config.get('config_path', ''))
                    synaptic_models_dir = synaptic_models_dir.replace('$COMPONENTS_DIR', 
                                                                    os.path.join(config_dir, 'components'))
                    synaptic_models_dir = synaptic_models_dir.replace('$BASE_DIR', config_dir)
                
                dynamics_file = os.path.join(synaptic_models_dir, dynamics_params)
                
                if os.path.exists(dynamics_file):
                    with open(dynamics_file, 'r') as f:
                        return json.load(f)
                else:
                    print(f"Warning: Dynamics params file not found: {dynamics_file}")
        except Exception as e:
            print(f"Warning: Error loading synaptic parameters: {e}")
        
        return {}

    def _load_available_networks(self) -> None:
        """
        Load available network names from the config file for the network dropdown feature.
        
        This method is automatically called during initialization when a config file is provided.
        It populates the available_networks list which enables the network dropdown in 
        InteractiveTuner when multiple networks are available.
        
        Network Dropdown Behavior:
        -------------------------
        - If only one network exists: No network dropdown is shown
        - If multiple networks exist: Network dropdown appears next to connection dropdown
        - Networks are loaded from the edges data in the config file
        - Current network defaults to the first available if not specified during init
        """
        if self.config is None:
            self.available_networks = []
            return
            
        try:
            edges = load_edges_from_config(self.config)
            self.available_networks = list(edges.keys())
            
            # Set current network to first available if not specified
            if self.current_network is None and self.available_networks:
                self.current_network = self.available_networks[0]
        except Exception as e:
            print(f"Warning: Could not load networks from config: {e}")
            self.available_networks = []

    def _build_conn_type_settings_from_config(self, config_path: str) -> Dict[str, dict]:
        # Load configuration and get nodes and edges using util.py methods
        config = load_config(config_path)
        # Ensure the config dict knows its source path so path substitutions can be resolved
        try:
            config['config_path'] = config_path
        except Exception:
            pass
        nodes = load_nodes_from_config(config_path)
        edges = load_edges_from_config(config_path)
        
        conn_type_settings = {}
        
        # Process all edge datasets
        for edge_dataset_name, edge_df in edges.items():
            if edge_df.empty:
                continue
            
            # Merging with node data to get model templates
            source_node_df = None
            target_node_df = None
            
            # First, try to deterministically parse the edge_dataset_name for patterns like '<src>_to_<tgt>'
            if '_to_' in edge_dataset_name:
                parts = edge_dataset_name.split('_to_')
                if len(parts) == 2:
                    src_name, tgt_name = parts
                    if src_name in nodes:
                        source_node_df = nodes[src_name].add_prefix('source_')
                    if tgt_name in nodes:
                        target_node_df = nodes[tgt_name].add_prefix('target_')
            
            # If not found by parsing name, fall back to inspecting a sample edge row
            if source_node_df is None or target_node_df is None:
                sample_edge = edge_df.iloc[0] if len(edge_df) > 0 else None
                if sample_edge is not None:
                    source_pop_name = sample_edge.get('source_population', '')
                    target_pop_name = sample_edge.get('target_population', '')
                    if source_pop_name in nodes:
                        source_node_df = nodes[source_pop_name].add_prefix('source_')
                    if target_pop_name in nodes:
                        target_node_df = nodes[target_pop_name].add_prefix('target_')
            
            # As a last resort, attempt to heuristically match
            if source_node_df is None or target_node_df is None:
                for pop_name, node_df in nodes.items():
                    if source_node_df is None and (edge_dataset_name.startswith(pop_name) or edge_dataset_name.endswith(pop_name)):
                        source_node_df = node_df.add_prefix('source_')
                    if target_node_df is None and (edge_dataset_name.startswith(pop_name) or edge_dataset_name.endswith(pop_name)):
                        target_node_df = node_df.add_prefix('target_')
            
            if source_node_df is None or target_node_df is None:
                print(f"Warning: Could not find node data for edge dataset {edge_dataset_name}")
                continue
            
            # Merge edge data with source node info
            edges_with_source = pd.merge(
                edge_df.reset_index(), 
                source_node_df, 
                how='left', 
                left_on='source_node_id', 
                right_index=True
            )
            
            # Merge with target node info
            edges_with_nodes = pd.merge(
                edges_with_source, 
                target_node_df, 
                how='left', 
                left_on='target_node_id', 
                right_index=True
            )
            
            # Skip edge datasets that don't have gap junction information
            if 'is_gap_junction' not in edges_with_nodes.columns:
                continue
            
            # Filter to only gap junction edges
            # Handle NaN values in is_gap_junction column
            gap_junction_mask = edges_with_nodes['is_gap_junction'].fillna(False) == True
            gap_junction_edges = edges_with_nodes[gap_junction_mask]
            if gap_junction_edges.empty:
                continue
            
            # Get unique edge types from the gap junction edges
            if 'edge_type_id' in gap_junction_edges.columns:
                edge_types = gap_junction_edges['edge_type_id'].unique()
            else:
                edge_types = [None]  # Single edge type
            
            # Process each edge type
            for edge_type_id in edge_types:
                # Filter edges for this type
                if edge_type_id is not None:
                    edge_type_data = gap_junction_edges[gap_junction_edges['edge_type_id'] == edge_type_id]
                else:
                    edge_type_data = gap_junction_edges
                
                if len(edge_type_data) == 0:
                    continue
                
                # Get representative edge for this type
                edge_info = edge_type_data.iloc[0]
                
                # Process gap junction
                source_model_template = edge_info.get('source_model_template', '')
                target_model_template = edge_info.get('target_model_template', '')
                
                source_cell_type = source_model_template.replace('hoc:', '') if source_model_template.startswith('hoc:') else source_model_template
                target_cell_type = target_model_template.replace('hoc:', '') if target_model_template.startswith('hoc:') else target_model_template
                
                if source_cell_type != target_cell_type:
                    continue  # Only process gap junctions between same cell types
                
                source_pop = edge_info.get('source_pop_name', '')
                target_pop = edge_info.get('target_pop_name', '')
                
                conn_name = f"{source_pop}2{target_pop}_gj"
                if edge_type_id is not None:
                    conn_name += f"_type_{edge_type_id}"
                
                conn_settings = {
                    'cell': source_cell_type,
                    'sec_id': 0,
                    'sec_x': 0.5,
                    'iclamp_amp': -0.01,
                    'spec_syn_param': {}
                }
                
                # Load dynamics params
                dynamics_file_name = edge_info.get('dynamics_params', '')
                if dynamics_file_name and dynamics_file_name.upper() != 'NULL':
                    try:
                        syn_params = self._load_synaptic_params_from_config(config, dynamics_file_name)
                        conn_settings['spec_syn_param'] = syn_params
                    except Exception as e:
                        print(f"Warning: could not load dynamics_params file '{dynamics_file_name}': {e}")
                
                conn_type_settings[conn_name] = conn_settings
        
        return conn_type_settings

    def _switch_connection(self, new_connection: str) -> None:
        """
        Switch to a different gap junction connection and update all related properties.
        
        Parameters:
        -----------
        new_connection : str
            Name of the new connection type to switch to.
        """
        if new_connection not in self.conn_type_settings:
            raise ValueError(f"Connection '{new_connection}' not found in conn_type_settings")
        
        # Update current connection
        self.current_connection = new_connection
        self.conn = self.conn_type_settings[new_connection]
        
        # Check if cell type changed
        new_cell_name = self.conn['cell']
        if self.cell_name != new_cell_name:
            self.cell_name = new_cell_name
            
            # Recreate cells
            if self.hoc_cell is None:
                self.cell1 = getattr(h, self.cell_name)()
                self.cell2 = getattr(h, self.cell_name)()
            else:
                # For hoc_cell, recreate the second cell
                self.cell2 = getattr(h, self.cell_name)()
            
            # Recreate IClamp
            self.icl = h.IClamp(self.cell1.soma[0](0.5))
            self.icl.delay = self.general_settings["tstart"]
            self.icl.dur = self.general_settings["tdur"]
            self.icl.amp = self.general_settings["iclamp_amp"]
        else:
            # Update IClamp parameters even if same cell type
            self.icl.amp = self.general_settings["iclamp_amp"]
        
        # Always recreate gap junctions when switching connections 
        # (even for same cell type, sec_id or sec_x might differ)
        
        # Clean up previous gap junctions and parallel context
        if hasattr(self, 'gap_junc_1'):
            del self.gap_junc_1
        if hasattr(self, 'gap_junc_2'):
            del self.gap_junc_2
        
        # Properly clean up the existing parallel context
        if hasattr(self, 'pc'):
            self.pc.done()  # Clean up existing parallel context
        
        # Force garbage collection and reset NEURON state
        import gc
        gc.collect()
        h.finitialize()
        
        # Create a fresh parallel context after cleanup
        self.pc = h.ParallelContext()
        
        try:
            sec1 = list(self.cell1.all)[self.conn["sec_id"]]
            sec2 = list(self.cell2.all)[self.conn["sec_id"]]
            
            # Use unique IDs to avoid conflicts with existing parallel context setups
            import time
            unique_id = int(time.time() * 1000) % 10000  # Use timestamp as unique base ID
            
            self.pc.source_var(sec1(self.conn["sec_x"])._ref_v, unique_id, sec=sec1)
            self.gap_junc_1 = h.Gap(sec1(0.5))
            self.pc.target_var(self.gap_junc_1._ref_vgap, unique_id + 1)
            
            self.pc.source_var(sec2(self.conn["sec_x"])._ref_v, unique_id + 1, sec=sec2)
            self.gap_junc_2 = h.Gap(sec2(0.5))
            self.pc.target_var(self.gap_junc_2._ref_vgap, unique_id)
            
            self.pc.setup_transfer()
        except Exception as e:
            print(f"Error setting up gap junctions: {e}")
            # Try to continue with basic setup
            self.gap_junc_1 = h.Gap(list(self.cell1.all)[self.conn["sec_id"]](0.5))
            self.gap_junc_2 = h.Gap(list(self.cell2.all)[self.conn["sec_id"]](0.5))
        
        # Reset NEURON state after complete setup
        h.finitialize()
        
        print(f"Successfully switched to connection: {new_connection}")

    def model(self, resistance):
        """
        Run a simulation with a specified gap junction resistance.

        Parameters:
        -----------
        resistance : float
            The gap junction resistance value (in MOhm) to use for the simulation.

        Notes:
        ------
        This method sets up the gap junction resistance, initializes recording vectors for time
        and membrane voltages of both cells, and runs the NEURON simulation.
        """
        self.gap_junc_1.g = resistance
        self.gap_junc_2.g = resistance

        t_vec = h.Vector()
        soma_v_1 = h.Vector()
        soma_v_2 = h.Vector()
        t_vec.record(h._ref_t)
        soma_v_1.record(self.cell1.soma[0](0.5)._ref_v)
        soma_v_2.record(self.cell2.soma[0](0.5)._ref_v)

        self.t_vec = t_vec
        self.soma_v_1 = soma_v_1
        self.soma_v_2 = soma_v_2

        h.finitialize(-70 * mV)
        h.continuerun(h.tstop * ms)

    def plot_model(self):
        """
        Plot the voltage traces of both cells to visualize gap junction coupling.

        This method creates a plot showing the membrane potential of both cells over time,
        highlighting the effect of gap junction coupling when a current step is applied to cell 1.
        """
        t_range = [
            self.general_settings["tstart"] - 100.0,
            self.general_settings["tstart"] + self.general_settings["tdur"] + 100.0,
        ]
        t = np.array(self.t_vec)
        v1 = np.array(self.soma_v_1)
        v2 = np.array(self.soma_v_2)
        tidx = (t >= t_range[0]) & (t <= t_range[1])

        plt.figure()
        plt.plot(t[tidx], v1[tidx], "b", label=f"{self.cell_name} 1")
        plt.plot(t[tidx], v2[tidx], "r", label=f"{self.cell_name} 2")
        plt.title(f"{self.cell_name} gap junction")
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Voltage (mV)")
        plt.legend()

    def coupling_coefficient(self, t, v1, v2, t_start, t_end, dt=h.dt):
        """
        Calculate the coupling coefficient between two cells connected by a gap junction.

        Parameters:
        -----------
        t : array-like
            Time vector.
        v1 : array-like
            Voltage trace of the cell receiving the current injection.
        v2 : array-like
            Voltage trace of the coupled cell.
        t_start : float
            Start time for calculating the steady-state voltage change.
        t_end : float
            End time for calculating the steady-state voltage change.
        dt : float, optional
            Time step of the simulation. Default is h.dt.

        Returns:
        --------
        float
            The coupling coefficient, defined as the ratio of voltage change in cell 2
            to voltage change in cell 1 (ΔV₂/ΔV₁).
        """
        t = np.asarray(t)
        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        idx1 = np.nonzero(t < t_start)[0][-1]
        idx2 = np.nonzero(t < t_end)[0][-1]
        return (v2[idx2] - v2[idx1]) / (v1[idx2] - v1[idx1])

    def InteractiveTuner(self):
        w_run = widgets.Button(description="Run", icon="history", button_style="primary")
        values = [i * 10**-4 for i in range(1, 1001)]  # From 1e-4 to 1e-1

        # Create the SelectionSlider widget with appropriate formatting
        resistance = widgets.FloatLogSlider(
            value=0.001,
            base=10,
            min=-4,  # max exponent of base
            max=-1,  # min exponent of base
            step=0.1,  # exponent step
            description="Resistance: ",
            continuous_update=True,
        )

        output = widgets.Output()

        ui_widgets = [w_run, resistance]

        def on_button(*args):
            with output:
                # Clear only the output widget, not the entire cell
                output.clear_output(wait=True)

                resistance_for_gap = resistance.value
                print(f"Running simulation with resistance: {resistance_for_gap:0.6f} and {self.general_settings['iclamp_amp']*1000}pA current clamps")

                try:
                    self.model(resistance_for_gap)
                    self.plot_model()

                    # Convert NEURON vectors to numpy arrays
                    t_array = np.array(self.t_vec)
                    v1_array = np.array(self.soma_v_1)
                    v2_array = np.array(self.soma_v_2)

                    cc = self.coupling_coefficient(t_array, v1_array, v2_array, 500, 1000)
                    print(f"coupling_coefficient is {cc:0.4f}")
                    plt.show()

                except Exception as e:
                    print(f"Error during simulation or analysis: {e}")
                    import traceback

                    traceback.print_exc()

        # Add connection dropdown if multiple connections exist
        if len(self.conn_type_settings) > 1:
            connection_dropdown = widgets.Dropdown(
                options=list(self.conn_type_settings.keys()),
                value=self.current_connection,
                description='Connection:',
            )
            def on_connection_change(change):
                if change['type'] == 'change' and change['name'] == 'value':
                    self._switch_connection(change['new'])
                    on_button()  # Automatically rerun the simulation after switching
            connection_dropdown.observe(on_connection_change)
            ui_widgets.insert(0, connection_dropdown)

        ui = VBox(ui_widgets)

        display(ui)
        display(output)

        # Run once initially
        on_button()
        w_run.on_click(on_button)


# optimizers!


@dataclass
class SynapseOptimizationResult:
    """Container for synaptic parameter optimization results"""

    optimal_params: Dict[str, float]
    achieved_metrics: Dict[str, float]
    target_metrics: Dict[str, float]
    error: float
    optimization_path: List[Dict[str, float]]


class SynapseOptimizer:
    def __init__(self, tuner):
        """
        Initialize the synapse optimizer with parameter scaling

        Parameters:
        -----------
        tuner : SynapseTuner
            Instance of the SynapseTuner class
        """
        self.tuner = tuner
        self.optimization_history = []
        self.param_scales = {}

    def _normalize_params(self, params: np.ndarray, param_names: List[str]) -> np.ndarray:
        """
        Normalize parameters to similar scales for better optimization performance.

        Parameters:
        -----------
        params : np.ndarray
            Original parameter values.
        param_names : List[str]
            Names of the parameters corresponding to the values.

        Returns:
        --------
        np.ndarray
            Normalized parameter values.
        """
        return np.array([params[i] / self.param_scales[name] for i, name in enumerate(param_names)])

    def _denormalize_params(
        self, normalized_params: np.ndarray, param_names: List[str]
    ) -> np.ndarray:
        """
        Convert normalized parameters back to original scale.

        Parameters:
        -----------
        normalized_params : np.ndarray
            Normalized parameter values.
        param_names : List[str]
            Names of the parameters corresponding to the normalized values.

        Returns:
        --------
        np.ndarray
            Denormalized parameter values in their original scale.
        """
        return np.array(
            [normalized_params[i] * self.param_scales[name] for i, name in enumerate(param_names)]
        )

    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate standard metrics from the current simulation.

        This method runs either a single event simulation, a train input simulation,
        or both based on configuration flags, and calculates relevant synaptic metrics.

        Returns:
        --------
        Dict[str, float]
            Dictionary of calculated metrics including:
            - induction: measure of synaptic facilitation/depression
            - ppr: paired-pulse ratio
            - recovery: recovery from facilitation/depression
            - max_amplitude: maximum synaptic response amplitude
            - rise_time: time for synaptic response to rise from 20% to 80% of peak
            - decay_time: time constant of synaptic response decay
            - latency: synaptic response latency
            - half_width: synaptic response half-width
            - baseline: baseline current
            - amp: peak amplitude from syn_props
        """
        # Set these to 0 for when we return the dict
        induction = 0
        ppr = 0
        recovery = 0
        amp = 0
        rise_time = 0
        decay_time = 0
        latency = 0
        half_width = 0
        baseline = 0
        syn_amp = 0

        if self.run_single_event:
            self.tuner.SingleEvent(plot_and_print=False)
            # Use the attributes set by SingleEvent method
            rise_time = getattr(self.tuner, "rise_time", 0)
            decay_time = getattr(self.tuner, "decay_time", 0)
            # Get additional syn_props directly
            syn_props = self.tuner._get_syn_prop()
            latency = syn_props.get("latency", 0)
            half_width = syn_props.get("half_width", 0)
            baseline = syn_props.get("baseline", 0)
            syn_amp = syn_props.get("amp", 0)

        if self.run_train_input:
            self.tuner._simulate_model(self.train_frequency, self.train_delay)
            amp = self.tuner._response_amplitude()
            ppr, induction, recovery = self.tuner._calc_ppr_induction_recovery(
                amp, print_math=False
            )
            amp = self.tuner._find_max_amp(amp)

        return {
            "induction": float(induction),
            "ppr": float(ppr),
            "recovery": float(recovery),
            "max_amplitude": float(amp),
            "rise_time": float(rise_time),
            "decay_time": float(decay_time),
            "latency": float(latency),
            "half_width": float(half_width),
            "baseline": float(baseline),
            "amp": float(syn_amp),
        }

    def _default_cost_function(
        self, metrics: Dict[str, float], target_metrics: Dict[str, float]
    ) -> float:
        """
        Default cost function that minimizes the squared difference between achieved and target induction.

        Parameters:
        -----------
        metrics : Dict[str, float]
            Dictionary of calculated metrics from the current simulation.
        target_metrics : Dict[str, float]
            Dictionary of target metrics to optimize towards.

        Returns:
        --------
        float
            The squared error between achieved and target induction.
        """
        return float((metrics["induction"] - target_metrics["induction"]) ** 2)

    def _objective_function(
        self,
        normalized_params: np.ndarray,
        param_names: List[str],
        cost_function: Callable,
        target_metrics: Dict[str, float],
    ) -> float:
        """
        Calculate error using provided cost function
        """
        # Denormalize parameters
        params = self._denormalize_params(normalized_params, param_names)

        # Set parameters
        for name, value in zip(param_names, params):
            setattr(self.tuner.syn, name, value)

        # just do this and have the SingleEvent handle it
        if self.run_single_event:
            self.tuner.using_optimizer = True
            self.tuner.param_names = param_names
            self.tuner.params = params

        # Calculate metrics and error
        metrics = self._calculate_metrics()
        error = float(cost_function(metrics, target_metrics))  # Ensure error is scalar

        # Store history with denormalized values
        history_entry = {
            "params": dict(zip(param_names, params)),
            "metrics": metrics,
            "error": error,
        }
        self.optimization_history.append(history_entry)

        return error

    def optimize_parameters(
        self,
        target_metrics: Dict[str, float],
        param_bounds: Dict[str, Tuple[float, float]],
        run_single_event: bool = False,
        run_train_input: bool = True,
        train_frequency: float = 50,
        train_delay: float = 250,
        cost_function: Optional[Callable] = None,
        method: str = "SLSQP",
        init_guess="random",
    ) -> SynapseOptimizationResult:
        """
        Optimize synaptic parameters to achieve target metrics.

        Parameters:
        -----------
        target_metrics : Dict[str, float]
            Target values for synaptic metrics (e.g., {'induction': 0.2, 'rise_time': 0.5})
        param_bounds : Dict[str, Tuple[float, float]]
            Bounds for each parameter to optimize (e.g., {'tau_d': (5, 50), 'Use': (0.1, 0.9)})
        run_single_event : bool, optional
            Whether to run single event simulations during optimization (default: False)
        run_train_input : bool, optional
            Whether to run train input simulations during optimization (default: True)
        train_frequency : float, optional
            Frequency of the stimulus train in Hz (default: 50)
        train_delay : float, optional
            Delay between pulse trains in ms (default: 250)
        cost_function : Optional[Callable]
            Custom cost function for optimization. If None, uses default cost function
            that optimizes induction.
        method : str, optional
            Optimization method to use (default: 'SLSQP')
        init_guess : str, optional
            Method for initial parameter guess ('random' or 'middle_guess')

        Returns:
        --------
        SynapseOptimizationResult
            Results of the optimization including optimal parameters, achieved metrics,
            target metrics, final error, and optimization path.

        Notes:
        ------
        This function uses scipy.optimize.minimize to find the optimal parameter values
        that minimize the difference between achieved and target metrics.
        """
        self.optimization_history = []
        self.train_frequency = train_frequency
        self.train_delay = train_delay
        self.run_single_event = run_single_event
        self.run_train_input = run_train_input

        param_names = list(param_bounds.keys())
        bounds = [param_bounds[name] for name in param_names]

        if cost_function is None:
            cost_function = self._default_cost_function

        # Calculate scaling factors
        self.param_scales = {
            name: max(abs(bounds[i][0]), abs(bounds[i][1])) for i, name in enumerate(param_names)
        }

        # Normalize bounds
        normalized_bounds = [
            (b[0] / self.param_scales[name], b[1] / self.param_scales[name])
            for name, b in zip(param_names, bounds)
        ]

        # picks with method of init value we want to use
        if init_guess == "random":
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        elif init_guess == "middle_guess":
            x0 = [(b[0] + b[1]) / 2 for b in bounds]
        else:
            raise Exception("Pick a vaid init guess method either random or midde_guess")
        normalized_x0 = self._normalize_params(np.array(x0), param_names)

        # Run optimization
        result = minimize(
            self._objective_function,
            normalized_x0,
            args=(param_names, cost_function, target_metrics),
            method=method,
            bounds=normalized_bounds,
        )

        # Get final parameters and metrics
        final_params = dict(zip(param_names, self._denormalize_params(result.x, param_names)))
        for name, value in final_params.items():
            setattr(self.tuner.syn, name, value)
        final_metrics = self._calculate_metrics()

        return SynapseOptimizationResult(
            optimal_params=final_params,
            achieved_metrics=final_metrics,
            target_metrics=target_metrics,
            error=result.fun,
            optimization_path=self.optimization_history,
        )

    def plot_optimization_results(self, result: SynapseOptimizationResult):
        """
        Plot optimization results including convergence and final traces.

        Parameters:
        -----------
        result : SynapseOptimizationResult
            Results from optimization as returned by optimize_parameters()

        Notes:
        ------
        This method generates three plots:
        1. Error convergence plot showing how the error decreased over iterations
        2. Parameter convergence plots showing how each parameter changed
        3. Final model response with the optimal parameters

        It also prints a summary of the optimization results including target vs. achieved
        metrics and the optimal parameter values.
        """
        # Ensure errors are properly shaped for plotting
        iterations = range(len(result.optimization_path))
        errors = np.array([float(h["error"]) for h in result.optimization_path]).flatten()

        # Plot error convergence
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(iterations, errors, label="Error")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Error")
        ax1.set_title("Error Convergence")
        ax1.set_yscale("log")
        ax1.legend()
        plt.tight_layout()
        plt.show()

        # Plot parameter convergence
        param_names = list(result.optimal_params.keys())
        num_params = len(param_names)
        fig2, axs = plt.subplots(nrows=num_params, ncols=1, figsize=(8, 5 * num_params))

        if num_params == 1:
            axs = [axs]

        for ax, param in zip(axs, param_names):
            values = [float(h["params"][param]) for h in result.optimization_path]
            ax.plot(iterations, values, label=f"{param}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Parameter Value")
            ax.set_title(f"Convergence of {param}")
            ax.legend()

        plt.tight_layout()
        plt.show()

        # Print final results
        print("Optimization Results:")
        print(f"Final Error: {float(result.error):.2e}\n")
        print("Target Metrics:")
        for metric, value in result.target_metrics.items():
            achieved = result.achieved_metrics.get(metric)
            if achieved is not None and metric != "amplitudes":  # Skip amplitude array
                print(f"{metric}: {float(achieved):.3f} (target: {float(value):.3f})")

        print("\nOptimal Parameters:")
        for param, value in result.optimal_params.items():
            print(f"{param}: {float(value):.3f}")

        # Plot final model response
        if self.run_train_input:
            self.tuner._plot_model(
                [
                    self.tuner.general_settings["tstart"] - self.tuner.nstim.interval / 3,
                    self.tuner.tstop,
                ]
            )
            amp = self.tuner._response_amplitude()
            self.tuner._calc_ppr_induction_recovery(amp)
        if self.run_single_event:
            self.tuner.ispk = None
            self.tuner.SingleEvent(plot_and_print=True)

# dataclass means just init the typehints as self.typehint. looks a bit cleaner
@dataclass
class GapOptimizationResult:
    """Container for gap junction optimization results"""

    optimal_resistance: float
    achieved_cc: float
    target_cc: float
    error: float
    optimization_path: List[Dict[str, float]]


class GapJunctionOptimizer:
    def __init__(self, tuner):
        """
        Initialize the gap junction optimizer

        Parameters:
        -----------
        tuner : GapJunctionTuner
            Instance of the GapJunctionTuner class
        """
        self.tuner = tuner
        self.optimization_history = []

    def _objective_function(self, resistance: float, target_cc: float) -> float:
        """
        Calculate error between achieved and target coupling coefficient

        Parameters:
        -----------
        resistance : float
            Gap junction resistance to try
        target_cc : float
            Target coupling coefficient to match

        Returns:
        --------
        float : Error between achieved and target coupling coefficient
        """
        # Run model with current resistance
        self.tuner.model(resistance)

        # Calculate coupling coefficient
        achieved_cc = self.tuner.coupling_coefficient(
            self.tuner.t_vec,
            self.tuner.soma_v_1,
            self.tuner.soma_v_2,
            self.tuner.general_settings["tstart"],
            self.tuner.general_settings["tstart"] + self.tuner.general_settings["tdur"],
        )

        # Calculate error
        error = (achieved_cc - target_cc) ** 2  # MSE

        # Store history
        self.optimization_history.append(
            {"resistance": resistance, "achieved_cc": achieved_cc, "error": error}
        )

        return error

    def optimize_resistance(
        self, target_cc: float, resistance_bounds: tuple = (1e-4, 1e-2), method: str = "bounded"
    ) -> GapOptimizationResult:
        """
        Optimize gap junction resistance to achieve a target coupling coefficient.

        Parameters:
        -----------
        target_cc : float
            Target coupling coefficient to achieve (between 0 and 1)
        resistance_bounds : tuple, optional
            (min, max) bounds for resistance search in MOhm. Default is (1e-4, 1e-2).
        method : str, optional
            Optimization method to use. Default is 'bounded' which works well
            for single-parameter optimization.

        Returns:
        --------
        GapOptimizationResult
            Container with optimization results including:
            - optimal_resistance: The optimized resistance value
            - achieved_cc: The coupling coefficient achieved with the optimal resistance
            - target_cc: The target coupling coefficient
            - error: The final error (squared difference between target and achieved)
            - optimization_path: List of all values tried during optimization

        Notes:
        ------
        Uses scipy.optimize.minimize_scalar with bounded method, which is
        appropriate for this single-parameter optimization problem.
        """
        self.optimization_history = []

        # Run optimization
        result = minimize_scalar(
            self._objective_function, args=(target_cc,), bounds=resistance_bounds, method=method
        )

        # Run final model with optimal resistance
        self.tuner.model(result.x)
        final_cc = self.tuner.coupling_coefficient(
            self.tuner.t_vec,
            self.tuner.soma_v_1,
            self.tuner.soma_v_2,
            self.tuner.general_settings["tstart"],
            self.tuner.general_settings["tstart"] + self.tuner.general_settings["tdur"],
        )

        # Package up our results
        optimization_result = GapOptimizationResult(
            optimal_resistance=result.x,
            achieved_cc=final_cc,
            target_cc=target_cc,
            error=result.fun,
            optimization_path=self.optimization_history,
        )

        return optimization_result

    def plot_optimization_results(self, result: GapOptimizationResult):
        """
        Plot optimization results including convergence and final voltage traces

        Parameters:
        -----------
        result : GapOptimizationResult
            Results from optimization
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot voltage traces
        t_range = [
            self.tuner.general_settings["tstart"] - 100.0,
            self.tuner.general_settings["tstart"] + self.tuner.general_settings["tdur"] + 100.0,
        ]
        t = np.array(self.tuner.t_vec)
        v1 = np.array(self.tuner.soma_v_1)
        v2 = np.array(self.tuner.soma_v_2)
        tidx = (t >= t_range[0]) & (t <= t_range[1])

        ax1.plot(t[tidx], v1[tidx], "b", label=f"{self.tuner.cell_name} 1")
        ax1.plot(t[tidx], v2[tidx], "r", label=f"{self.tuner.cell_name} 2")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Membrane Voltage (mV)")
        ax1.legend()
        ax1.set_title("Optimized Voltage Traces")

        # Plot error convergence
        errors = [h["error"] for h in result.optimization_path]
        ax2.plot(errors)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Error")
        ax2.set_title("Error Convergence")
        ax2.set_yscale("log")

        # Plot resistance convergence
        resistances = [h["resistance"] for h in result.optimization_path]
        ax3.plot(resistances)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Resistance")
        ax3.set_title("Resistance Convergence")
        ax3.set_yscale("log")

        # Print final results
        result_text = (
            f"Optimal Resistance: {result.optimal_resistance:.2e}\n"
            f"Target CC: {result.target_cc:.3f}\n"
            f"Achieved CC: {result.achieved_cc:.3f}\n"
            f"Final Error: {result.error:.2e}"
        )
        ax4.text(0.1, 0.7, result_text, transform=ax4.transAxes, fontsize=10)
        ax4.axis("off")

        plt.tight_layout()
        plt.show()

    def parameter_sweep(self, resistance_range: np.ndarray) -> dict:
        """
        Perform a parameter sweep across different resistance values.

        Parameters:
        -----------
        resistance_range : np.ndarray
            Array of resistance values to test.

        Returns:
        --------
        dict
            Dictionary containing the results of the parameter sweep, with keys:
            - 'resistance': List of resistance values tested
            - 'coupling_coefficient': Corresponding coupling coefficients

        Notes:
        ------
        This method is useful for understanding the relationship between gap junction
        resistance and coupling coefficient before attempting optimization.
        """
        results = {"resistance": [], "coupling_coefficient": []}

        for resistance in tqdm(resistance_range, desc="Sweeping resistance values"):
            self.tuner.model(resistance)
            cc = self.tuner.coupling_coefficient(
                self.tuner.t_vec,
                self.tuner.soma_v_1,
                self.tuner.soma_v_2,
                self.tuner.general_settings["tstart"],
                self.tuner.general_settings["tstart"] + self.tuner.general_settings["tdur"],
            )

            results["resistance"].append(resistance)
            results["coupling_coefficient"].append(cc)

        return results