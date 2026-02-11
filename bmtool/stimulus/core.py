import os
import numpy as np
import pandas as pd
from bmtool.util import util
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
from . import generators, assemblies

class StimulusBuilder:
    """Class to manage and generate stimuli for BMTK networks.
    
    This class provides a unified interface for defining node assemblies and 
    generating time-varying Poisson spike trains (SONATA format) for those assemblies.
    
    Attributes:
        config (dict): BMTK simulation configuration.
        nodes (dict): Dictionary of pandas DataFrames for each network.
        assemblies (dict): Named groups of node IDs.
        net_seed (int): Seed for assembly generation.
        psg_seed (int): Seed for Poisson spike generation.
    """

    def __init__(self, config=None, net_seed=123, psg_seed=1):
        """Initialize the StimulusBuilder.
        
        Args:
            config (str/dict, optional): Path to BMTK config file or dictionary.
            net_seed (int): Random seed for assembly assignment (default: 123).
            psg_seed (int): Random seed for Poisson spike generation (default: 1).
        """
        self.config = config
        self.nodes = util.load_nodes_from_config(config)
        self.assemblies = {} 
        self.net_seed = net_seed
        self.psg_seed = psg_seed
        self.rng = np.random.default_rng(net_seed)

    def get_nodes(self, network_name, pop_name=None, node_query=None):
        """Helper to get node IDs from loaded nodes.
        
        Args:
            network_name (str): Name of the network (e.g., 'thalamus').
            pop_name (str, optional): Filter by 'pop_name' column.
            node_query (str, optional): Placeholder for custom filtering logic.
            
        Returns:
            pd.DataFrame: Filtered nodes.
        """
        if network_name not in self.nodes:
            raise ValueError(f"Network {network_name} not found in configuration.")
        
        df = self.nodes[network_name]
        
        if pop_name:
            df = util.get_pop(df, pop_name, key='pop_name')
            
        if node_query:
            # Generic query support if needed, for now just custom DF filtering by user
            pass
            
        return df

    def create_assemblies(self, name, network_name, method='random', seed=None, **kwargs):
        """Create node assemblies (subsets) and store them by name.
        
        Args:
            name (str): Unique name for this assembly group.
            network_name (str): Name of the network to draw nodes from.
            method (str): Method to group nodes. Options:
                - 'random': Assign nodes to n_assemblies randomly.
                - 'grid': Group nodes into a spatial grid based on x, y position.
                - 'property': Group nodes by a column name (e.g., 'pulse_group_id').
            seed (int, optional): Random seed for reproducibility. Overrides instance net_seed for this call.
            **kwargs: Arguments passed to the respective assembly generator:
                - pop_name (str): Filter nodes before assembly creation.
                - n_assemblies (int): Number of assemblies for 'random'.
                - prob_in_assembly (float): Probability of node inclusion (0-1).
                - property_name (str): Column name for 'property' grouping.
                - grid_id (ndarray): 2D array of assembly IDs for 'grid'.
                - grid_size (list): [[min_x, max_x], [min_y, max_y]] for 'grid'.
        """
        nodes_df = self.get_nodes(network_name, kwargs.get('pop_name'))
        node_ids = nodes_df.index.values
        
        # Use provided seed or default to instance net_seed
        rng = np.random.default_rng(seed if seed is not None else self.net_seed)
        
        if method == 'random':
            n_assemblies = kwargs.get('n_assemblies', 1)
            prob = kwargs.get('prob_in_assembly', 1.0)
            
            # Use utility to get assignments
            assy_indices = assemblies.assign_assembly(
                len(node_ids), n_assemblies, rng=rng, seed=None, prob_in_assembly=prob
            )
            
            # Map back to node IDs
            assembly_list = assemblies.get_assembly_ids(node_ids, assy_indices)
            self.assemblies[name] = assembly_list
            
        elif method == 'grid':
            grid_id = kwargs.get('grid_id')
            grid_size = kwargs.get('grid_size')
            
            nodes_assy, _ = assemblies.get_grid_assembly(nodes_df, grid_id, grid_size)
            self.assemblies[name] = nodes_assy
            
        elif method == 'property':
            prop_name = kwargs.get('property_name')
            prob = kwargs.get('probability', 1.0)
            
            assembly_list = assemblies.get_assemblies_by_property(
                nodes_df, prop_name, probability=prob, rng=rng, seed=None
            )
            self.assemblies[name] = assembly_list
            
        else:
            raise ValueError(f"Unknown assembly method: {method}")
    
    def _generate_firing_rates(self, n_nodes, mean, stdev, distribution='lognormal'):
        """Helper to generate firing rates based on distribution.
        
        Args:
            n_nodes (int): Number of rates to generate.
            mean (float): Mean firing rate.
            stdev (float): Standard deviation of firing rates.
            distribution (str): 'lognormal' or 'normal'.
            
        Returns:
            np.ndarray: Array of firing rates.
        """
        if distribution == 'lognormal':
            sigma2 = np.log((stdev / mean) ** 2 + 1)
            mu = np.log(mean) - sigma2 / 2
            sigma = sigma2 ** 0.5
            rates = self.rng.lognormal(mu, sigma, n_nodes)
        elif distribution == 'normal':
            rates = self.rng.normal(mean, stdev, n_nodes)
            rates = np.maximum(rates, 0.0)  # Clamp to non-negative
        else:
            raise ValueError(f"Unknown distribution: {distribution}. Must be 'lognormal' or 'normal'.")
        
        return rates
    
    def generate_background(self, output_path, network_name, population_params,
                           groupby='pop_name', t_start=0.0, t_stop=10.0, 
                           verbose=False, seed=None):
        """Generate background (spontaneous) activity for network nodes grouped by property.
        
        This function generates baseline spiking activity, grouped by a specified node property.
        Each group can use either a constant firing rate or a distribution-based rate.
        
        Args:
            output_path (str): Path to save the resulting .h5 file.
            network_name (str): BMTK network name.
            population_params (dict): Parameters for each population/group.
                Keys should match values in the node property specified by groupby.
                Each value is a dict with:
                    - 'mean_firing_rate' (float): Mean firing rate in Hz (required)
                    - 'stdev' (float, optional): Standard deviation. If provided, uses lognormal distribution.
                                                If omitted, uses constant firing rate.
                Example:
                    {
                        'PN': {'mean_firing_rate': 20.0, 'stdev': 2.0},
                        'PV': {'mean_firing_rate': 30.0},  # constant rate
                        'SST': {'mean_firing_rate': 15.0, 'stdev': 1.5}
                    }
            groupby (str): Node property to group by (default: 'pop_name').
                          Will match against keys in population_params.
            t_start, t_stop (float): Time range for activity (seconds).
            verbose (bool): If True, print detailed information (default: False).
            seed (int, optional): Random seed for distribution sampling. Overrides instance psg_seed.
            
        Examples:
            # Population-specific rates with mixed distributions
            params = {
                'PN': {'mean_firing_rate': 20.0, 'stdev': 2.0},
                'PV': {'mean_firing_rate': 30.0},  # constant rate
                'SST': {'mean_firing_rate': 15.0, 'stdev': 1.5}
            }
            sb.generate_background(
                output_path='background.h5',
                network_name='input',
                population_params=params,
                t_start=0.0, t_stop=15.0
            )
            
            # Group by custom property (e.g., layer)
            layer_params = {
                'L1': {'mean_firing_rate': 10.0, 'stdev': 1.0},
                'L2/3': {'mean_firing_rate': 15.0, 'stdev': 2.0}
            }
            sb.generate_background(
                output_path='layer_background.h5',
                network_name='input',
                population_params=layer_params,
                groupby='layer'
            )
        """
        if population_params is None or not isinstance(population_params, dict):
            raise ValueError("population_params must be a non-empty dict")
        
        nodes_df = self.get_nodes(network_name)
        
        # Verify groupby column exists
        if groupby not in nodes_df.columns:
            raise ValueError(f"Node property '{groupby}' not found in network '{network_name}'")
        
        # Use provided seed or default to instance psg_seed
        psg_seed = seed if seed is not None else self.psg_seed
        
        population = network_name  # Default population name in PSG
        psg = PoissonSpikeGenerator(population=population, seed=psg_seed)
        
        times = (t_start, t_stop)
        total_nodes = 0
        
        for group_key, params in population_params.items():
            # Find nodes matching this group
            nodes_in_group = nodes_df[nodes_df[groupby] == group_key].index.values
            
            if len(nodes_in_group) == 0:
                if verbose:
                    print(f"  Warning: No nodes found with {groupby}='{group_key}'")
                continue
            
            total_nodes += len(nodes_in_group)
            
            if not isinstance(params, dict) or 'mean_firing_rate' not in params:
                raise ValueError(f"params['{group_key}'] must be a dict with 'mean_firing_rate' key")
            
            mean_rate = params['mean_firing_rate']
            stdev = params.get('stdev', None)
            
            # Determine: constant vs distribution-based
            if stdev is not None:
                # Use distribution (lognormal)
                firing_rates = self._generate_firing_rates(len(nodes_in_group), mean_rate, stdev, 'lognormal')
                for node_id, rate in zip(nodes_in_group, firing_rates):
                    psg.add(node_ids=node_id, firing_rate=rate, times=times)
                if verbose:
                    print(f"  {group_key}: {len(nodes_in_group)} nodes, {mean_rate:.1f}±{stdev:.1f} Hz (lognormal)")
            else:
                # Use constant firing rate
                psg.add(node_ids=nodes_in_group.tolist(), firing_rate=mean_rate, times=times)
                if verbose:
                    print(f"  {group_key}: {len(nodes_in_group)} nodes, {mean_rate:.1f} Hz (constant)")
        
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        psg.to_sonata(output_path)
        if verbose:
            print(f"Generated background activity: {total_nodes} nodes to {output_path}")
            
    def generate_stimulus(self, output_path, pattern_type, assembly_name, verbose=False, seed=None, **kwargs):
        """Generate a BMTK Poisson spike file (SONATA) for a specific assembly group.
        
        Use create_assemblies() first to define your stimulus assemblies, then call this
        function to generate time-varying firing patterns for those assemblies.
        
        Args:
            output_path (str): Path to save the resulting .h5 file.
            pattern_type (str): Firing rate template ('short', 'long', 'ramp', etc).
            assembly_name (str): Name of the assembly group created via create_assemblies.
            verbose (bool): If True, print detailed information (default: False).
            seed (int, optional): Random seed for Poisson spike generation. Overrides instance psg_seed.
            **kwargs: Arguments passed to the generator function and PoissonSpikeGenerator.
                - population (str): Name of the spike population (for BMTK).
                - firing_rate (3-tuple): (off_rate, burst_rate, silent_rate).
                - on_time (float): Duration of active period.
                - off_time (float): Duration of silent period.
                - t_start (float): Start time of cycles.
                - t_stop (float): End time of cycles.
                
        Example:
            # First create assemblies
            sb.create_assemblies(name='stim_groups', network_name='thalamus', 
                                method='property', property_name='pulse_group_id')
            
            # Then generate stimulus
            sb.generate_stimulus(output_path='stim.h5', pattern_type='long', 
                                assembly_name='stim_groups', population='thalamus',
                                firing_rate=(0.0, 50.0, 0.0), t_start=1.0, t_stop=15.0,
                                on_time=1.0, off_time=0.5)
        """
        if assembly_name not in self.assemblies:
            raise ValueError(f"Assembly '{assembly_name}' not defined. Use create_assemblies() first.")
            
        assembly_list = self.assemblies[assembly_name]
        n_assemblies = len(assembly_list)

        # Get population name for PSG (consumed here)
        population = kwargs.pop('population', 'stimulus')
        
        # Use provided seed or default to instance psg_seed
        psg_seed = seed if seed is not None else self.psg_seed
        
        # Dispatch to generator
        generator_func = getattr(generators, f"get_fr_{pattern_type}", None)
        if not generator_func:
             raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Generate traces
        params = generator_func(n_assemblies=n_assemblies, verbose=verbose, **kwargs)
        
        # Create PSG
        psg = PoissonSpikeGenerator(population=population, seed=psg_seed)
        
        # Add spikes
        if verbose:
            print(f"Generating spiking for {n_assemblies} assemblies...")
        for ids, param_dict in zip(assembly_list, params):
            psg.add(node_ids=ids, firing_rate=param_dict['firing_rate'], times=param_dict['times'])
            
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        psg.to_sonata(output_path)
        if verbose:
            print(f"Written stimulus to {output_path}")
