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
            
    def generate_stimulus(self, output_path, pattern_type, assembly_name, verbose=False, seed=None, **kwargs):
        """Generate a BMTK Poisson spike file (SONATA) for a specific assembly group.
        
        Args:
            output_path (str): Path to save the resulting .h5 file.
            pattern_type (str): Firing rate template ('short', 'long', 'ramp', etc).
            assembly_name (str): Name of the assembly group created via create_assemblies.
            verbose (bool): If True, print detailed information (default: False).
            seed (int, optional): Random seed for Poisson spike generation. Overrides instance psg_seed for this call.
            **kwargs: Arguments passed to the generator function and PoissonSpikeGenerator.
                - population (str): Name of the spike population (for BMTK).
                - firing_rate (3-tuple): (off_rate, burst_rate, silent_rate).
                - on_time (float): Duration of active period.
                - off_time (float): Duration of silent period.
                - t_start (float): Start time of cycles.
                - t_stop (float): End time of cycles.
        """
        if assembly_name not in self.assemblies:
            raise ValueError(f"Assembly '{assembly_name}' not defined.")
            
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

    def generate_baseline(self, output_path, network_name, pop_name=None, distribution='constant', 
                         mean=None, stdev=None, firing_rate=None, t_start=0.0, t_stop=10.0, verbose=False, seed=None):
        """Generate baseline activity for a selection of nodes.
        
        Args:
            output_path (str): Path to save the resulting .h5 file.
            network_name (str): BMTK network name.
            pop_name (str, optional): Filter nodes by population name.
            distribution (str): 'constant', 'lognormal', or 'normal'.
            mean (float): Mean for lognormal/normal or constant rate (if firing_rate omitted).
            stdev (float): Standard deviation for lognormal/normal.
            firing_rate (float, optional): Constant firing rate.
            t_start, t_stop (float): Time range for activity.
            verbose (bool): If True, print detailed information (default: False).
            seed (int, optional): Random seed for distribution sampling. Overrides instance psg_seed for this call.
        """
        nodes_df = self.get_nodes(network_name, pop_name)
        node_ids = nodes_df.index.values.tolist()
        
        # Use provided seed or default to instance psg_seed
        psg_seed = seed if seed is not None else self.psg_seed
        
        population = network_name # Default population name in PSG
        psg = PoissonSpikeGenerator(population=population, seed=psg_seed)
        
        times = (t_start, t_stop)
        
        if distribution == 'constant':
            if firing_rate is None:
                if mean is not None: 
                    firing_rate = mean
                else:
                    raise ValueError("Must provide firing_rate for constant distribution")
            
            psg.add(node_ids=node_ids, firing_rate=firing_rate, times=times)
            
        elif distribution in ['lognormal', 'normal']:
            if mean is None or stdev is None:
                raise ValueError(f"Must provide mean and stdev for {distribution} distribution")
            
            firing_rates = self._generate_firing_rates(len(node_ids), mean, stdev, distribution)
            
            for node_id, fr in zip(node_ids, firing_rates):
                psg.add(node_ids=node_id, firing_rate=fr, times=times)
                
        else:
             raise ValueError(f"Unknown distribution: {distribution}")
             
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        psg.to_sonata(output_path)
        if verbose:
            print(f"Written baseline to {output_path}")

    def generate_shell_input(self, output_path, network_name, shell_params, 
                            distribution='lognormal', t_start=0.0, t_stop=15.0, verbose=False, seed=None):
        """Generate shell (background) stimulus with population-specific rates.
        
        Args:
            output_path (str): Path to save the resulting .h5 file.
            network_name (str): BMTK network name.
            shell_params (dict): Population-specific (mean, stdev) tuples.
                Example: {'ET': (1.9, 1.8), 'IT': (1.3, 1.4), 'PV': (7.5, 6.4), 'SST': (5.0, 6.0)}
            distribution (str): 'lognormal' or 'normal' (default: 'lognormal').
            t_start, t_stop (float): Time range for activity.
            verbose (bool): If True, print detailed information (default: False).
            seed (int, optional): Random seed for distribution sampling. Overrides instance psg_seed for this call.
        """
        nodes_df = self.get_nodes(network_name)
        
        # Use provided seed or default to instance psg_seed
        psg_seed = seed if seed is not None else self.psg_seed
        
        psg = PoissonSpikeGenerator(population=network_name, seed=psg_seed)
        
        total_nodes = 0
        for pop_name, (mean, stdev) in shell_params.items():
            nodes_in_pop = nodes_df[nodes_df['pop_name'] == pop_name].index.values
            if len(nodes_in_pop) == 0:
                continue
            
            total_nodes += len(nodes_in_pop)
            
            # Generate rates using helper function
            rates = self._generate_firing_rates(len(nodes_in_pop), mean, stdev, distribution)
            
            # Add to PSG
            for node_id, rate in zip(nodes_in_pop, rates):
                psg.add(node_ids=node_id, firing_rate=rate, times=(t_start, t_stop))
        
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        psg.to_sonata(output_path)
        if verbose:
            print(f"Generated shell stimulus ({distribution}): {total_nodes} nodes to {output_path}")
