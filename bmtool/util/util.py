import argparse
from argparse import RawTextHelpFormatter,SUPPRESS
import glob, json, os, re, sys
import math
import numpy as np
from numpy import genfromtxt
import h5py
import pandas as pd

#from bmtk.utils.io.cell_vars import CellVarsFile
#from bmtk.analyzer.cell_vars import _get_cell_report
#from bmtk.analyzer.io_tools import load_config

def get_argparse(use_description):
    parser = argparse.ArgumentParser(description=use_description, formatter_class=RawTextHelpFormatter,usage=SUPPRESS)
    return parser
    
def verify_parse(parser):
    try:
        if not len(sys.argv) > 1:
            raise
        #if sys.argv[1] in ['-h','--h','-help','--help','help']:
        #    raise
        parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
        
use_description = """
BMTK model utilties.

python -m bmtool.util 
"""

if __name__ == '__main__':
    parser = get_argparse(use_description)
    verify_parse(parser)
    

class CellVarsFile(object):
    VAR_UNKNOWN = 'Unknown'
    UNITS_UNKNOWN = 'NA'

    def __init__(self, filename, mode='r', **params):
        
        import h5py
        self._h5_handle = h5py.File(filename, 'r')
        self._h5_root = self._h5_handle[params['h5_root']] if 'h5_root' in params else self._h5_handle['/']
        self._var_data = {}
        self._var_units = {}

        self._mapping = None

        # Look for variabl and mapping groups
        for var_name in self._h5_root.keys():
            hf_grp = self._h5_root[var_name]

            if var_name == 'data':
                # According to the sonata format the /data table should be located at the root
                var_name = self._h5_root['data'].attrs.get('variable_name', CellVarsFile.VAR_UNKNOWN)
                self._var_data[var_name] = self._h5_root['data']
                self._var_units[var_name] = self._find_units(self._h5_root['data'])

            if not isinstance(hf_grp, h5py.Group):
                continue

            if var_name == 'mapping':
                # Check for /mapping group
                self._mapping = hf_grp
            else:
                # In the bmtk we can support multiple variables in the same file (not sonata compliant but should be)
                # where each variable table is separated into its own group /<var_name>/data
                if 'data' not in hf_grp:
                    print('Warning: could not find "data" dataset in {}. Skipping!'.format(var_name))
                else:
                    self._var_data[var_name] = hf_grp['data']
                    self._var_units[var_name] = self._find_units(hf_grp['data'])

        # create map between gids and tables
        self._gid2data_table = {}
        if self._mapping is None:
            raise Exception('could not find /mapping group')
        else:
            gids_ds = self._mapping['gids']
            index_pointer_ds = self._mapping['index_pointer']
            for indx, gid in enumerate(gids_ds):
                self._gid2data_table[gid] = (index_pointer_ds[indx], index_pointer_ds[indx+1])  # slice(index_pointer_ds[indx], index_pointer_ds[indx+1])

            time_ds = self._mapping['time']
            self._t_start = time_ds[0]
            self._t_stop = time_ds[1]
            self._dt = time_ds[2]
            self._n_steps = int((self._t_stop - self._t_start) / self._dt)

    @property
    def variables(self):
        return list(self._var_data.keys())

    @property
    def gids(self):
        return list(self._gid2data_table.keys())

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    @property
    def dt(self):
        return self._dt

    @property
    def time_trace(self):
        return np.linspace(self.t_start, self.t_stop, num=self._n_steps, endpoint=True)

    @property
    def h5(self):
        return self._h5_root

    def _find_units(self, data_set):
        return data_set.attrs.get('units', CellVarsFile.UNITS_UNKNOWN)

    def units(self, var_name=VAR_UNKNOWN):
        return self._var_units[var_name]

    def n_compartments(self, gid):
        bounds = self._gid2data_table[gid]
        return bounds[1] - bounds[0]

    def compartment_ids(self, gid):
        bounds = self._gid2data_table[gid]
        return self._mapping['element_id'][bounds[0]:bounds[1]]

    def compartment_positions(self, gid):
        bounds = self._gid2data_table[gid]
        return self._mapping['element_pos'][bounds[0]:bounds[1]]

    def data(self, gid, var_name=VAR_UNKNOWN,time_window=None, compartments='origin'):
        if var_name not in self.variables:
            raise Exception('Unknown variable {}'.format(var_name))

        if time_window is None:
            time_slice = slice(0, self._n_steps)
        else:
            if len(time_window) != 2:
                raise Exception('Invalid time_window, expecting tuple [being, end].')

            window_beg = max(int((time_window[0] - self.t_start)/self.dt), 0)
            window_end = min(int((time_window[1] - self.t_start)/self.dt), self._n_steps/self.dt)
            time_slice = slice(window_beg, window_end)

        multi_compartments = True
        if compartments == 'origin' or self.n_compartments(gid) == 1:
            # Return the first (and possibly only) compartment for said gid
            gid_slice = self._gid2data_table[gid][0]
            multi_compartments = False
        elif compartments == 'all':
            # Return all compartments
            gid_slice = slice(self._gid2data_table[gid][0], self._gid2data_table[gid][1])
        else:
            # return all compartments with corresponding element id
            compartment_list = list(compartments) if isinstance(compartments, (long, int)) else compartments
            begin = self._gid2data_table[gid][0]
            end = self._gid2data_table[gid][1]
            gid_slice = [i for i in range(begin, end) if self._mapping[i] in compartment_list]

        data = np.array(self._var_data[var_name][time_slice, gid_slice])
        return data.T if multi_compartments else data
    
def load_config(config_file):
    import bmtk.simulator.core.simulation_config as config
    conf = config.from_json(config_file)
    #from bmtk.simulator import bionet
    #conf = bionet.Config.from_json(config_file, validate=True)
    return conf

def load_nodes_edges_from_config(fp):
    if fp is None:
        fp = 'simulation_config.json'
    config = load_config(fp)
    nodes = load_nodes_from_paths(config['networks']['nodes'])
    edges = load_edges_from_paths(config['networks']['edges'])
    return nodes, edges

def load_nodes(nodes_file, node_types_file):
    nodes_arr = [{"nodes_file":nodes_file,"node_types_file":node_types_file}]
    nodes = list(load_nodes_from_paths(nodes_arr).items())[0]  # single item
    return nodes  # return (population, nodes_df)

def load_nodes_from_config(config):
    if config is None:
        config = 'simulation_config.json'
    networks = load_config(config)['networks']
    return load_nodes_from_paths(networks['nodes'])

def load_nodes_from_paths(node_paths):
    """
        node_paths must be in the format in a circuit config file:
        [
            {
            "nodes_file":"filepath",
            "node_types_file":"filepath"
            },...
        ]
        #Glob all files for *_nodes.h5
        #Glob all files for *_edges.h5

        Returns a dictionary indexed by population, of pandas tables in the following format:
                 node_type_id   model_template morphology   model_type pop_name   pos_x   pos_y  pos_z
        node_id
        0                 100  hoc:IzhiCell_EC  blank.swc  biophysical       EC  1.5000  0.2500   10.0

        Where pop_name was a user defined cell property
    """
    import h5py
    import pandas as pd
    
    #nodes_regex = "_nodes.h5"
    #node_types_regex = "_node_types.csv"

    #nodes_h5_fpaths = glob.glob(os.path.join(network_dir,'*'+nodes_regex))
    #node_types_fpaths = glob.glob(os.path.join(network_dir,'*'+node_types_regex))

    #regions = [re.findall('^[^_]+', os.path.basename(n))[0] for n in nodes_h5_fpaths]
    region_dict = {}

    pos_labels = ('pos_x', 'pos_y', 'pos_z')

    #Need to get all cell groups for each region
    def get_node_table(cell_models_file, cells_file, population=None):
        cm_df = pd.read_csv(cells_file, sep=' ')
        cm_df.set_index('node_type_id', inplace=True)

        cells_h5 = h5py.File(cell_models_file, 'r')
        if population is None:
            if len(cells_h5['/nodes']) > 1:
                raise Exception('Multiple populations in nodes file. Not currently supported. Should be easy to implement when needed. Let Tyler know.')
            else:
                population = list(cells_h5['/nodes'])[0]

        nodes_grp = cells_h5['/nodes'][population]
        c_df = pd.DataFrame({key: nodes_grp[key] for key in ('node_id', 'node_type_id')})
        c_df.set_index('node_id', inplace=True)

        nodes_df = pd.merge(left=c_df, right=cm_df, how='left',
                            left_on='node_type_id', right_index=True)  # use 'model_id' key to merge, for right table the "model_id" is an index

        # extra properties of individual nodes (see SONATA Data format)
        if nodes_grp.get('0'):
            node_id = nodes_grp['node_id'][()]
            node_group_id = nodes_grp['node_group_id'][()]
            node_group_index = nodes_grp['node_group_index'][()]
            n_group = node_group_id.max() + 1
            prop_dtype = {}
            for group_id in range(n_group):
                group = nodes_grp[str(group_id)]
                idx = node_group_id == group_id
                group_node = node_id[idx]
                group_index = node_group_index[idx]
                for prop in group:
                    if prop == 'positions':
                        positions = group[prop][group_index]
                        for i in range(positions.shape[1]):
                            if pos_labels[i] not in nodes_df:
                                nodes_df[pos_labels[i]] = np.nan
                            nodes_df.loc[group_node, pos_labels[i]] = positions[:, i]
                    else:
                        # create new column with NaN if property does not exist
                        if prop not in nodes_df: 
                            nodes_df[prop] = np.nan
                        nodes_df.loc[group_node, prop] = group[prop][group_index]
                        prop_dtype[prop] = group[prop].dtype
            # convert to original data type if possible
            for prop, dtype in prop_dtype.items():
                nodes_df[prop] = nodes_df[prop].astype(dtype, errors='ignore')

        return population, nodes_df

    #for region, cell_models_file, cells_file in zip(regions, node_types_fpaths, nodes_h5_fpaths):
    #    region_dict[region] = get_node_table(cell_models_file,cells_file,population=region)
    for nodes in node_paths:
        cell_models_file = nodes["nodes_file"]
        cells_file = nodes["node_types_file"]
        region_name, region = get_node_table(cell_models_file, cells_file)
        region_dict[region_name] = region

    #cell_num = 2
    #print(region_dict["hippocampus"].iloc[cell_num]["node_type_id"])
    #print(list(set(region_dict["hippocampus"]["node_type_id"]))) #Unique

    return region_dict
    
def load_edges_from_config(config):
    if config is None:
        config = 'simulation_config.json'
    networks = load_config(config)['networks']
    return load_edges_from_paths(networks['edges'])

def load_edges(edges_file, edge_types_file):
    edges_arr = [{"edges_file":edges_file,"edge_types_file":edge_types_file}]
    edges = list(load_edges_from_paths(edges_arr).items())[0]  # single item
    return edges  # return (population, edges_df)

def load_edges_from_paths(edge_paths):#network_dir='network'):
    """
    Returns: A dictionary of connections with filenames (minus _edges.h5) as keys

    edge_paths must be in the format in a circuit config file:
        [
            {
            "edges_file":"filepath", (csv)
            "edge_types_file":"filepath" (h5)
            },...
        ]
    util.load_edges_from_paths([{"edges_file":"network/hippocampus_hippocampus_edges.h5","edge_types_file":"network/hippocampus_hippocampus_edge_types.csv"}])
    """
    import h5py
    import pandas as pd
    #edges_regex = "_edges.h5"
    #edge_types_regex = "_edge_types.csv"

    #edges_h5_fpaths = glob.glob(os.path.join(network_dir,'*'+edges_regex))
    #edge_types_fpaths = glob.glob(os.path.join(network_dir,'*'+edge_types_regex))

    #connections = [re.findall('^[A-Za-z0-9]+_[A-Za-z0-9][^_]+', os.path.basename(n))[0] for n in edges_h5_fpaths]
    edges_dict = {}
    def get_edge_table(edges_file, edge_types_file, population=None):

        # dataframe where each row is an edge type
        cm_df = pd.read_csv(edge_types_file, sep=' ')
        cm_df.set_index('edge_type_id', inplace=True)

        with h5py.File(edges_file, 'r') as connections_h5:
            if population is None:
                if len(connections_h5['/edges']) > 1:
                    raise Exception('Multiple populations in edges file. Not currently implemented, should not be hard to do, contact Tyler')
                else:
                    population = list(connections_h5['/edges'])[0]
            conn_grp = connections_h5['/edges'][population]

            # dataframe where each row is an edge
            c_df = pd.DataFrame({key: conn_grp[key] for key in (
                'edge_type_id', 'source_node_id', 'target_node_id')})

            c_df.reset_index(inplace=True)
            c_df.rename(columns={'index': 'edge_id'}, inplace=True)
            c_df.set_index('edge_type_id', inplace=True)

            # add edge type properties to df of edges
            edges_df = pd.merge(left=c_df, right=cm_df, how='left',
                                left_index=True, right_index=True)

            # extra properties of individual edges (see SONATA Data format)
            if conn_grp.get('0'):
                edge_group_id = conn_grp['edge_group_id'][()]
                edge_group_index = conn_grp['edge_group_index'][()]
                n_group = edge_group_id.max() + 1
                prop_dtype = {}
                for group_id in range(n_group):
                    group = conn_grp[str(group_id)]
                    idx = edge_group_id == group_id
                    for prop in group:
                        # create new column with NaN if property does not exist
                        if prop not in edges_df: 
                            edges_df[prop] = np.nan
                        edges_df.loc[idx, prop] = tuple(group[prop][edge_group_index[idx]])
                        prop_dtype[prop] = group[prop].dtype
                # convert to original data type if possible
                for prop, dtype in prop_dtype.items():
                    edges_df[prop] = edges_df[prop].astype(dtype, errors='ignore')

        return population, edges_df

    #for edges_dict, conn_models_file, conns_file in zip(connections, edge_types_fpaths, edges_h5_fpaths):
    #    connections_dict[connection] = get_connection_table(conn_models_file,conns_file)
    try:
        for nodes in edge_paths:
            edges_file = nodes["edges_file"]
            edge_types_file = nodes["edge_types_file"]
            region_name, region = get_edge_table(edges_file, edge_types_file)
            edges_dict[region_name] = region
    except Exception as e:
        print(repr(e))
        print("Hint: Are you loading the correct simulation config file?")
        print("Command Line: bmtool plot --config yourconfig.json <rest of command>")
        print("Python: bmplot.connection_matrix(config='yourconfig.json')")
    
    return edges_dict

def cell_positions_by_id(config=None, nodes=None, populations=[], popids=[], prepend_pop=True):
    """
    Returns a dictionary of arrays of arrays {"population_popid":[[1,2,3],[1,2,4]],...
    """
    if not nodes:
        nodes = load_nodes_from_config(config)

    import pdb

    if 'all' in populations or not populations or not len(populations):
        populations = list(nodes)

    popids += (len(populations)-len(popids)) * ["node_type_id"] #Extend the array to default values if not enough given
    cells_by_id = {}
    for population,pid in zip(populations,popids):
        #get a list of unique cell types based on pid
        pdb.set_trace()
        cell_types = list(nodes[population][str(pid)].unique())
        for ct in cell_types:
            cells_by_id[population+'_'+ct] = 0
        
    return cells_by_id

def relation_matrix(config=None, nodes=None,edges=None,sources=[],targets=[],sids=[],tids=[],prepend_pop=True,relation_func=None,return_type=float,drop_point_process=False,synaptic_info='0'):
     
    import pandas as pd
    
    if not nodes and not edges:
        nodes,edges = load_nodes_edges_from_config(config)
    if not nodes:
        nodes = load_nodes_from_config(config)
    if not edges:
        edges = load_edges_from_config(config)
    if not edges and not nodes and not config:
        raise Exception("No information given to load nodes/edges")
    
    if 'all' in sources:
        sources = list(nodes)
    if 'all' in targets:
        targets = list(nodes)
    sids += (len(sources)-len(sids)) * ["node_type_id"] #Extend the array to default values if not enough given
    tids += (len(targets)-len(tids)) * ["node_type_id"]

    total_source_cell_types = 0
    total_target_cell_types = 0
    source_uids = []
    target_uids = []
    source_pop_names = []
    target_pop_names = []
    source_totals = []
    target_totals = []

    source_map = {}#Sometimes we don't add the item to sources or targets, need to keep track of the index
    target_map = {}#Or change to be a dictionary sometime

    for source,sid in zip(sources,sids):
        do_process = False
        for t, target in enumerate(targets):
            e_name = source+"_to_"+target
            if e_name in list(edges):
                do_process=True
        if not do_process: # This is not seen as an input, don't process it.
            continue
        
        if drop_point_process:
            nodes_src = pd.DataFrame(nodes[source][nodes[source]['model_type']!='point_process'])
        else:
            nodes_src = pd.DataFrame(nodes[source])
        total_source_cell_types = total_source_cell_types + len(list(set(nodes_src[sid])))
        unique_ = nodes_src[sid].unique()
        source_uids.append(unique_)
        prepend_str = ""
        if prepend_pop:
            prepend_str = str(source) +"_"
        unique_= list(np.array((prepend_str+ pd.DataFrame(unique_).astype(str)).values.tolist()).ravel())
        source_pop_names = source_pop_names + unique_
        source_totals.append(len(unique_))
        source_map[source] = len(source_uids)-1
    for target,tid in zip(targets,tids):
        do_process = False
        for s, source in enumerate(sources):
            e_name = source+"_to_"+target
            if e_name in list(edges):
                do_process=True
        if not do_process:
            continue

        if drop_point_process:
            nodes_trg = pd.DataFrame(nodes[target][nodes[target]['model_type']!='point_process'])
        else:
            nodes_trg = pd.DataFrame(nodes[target])

        total_target_cell_types = total_target_cell_types + len(list(set(nodes_trg[tid])))
        
        unique_ = nodes_trg[tid].unique()
        target_uids.append(unique_)
        prepend_str = ""
        if prepend_pop:
            prepend_str = str(target) +"_"
        unique_ = list(np.array((prepend_str + pd.DataFrame(unique_).astype(str)).values.tolist()).ravel())
        target_pop_names = target_pop_names + unique_
        target_totals.append(len(unique_))
        target_map[target] = len(target_uids) -1

    e_matrix = np.zeros((total_source_cell_types,total_target_cell_types),dtype=return_type)
    syn_info = np.zeros((total_source_cell_types,total_target_cell_types),dtype=object)
    sources_start =  np.cumsum(source_totals) -source_totals
    target_start = np.cumsum(target_totals) -target_totals
    total = 0
    stdev=0
    mean=0
    for s, source in enumerate(sources):
        for t, target in enumerate(targets):
            e_name = source+"_to_"+target
            if e_name not in list(edges):
                continue
            if relation_func:
                source_nodes = nodes[source].add_prefix('source_')
                target_nodes = nodes[target].add_prefix('target_')

                c_edges = pd.merge(left=edges[e_name],
                            right=source_nodes,
                            how='left',
                            left_on='source_node_id',
                            right_index=True)

                c_edges = pd.merge(left=c_edges,
                            right=target_nodes,
                            how='left',
                            left_on='target_node_id',
                            right_index=True)
                
                sm = source_map[source]
                tm = target_map[target]
                
                def syn_info_func(**kwargs):
                    edges = kwargs["edges"]
                    source_id_type = kwargs["sid"]
                    target_id_type = kwargs["tid"]
                    source_id = kwargs["source_id"]
                    target_id = kwargs["target_id"]
                    if edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]["dynamics_params"].count()!=0:
                        params = str(edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]["dynamics_params"].drop_duplicates().values[0])
                        params = params[:-5]
                        mod = str(edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]["model_template"].drop_duplicates().values[0])
                        if mod and synaptic_info=='1':
                            return mod
                        elif params and synaptic_info=='2':
                            return params
                        else:
                            return None

                def conn_mean_func(**kwargs):
                    edges = kwargs["edges"]
                    source_id_type = kwargs["sid"]
                    target_id_type = kwargs["tid"]
                    source_id = kwargs["source_id"]
                    target_id = kwargs["target_id"]
                    mean = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]['target_node_id'].value_counts().mean()
                    return mean

                def conn_stdev_func(**kwargs):
                    edges = kwargs["edges"]
                    source_id_type = kwargs["sid"]
                    target_id_type = kwargs["tid"]
                    source_id = kwargs["source_id"]
                    target_id = kwargs["target_id"]
                    stdev = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]['target_node_id'].value_counts().std()
                    return stdev

                for s_type_ind,s_type in enumerate(source_uids[sm]):
            
                    for t_type_ind,t_type in enumerate(target_uids[tm]): 
                        source_index = int(s_type_ind+sources_start[sm])
                        target_index = int(t_type_ind+target_start[tm])
                
                        total = relation_func(source_nodes=source_nodes, target_nodes=target_nodes, edges=c_edges, source=source,sid="source_"+sids[s], target=target,tid="target_"+tids[t],source_id=s_type,target_id=t_type)
                        if synaptic_info=='0':
                            if isinstance(total, tuple):
                                syn_info[source_index, target_index] = str(round(total[0], 1)) + '\n' + str(round(total[1], 1))
                            else:
                                syn_info[source_index,target_index] = total
                        elif synaptic_info=='1':
                            mean = conn_mean_func(source_nodes=source_nodes, target_nodes=target_nodes, edges=c_edges, source=source,sid="source_"+sids[s], target=target,tid="target_"+tids[t],source_id=s_type,target_id=t_type)
                            stdev = conn_stdev_func(source_nodes=source_nodes, target_nodes=target_nodes, edges=c_edges, source=source,sid="source_"+sids[s], target=target,tid="target_"+tids[t],source_id=s_type,target_id=t_type)
                            if math.isnan(mean):
                                mean=0
                            if math.isnan(stdev):
                                stdev=0 
                            syn_info[source_index,target_index] = str(round(mean,1)) + '\n'+ str(round(stdev,1))
                        elif synaptic_info=='2':
                            syn_list = syn_info_func(source_nodes=source_nodes, target_nodes=target_nodes, edges=c_edges, source=source,sid="source_"+sids[s], target=target,tid="target_"+tids[t],source_id=s_type,target_id=t_type)
                            if syn_list is None:
                                syn_info[source_index,target_index] = ""
                            else:
                                syn_info[source_index,target_index] = syn_list
                        elif synaptic_info=='3':
                            syn_list = syn_info_func(source_nodes=source_nodes, target_nodes=target_nodes, edges=c_edges, source=source,sid="source_"+sids[s], target=target,tid="target_"+tids[t],source_id=s_type,target_id=t_type)
                            if syn_list is None:
                                syn_info[source_index,target_index] = ""
                            else:
                                syn_info[source_index,target_index] = syn_list
                        if isinstance(total, tuple):
                            e_matrix[source_index,target_index]=total[0]
                        else:
                            e_matrix[source_index,target_index]=total

                                                
    return syn_info, e_matrix, source_pop_names, target_pop_names

def connection_totals(config=None,nodes=None,edges=None,sources=[],targets=[],sids=[],tids=[],prepend_pop=True,synaptic_info='0',include_gap=True):
    
    def total_connection_relationship(**kwargs):
        edges = kwargs["edges"]
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]

        total = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]
        if include_gap == False:
            try: 
                cons = cons[cons['is_gap_junction'] != True]
            except:
                raise Exception("no gap junctions found to drop from connections")
            
        total = total.count()
        total = total.source_node_id # may not be the best way to pick
        return total
    return relation_matrix(config,nodes,edges,sources,targets,sids,tids,prepend_pop,relation_func=total_connection_relationship,synaptic_info=synaptic_info)


def percent_connections(config=None,nodes=None,edges=None,sources=[],targets=[],sids=[],tids=[],prepend_pop=True,type='convergence',method=None,include_gap=True):


    def precent_func(**kwargs): 
        edges = kwargs["edges"]
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]
        t_list = kwargs["target_nodes"]
        s_list = kwargs["source_nodes"]

        cons = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]
        if include_gap == False:
            try: 
                cons = cons[cons['is_gap_junction'] != True]
            except:
                raise Exception("no gap junctions found to drop from connections")
            
        total_cons = cons.count().source_node_id
        # to determine reciprocal connectivity
        # create a copy and flip source/dest
        cons_flip = edges[(edges[source_id_type] == target_id) & (edges[target_id_type]==source_id)]
        cons_flip = cons_flip.rename(columns={'source_node_id':'target_node_id','target_node_id':'source_node_id'})
        # append to original 
        cons_recip = pd.concat([cons, cons_flip])

        # determine dropped duplicates (keep=False)
        cons_recip_dedup = cons_recip.drop_duplicates(subset=['source_node_id','target_node_id'])

        # note counts
        num_bi = (cons_recip.count().source_node_id - cons_recip_dedup.count().source_node_id)
        num_uni = total_cons - num_bi    

        #num_sources = s_list.apply(pd.Series.value_counts)[source_id_type].dropna().sort_index().loc[source_id]
        #num_targets = t_list.apply(pd.Series.value_counts)[target_id_type].dropna().sort_index().loc[target_id]

        num_sources = s_list[source_id_type].value_counts().sort_index().loc[source_id]
        num_targets = t_list[target_id_type].value_counts().sort_index().loc[target_id]


        total = round(total_cons / (num_sources*num_targets) * 100,2)
        uni = round(num_uni / (num_sources*num_targets) * 100,2)
        bi = round(num_bi / (num_sources*num_targets) * 100,2)
        if method == 'total':
            return total
        if method == 'uni':
            return uni
        if method == 'bi':
            return bi


    return relation_matrix(config,nodes,edges,sources,targets,sids,tids,prepend_pop,relation_func=precent_func)


def connection_divergence(config=None,nodes=None,edges=None,sources=[],targets=[],sids=[],tids=[],prepend_pop=True,convergence=False,method='mean+std',include_gap=True):

    import pandas as pd

    def total_connection_relationship(**kwargs):
        edges = kwargs["edges"]
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]
        t_list = kwargs["target_nodes"]
        s_list = kwargs["source_nodes"]
        count = 1

        cons = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]
        if include_gap == False:
            try: 
                cons = cons[cons['is_gap_junction'] != True]
            except:
                raise Exception("no gap junctions found to drop from connections")

        if convergence:
            if method == 'min':
                count = cons['target_node_id'].value_counts().min()
                return round(count,2)
            elif method == 'max':
                count = cons['target_node_id'].value_counts().max()
                return round(count,2)
            elif method == 'std':
                std = cons['target_node_id'].value_counts().std()
                return round(std,2)
            elif method == 'mean': 
                mean = cons['target_node_id'].value_counts().mean()
                return round(mean,2)
            elif method == 'mean+std': #default is mean + std
                mean = cons['target_node_id'].value_counts().mean()
                std = cons['target_node_id'].value_counts().std()
                #std = cons.apply(pd.Series.value_counts).target_node_id.dropna().std() no longer a valid way
                return (round(mean,2)), (round(std,2))
        else: #divergence
            if method == 'min':
                count = cons['source_node_id'].value_counts().min()
                return round(count,2)
            elif method == 'max':
                count = cons['source_node_id'].value_counts().max()
                return round(count,2)
            elif method == 'std':
                std = cons['source_node_id'].value_counts().std()
                return round(std,2)
            elif method == 'mean': 
                mean = cons['source_node_id'].value_counts().mean()
                return round(mean,2)
            elif method == 'mean+std': #default is mean + std
                mean = cons['source_node_id'].value_counts().mean()
                std = cons['source_node_id'].value_counts().std()
                return (round(mean,2)), (round(std,2))

    return relation_matrix(config,nodes,edges,sources,targets,sids,tids,prepend_pop,relation_func=total_connection_relationship)

def gap_junction_connections(config=None,nodes=None,edges=None,sources=[],targets=[],sids=[],tids=[],prepend_pop=True,type='convergence'):
    import pandas as pd

    
    def total_connection_relationship(**kwargs): #reduced version of original function; only gets mean+std
        edges = kwargs["edges"]
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]

        cons = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)] 
        #print(cons)
        
        try: 
            cons = cons[cons['is_gap_junction'] != True]
        except:
            raise Exception("no gap junctions found to drop from connections")
        mean = cons['target_node_id'].value_counts().mean()
        std = cons['target_node_id'].value_counts().std()
        return (round(mean,2)), (round(std,2))
    
    def precent_func(**kwargs): #barely different than original function; only gets gap_junctions.
        edges = kwargs["edges"]
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]
        t_list = kwargs["target_nodes"]
        s_list = kwargs["source_nodes"]

        cons = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]
        #add functionality that shows only the one's with gap_junctions
        try: 
            cons = cons[cons['is_gap_junction'] != True]
        except:
            raise Exception("no gap junctions found to drop from connections")
        
        total_cons = cons.count().source_node_id

        num_sources = s_list[source_id_type].value_counts().sort_index().loc[source_id]
        num_targets = t_list[target_id_type].value_counts().sort_index().loc[target_id]


        total = round(total_cons / (num_sources*num_targets) * 100,2)
        return total
    
    if type == 'convergence':
        return relation_matrix(config,nodes,edges,sources,targets,sids,tids,prepend_pop,relation_func=total_connection_relationship)
    elif type == 'percent':
        return relation_matrix(config,nodes,edges,sources,targets,sids,tids,prepend_pop,relation_func=precent_func)
        

def gap_junction_percent_connections(config=None,nodes=None,edges=None,sources=[],targets=[],sids=[],tids=[],prepend_pop=True,method=None):
    import pandas as pd
    
        
    
def connection_probabilities(config=None,nodes=None,edges=None,sources=[],
    targets=[],sids=[],tids=[],prepend_pop=True,dist_X=True,dist_Y=True,dist_Z=True,num_bins=10,include_gap=True):
    
    import pandas as pd
    from scipy.spatial import distance
    import matplotlib.pyplot as plt
    pd.options.mode.chained_assignment = None

    def connection_relationship(**kwargs):
        edges = kwargs["edges"]
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]
        t_list = kwargs["target_nodes"]
        s_list = kwargs["source_nodes"]

        
        """
        count = 1

        if convergence:
            vc = t_list.apply(pd.Series.value_counts)
            vc = vc[target_id_type].dropna().sort_index()
            count = vc.ix[target_id]#t_list[t_list[target_id_type]==target_id]
        else:
            vc = s_list.apply(pd.Series.value_counts)
            vc = vc[source_id_type].dropna().sort_index()
            count = vc.ix[source_id]#count = s_list[s_list[source_id_type]==source_id]

        total = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)].count()
        total = total.source_node_id # may not be the best way to pick
        return round(total/count,1)
        """
        
        def eudist(df,use_x=True,use_y=True,use_z=True):
            def _dist(x):
                if len(x) == 6:
                    return distance.euclidean((x.iloc[0], x.iloc[1], x.iloc[2]), (x.iloc[3], x.iloc[4], x.iloc[5]))
                elif len(x) == 4:
                    return distance.euclidean((x.iloc[0],x.iloc[1]),(x.iloc[2],x.iloc[3]))
                elif len(x) == 2:
                    return distance.euclidean((x.iloc[0]),(x.iloc[1]))
                else:
                    return -1

            if use_x and use_y and use_z: #(XYZ)
                cols = ['source_pos_x','source_pos_y','source_pos_z',
                    'target_pos_x','target_pos_y','target_pos_z']
            elif use_x and use_y and not use_z: #(XY)
                cols = ['source_pos_x','source_pos_y',
                    'target_pos_x','target_pos_y',]
            elif use_x and not use_y and use_z: #(XZ)
                cols = ['source_pos_x','source_pos_z',
                    'target_pos_x','target_pos_z']
            elif not use_x and use_y and use_z: #(YZ)
                cols = ['source_pos_y','source_pos_z',
                    'target_pos_y','target_pos_z']
            elif use_x and not use_y and not use_z: #(X)
                cols = ['source_pos_x','target_pos_x']
            elif not use_x and use_y and not use_z: #(Y)
                cols = ['source_pos_y','target_pos_y']
            elif not use_x and not use_y and use_z: #(Z)
                cols = ['source_pos_z','target_pos_z']
            else:
                cols = []

            if ('source_pos_x' in df and 'target_pos_x' in df) or ('source_pos_y' in df and 'target_pos_y' in df) or ('source_pos_' in df and 'target_pos_z' in df):
                ret = df.loc[:,cols].apply(_dist,axis=1)
            else:
                print('No x, y, or z positions defined')
                ret=np.zeros(1)
            
            return ret

        relevant_edges = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]
        if include_gap == False:
            try: 
                relevant_edges = relevant_edges[relevant_edges['is_gap_junction'] != True]
            except:
                raise Exception("no gap junctions found to drop from connections")
        connected_distances = eudist(relevant_edges,dist_X,dist_Y,dist_Z).values.tolist()
        if len(connected_distances)>0:
            if connected_distances[0]==0:
                return -1
        sl = s_list[s_list[source_id_type]==source_id]
        tl = t_list[t_list[target_id_type]==target_id]
        
        target_rows = ["target_pos_x","target_pos_y","target_pos_z"]
        
        all_distances = []
        for target in tl.iterrows():
            target = target[1]
            for new_col in target_rows:
                sl[new_col] = target[new_col]
            #sl[target_rows] = target.loc[target_rows].tolist()
            row_distances = eudist(sl,dist_X,dist_Y,dist_Z).tolist()
            all_distances = all_distances + row_distances
        plt.ioff()
        ns,bins,patches = plt.hist([connected_distances,all_distances],density=False,histtype='stepfilled',bins=num_bins)
        plt.ion()
        return {"ns":ns,"bins":bins}
        #import pdb;pdb.set_trace()
        # edges contains all edges

    return relation_matrix(config,nodes,edges,sources,targets,sids,tids,prepend_pop,relation_func=connection_relationship,return_type=object,drop_point_process=True)


def connection_graph_edge_types(config=None,nodes=None,edges=None,sources=[],targets=[],sids=[],tids=[],prepend_pop=True,edge_property='model_template'):

    def synapse_type_relationship(**kwargs):
        edges = kwargs["edges"]
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]

        connections = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]
        
        return list(connections[edge_property].unique())

    return relation_matrix(config,nodes,edges,sources,targets,sids,tids,prepend_pop,relation_func=synapse_type_relationship,return_type=object)


def edge_property_matrix(edge_property, config=None, nodes=None, edges=None, sources=[],targets=[],sids=[],tids=[],prepend_pop=True,report=None,time=-1,time_compare=None):
    
    var_report = None
    if time>=0 and report:
        cfg = load_config(config)
        #report_full, report_file = _get_cell_report(config,report)
        report_file = report # Same difference
        var_report = EdgeVarsFile(os.path.join(cfg['output']['output_dir'],report_file+'.h5'))

    def weight_hist_relationship(**kwargs):
        edges = kwargs["edges"]
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]

        connections = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]
        nonlocal time, report, var_report
        ret = []

        if time>=0 and report:
            sources = list(connections['source_node_id'].unique())
            sources.sort()
            targets = list(connections['target_node_id'].unique())
            targets.sort() 
                
            data,sources,targets = get_synapse_vars(None,None,edge_property,targets,source_gids=sources,compartments='all',var_report=var_report,time=time,time_compare=time_compare)
            if len(data.shape) and data.shape[0]!=0:
                ret = data[:,0]
            else:
                ret = []
        else:
            #if connections.get(edge_property) is not None: #Maybe we should fail if we can't find the variable...
            ret = list(connections[edge_property])

        return ret

    return relation_matrix(config,nodes,edges,sources,targets,sids,tids,prepend_pop,relation_func=weight_hist_relationship,return_type=object)


def percent_connectivity(config=None,nodes=None,edges=None,sources=[],targets=[],sids=[],tids=[],prepend_pop=True):
    
    import pandas as pd
    
    if not nodes and not edges:
        nodes,edges = load_nodes_edges_from_config(config)
    if not nodes:
        nodes = load_nodes_from_config(config)
    if not edges:
        edges = load_edges_from_config(config)
    if not edges and not nodes and not config:
        raise Exception("No information given to load nodes/edges")

    data, source_labels, target_labels = connection_totals(config=config,nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=prepend_pop)

    #total_cell_types = len(list(set(nodes[populations[0]]["node_type_id"])))
    vc = nodes[sources[0]].apply(pd.Series.value_counts)
    vc = vc["node_type_id"].dropna().sort_index()
    vc = list(vc)

    max_connect = np.ones((len(vc),len(vc)),dtype=np.float)

    for a, i in enumerate(vc):
        for b, j in enumerate(vc):
            max_connect[a,b] = i*j
    ret = data/max_connect
    ret = ret*100
    ret = np.around(ret, decimals=1)

    return ret, source_labels, target_labels
    

def connection_average_synapses():
    return


def connection_divergence_average_old(config=None, nodes=None, edges=None,populations=[],convergence=False):
    """
    For each cell in source count # of connections in target and average
    """

    import pandas as pd
    
    if not nodes and not edges:
        nodes,edges = load_nodes_edges_from_config(config)
    if not nodes:
        nodes = load_nodes_from_config(config)
    if not edges:
        edges = load_edges_from_config(config)
    if not edges and not nodes and not config:
        raise Exception("No information given to load nodes/edges")

    nodes_hip = pd.DataFrame(nodes[populations[0]])
    pop_names = nodes_hip.pop_name.unique()

    nodes = nodes[list(nodes)[1]]
    edges = edges[list(edges)[1]]
    pdb.set_trace()
    src_df = pd.DataFrame({'edge_node_id': nodes.index,'source_node_pop_name':nodes['pop_name'],'source_node_type_id':nodes['node_type_id']})
    tgt_df = pd.DataFrame({'edge_node_id': nodes.index,'target_node_pop_name':nodes['pop_name'],'target_node_type_id':nodes['node_type_id']})
    
    src_df.set_index('edge_node_id', inplace=True)
    tgt_df.set_index('edge_node_id', inplace=True)
    
    edges_df = pd.merge(left=edges,
                            right=src_df,
                            how='left',
                            left_on='source_node_id',
                            right_index=True)
    
    edges_df = pd.merge(left=edges_df,
                            right=tgt_df,
                            how='left',
                            left_on='target_node_id',
                            right_index=True)
    
    edges_df_trim = edges_df.drop(edges_df.columns.difference(['source_node_type_id','target_node_type_id','source_node_pop_name','target_node_pop_name']), 1, inplace=False)

    vc = nodes.apply(pd.Series.value_counts)
    vc = vc["node_type_id"].dropna().sort_index()
    vc = list(vc)

    """
    For each source type
        For each target type
            temp = df[(df.src == source) & (df.tgt < target)]
            #edge_totals[src,tgt] = temp.sum
    """
    src_list_node_types = list(set(edges_df_trim["source_node_type_id"]))
    tgt_list_node_types = list(set(edges_df_trim["target_node_type_id"]))
    node_types = list(set(src_list_node_types+tgt_list_node_types))

    e_matrix = np.zeros((len(node_types),len(node_types)))

    for a, i in enumerate(node_types):
        for b, j in enumerate(node_types):
            num_conns = edges_df_trim[(edges_df_trim.source_node_type_id == i) & (edges_df_trim.target_node_type_id==j)].count()
            c = b if convergence else a #Show convergence if set. By dividing by targe totals instead of source

            e_matrix[a,b] = num_conns.source_node_type_id/vc[c]

    ret = np.around(e_matrix, decimals=1)

    return ret, pop_names


class EdgeVarsFile(CellVarsFile):
    def __init__(self, filename, mode='r', **params):
        super().__init__(filename, mode, **params)
        self._var_src_ids = []
        self._var_trg_ids = []
        for var_name in self._h5_root['mapping'].keys():
            if var_name == 'src_ids':
                self._var_src_ids = list(self._h5_root['mapping']['src_ids'])
            if var_name == 'trg_ids':
                self._var_trg_ids = list(self._h5_root['mapping']['trg_ids'])
    def sources(self,target_gid=None):
        if target_gid:
            tb = self._gid2data_table[target_gid]
            return self._h5_root['mapping']['src_ids'][tb[0]:tb[1]]
        else:
            return self._var_src_ids
    def targets(self):
        return self._var_trg_ids
    def data(self,gid,var_name=CellVarsFile.VAR_UNKNOWN,time_window=None,compartments='origin',sources=None):
        d = super().data(gid,var_name,time_window,compartments)
        if not sources:
            return d
        else:
            if type(sources) is int:
                sources = [sources]
            d_new = None
            for dl, s in zip(d, self.sources()):
                if s in sources:
                    if d_new is None:
                        d_new = np.array([dl])
                    else:
                        d_new = np.append(d_new, [dl],axis=0)
            if d_new is None:
                d_new = np.array([])            
            return d_new


def get_synapse_vars(config,report,var_name,target_gids,source_gids=None,compartments='all',var_report=None,time=None,time_compare=None):
    """
    Ex: data, sources = get_synapse_vars('9999_simulation_config.json', 'syn_report', 'W_ampa', 31)
    """
    if not var_report:
        cfg = load_config(config)
        #report, report_file = _get_cell_report(config,report)
        report_file = report # Same difference
        var_report = EdgeVarsFile(os.path.join(cfg['output']['output_dir'],report_file+'.h5'))

    if type(target_gids) is int:
        target_gids = [target_gids]
    
    data_ret = None
    sources_ret = None
    targets_ret = None
    
    for target_gid in target_gids:
        if not var_report._gid2data_table.get(target_gid):#This cell was not reported
            continue
        data = var_report.data(gid=target_gid, var_name=var_name, compartments=compartments)
        if(len(data.shape)==1):
            data = data.reshape(1,-1)

        if time is not None and time_compare is not None:
            data = np.array(data[:,time_compare] - data[:,time]).reshape(-1,1)
        elif time is not None:
            data = np.delete(data,np.s_[time+1:],1)
            data = np.delete(data,np.s_[:time],1)

        sources = var_report.sources(target_gid=target_gid)
        if source_gids:
            if type(source_gids) is int:
                source_gids = [source_gids]
            data = [d for d,s in zip(data,sources) if s in source_gids]
            sources = [s for s in sources if s in source_gids]
            
        targets = np.zeros(len(sources))
        targets.fill(target_gid)

        if data_ret is None or data_ret is not None and len(data_ret)==0:
            data_ret = data
        else:
            data_ret = np.append(data_ret, data,axis=0)
        if sources_ret is None or sources_ret is not None and len(sources_ret)==0:
            sources_ret = sources
        else:
            sources_ret = np.append(sources_ret, sources,axis=0)
        if targets_ret is None or targets_ret is not None and len(targets_ret)==0:
            targets_ret = targets
        else:
            targets_ret = np.append(targets_ret, targets,axis=0)

    return np.array(data_ret), np.array(sources_ret), np.array(targets_ret)


def tk_email_input(title="Send Model Files (with simplified GUI)",prompt="Enter your email address. (CHECK YOUR SPAM FOLDER)"):
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    # the input dialog
    user_inp = simpledialog.askstring(title=title, prompt=prompt)
    return user_inp

def popupmsg(msg):
    import tkinter as tk
    from tkinter import ttk
    popup = tk.Tk()
    popup.wm_title("!")
    NORM_FONT = ("Helvetica", 10)
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate


def send_mail(send_from, send_to, subject, text, files=None,server="127.0.0.1"):
    assert isinstance(send_to, list)
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject
    msg.attach(MIMEText(text))
    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)
    smtp = smtplib.SMTP(server)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()

def load_csv(csvfile):
    # TODO: make the separator more flexible
    if isinstance(csvfile, pd.DataFrame):
        return csvfile

    # TODO: check if it is csv object and convert to a pd dataframe
    return pd.read_csv(csvfile, sep=' ', na_values='NONE')

# The following code was developed by Matthew Stroud 7/15/21: Neural engineering group supervisor: Satish Nair
# This is an extension of bmtool: a development of Tyler Banks. 
# These are helper functions for I_clamps and spike train information.

def load_I_clamp_from_paths(Iclamp_paths):
    # Get info from .h5 files
    if Iclamp_paths.endswith('.h5'):
        f = h5py.File(Iclamp_paths, 'r')
        if 'amplitudes' in f and 'dts' in f and 'gids' in f:
            [amplitudes]=f['amplitudes'][:].tolist()
            dts=list(f['dts'])
            dts=dts[0]
            dset=f['gids']
            gids = dset[()]
            if gids == 'all':
                gids=' All biophysical cells'
            clamp=[amplitudes,dts,gids]
        else:
            raise Exception('.h5 file is not in the format "amplitudes","dts","gids". Cannot parse.')
    else:
        raise Exception('Input file is not of type .h5. Cannot parse.')
    return clamp

def load_I_clamp_from_config(fp):
    if fp is None:
        fp = 'config.json'
    config = load_config(fp)
    inputs=config['inputs']
    # Load in all current clamps
    ICLAMPS=[]
    for i in inputs:
        if inputs[i]['input_type']=="current_clamp":
            I_clamp=inputs[i]
            # Get current clamp info where an input file is provided
            if 'input_file' in I_clamp:
                ICLAMPS.append(load_I_clamp_from_paths(I_clamp['input_file']))
            # Get current clamp info when provided in "amp", "delay", "duration" format
            elif 'amp' in I_clamp and 'delay' in I_clamp and 'duration' in I_clamp:
                # Get simulation info from config
                run=config['run']
                dts=run['dt']
                if 'tstart' in run:
                    tstart=run['tstart']
                else:
                    tstart=0
                tstop=run['tstop']
                simlength=tstop-tstart
                nstep=int(simlength/dts)
                # Get input info from config
                amp=I_clamp['amp']
                gids=I_clamp['node_set']
                delay=I_clamp['delay']
                duration=I_clamp['duration']
                # Create a list with amplitude at each time step in the simulation
                amplitude=[]
                for i in range(nstep):
                    if i*dts>=delay and i*dts<=delay+duration:
                        amplitude.append(amp)
                    else:
                        amplitude.append(0)
                ICLAMPS.append([amplitude,dts,gids])
            else:
                raise Exception('No information found about this current clamp.')
    return ICLAMPS

def load_inspikes_from_paths(inspike_paths):
    # Get info from .h5 files
    if inspike_paths.endswith('.h5'):
        f = h5py.File(inspike_paths, 'r')
        # This is assuming that the first object in the file is named 'spikes'
        spikes=f['spikes']
        for i in spikes:
            inp=spikes[i]
            if 'node_ids' in inp and 'timestamps' in inp:
                node_ids=list(inp['node_ids'])
                timestamps=list(inp['timestamps'])
        data=[]        
        for j in range(len(node_ids)):
            data.append([str(timestamps[j]),str(node_ids[j]),''])
        data=np.array(data, dtype=object)
    elif inspike_paths.endswith('.csv'):
        # Loads in .csv and if it is of the form (timestamps node_ids population) it skips the conditionals.
        data=np.loadtxt(open(inspike_paths, 'r'), delimiter=" ",dtype=object,skiprows=1)
        if len(data[0])==2:   #This assumes gid in first column and spike times comma separated in second column
            temp=[]
            for i in data:
                timestamps=i[1]
                timestamps=timestamps.split(',')
                for j in timestamps:
                    temp.append([j,i[0],''])
            data=np.array(temp, dtype=object)
        #If the .csv is not in the form (timestamps node_ids population) or (gid timestamps)
        elif not len(data[0])==3:   
            print('The .csv spike file '+ inspike_paths +' is not in the correct format')
            return
    else:
        raise Exception('Input file is not of type .h5 or .csv. Cannot parse.')
    return data

def load_inspikes_from_config(fp):
    if fp is None:
        fp = 'config.json'
    config = load_config(fp)
    inputs=config['inputs']
    # Load in all current clamps
    INSPIKES=[]
    for i in inputs:
        if inputs[i]['input_type']=="spikes":
            INPUT=inputs[i]
            # Get current clamp info where an input file is provided
            if 'input_file' in INPUT:
                INSPIKES.append(load_inspikes_from_paths(INPUT['input_file']))
            else:
                raise Exception('No information found about this current clamp.')
    return INSPIKES
