import argparse
from argparse import RawTextHelpFormatter,SUPPRESS
import glob, json, os, re, sys

import numpy as np
from numpy import genfromtxt
import pandas as pd

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

python -m bmtools.util 
"""

if __name__ == '__main__':
    parser = get_argparse(use_description)
    verify_parse(parser)
    
    
def load_config(fp):
    data = None
    with open(fp) as f:
        data = json.load(f)
    return data
    

def nodes_edges_from_config(fp):
    #nodes = load_nodes_from_config(fp)
    #edges = load_nodes_from_config(fp)
    return None, None

def load_nodes(nodes_file, node_types_file):
    #nodes_arr = [{"nodes_file":nodes_file,"node_types_file":node_types_file}]
    #nodes = load_nodes_from_paths(nodes_arr)
    return

def load_nodes_from_config(config):
    # load config
    # pass circuit-networks-nodes section into load_nodes_from_paths    
    return

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
    
    #nodes_regex = "_nodes.h5"
    #node_types_regex = "_node_types.csv"

    #nodes_h5_fpaths = glob.glob(os.path.join(network_dir,'*'+nodes_regex))
    #node_types_fpaths = glob.glob(os.path.join(network_dir,'*'+node_types_regex))

    #regions = [re.findall('^[^_]+', os.path.basename(n))[0] for n in nodes_h5_fpaths]
    region_dict = {}

    #Need to get all cell groups for each region
    def get_node_table(cell_models_file, cells_file, population=None):
        cm_df = pd.read_csv(cell_models_file, sep=' ')
        cm_df.set_index('node_type_id', inplace=True)

        cells_h5 = h5py.File(cells_file, 'r')
        if population is None:
            if len(cells_h5['/nodes']) > 1:
                raise Exception('Multiple populations in nodes file. Not currently supported. Should be easy to implement when needed. Let Tyler know.')
            else:
                population = list(cells_h5['/nodes'])[0]

        nodes_grp = cells_h5['/nodes'][population]
        c_df = pd.DataFrame({'node_id': nodes_grp['node_id'], 'node_type_id': nodes_grp['node_type_id']})
        c_df.set_index('node_id', inplace=True)

        nodes_df = pd.merge(left=c_df,
                            right=cm_df,
                            how='left',
                            left_on='node_type_id',
                            right_index=True)  # use 'model_id' key to merge, for right table the "model_id" is an index
        
        if 'positions' in list(nodes_grp['0']):
            cpos = pd.DataFrame({'node_id': nodes_grp['node_id'],"pos_x":nodes_grp['0']['positions'][:,0],"pos_y":nodes_grp['0']['positions'][:,1],"pos_z":nodes_grp['0']['positions'][:,2]})
            cpos.set_index('node_id', inplace=True)

            nodes_df = pd.merge(left=nodes_df,
                                right=cpos,
                                how='left',
                                left_index=True,
                                right_index=True)

        return nodes_df
    
    #for region, cell_models_file, cells_file in zip(regions, node_types_fpaths, nodes_h5_fpaths):
    #    region_dict[region] = get_node_table(cell_models_file,cells_file,population=region)
    for nodes in node_paths:
        cell_models_file = nodes["nodes_file"]
        cells_file = nodes["node_types_file"]
        region, region_name = get_node_table(cell_models_file,cells_file)
        region_dict[region_name] = region

    #cell_num = 2
    #print(region_dict["hippocampus"].iloc[cell_num]["node_type_id"])
    #print(list(set(region_dict["hippocampus"]["node_type_id"]))) #Unique

    return region_dict
    
def load_edges_from_config(config):
    return

def load_edges(edges_file, edge_types_file):
    return

def load_edges_from_paths(network_dir='network'):
    """
    Returns: A dictionary of connections with filenames (minus _edges.h5) as keys

    TODO there is an unhealthy reliance on filenames
    """
    import h5py
    
    edges_regex = "_edges.h5"
    edge_types_regex = "_edge_types.csv"

    edges_h5_fpaths = glob.glob(os.path.join(network_dir,'*'+edges_regex))
    edge_types_fpaths = glob.glob(os.path.join(network_dir,'*'+edge_types_regex))

    connections = [re.findall('^[A-Za-z0-9]+_[A-Za-z0-9][^_]+', os.path.basename(n))[0] for n in edges_h5_fpaths]
    connections_dict = {}

    def get_connection_table(connection_models_file, connections_file,population=None):
        cm_df = pd.read_csv(connection_models_file, sep=' ')
        cm_df.set_index('edge_type_id', inplace=True)

        connections_h5 = h5py.File(connections_file, 'r')

        if population is None:
            if len(connections_h5['/edges']) > 1:
                raise Exception('Multiple populations in edges file. Please specify one to plot using population param')
            else:
                population = list(connections_h5['/edges'])[0]

        conn_grp = connections_h5['/edges'][population]
        c_df = pd.DataFrame({'edge_type_id': conn_grp['edge_type_id'], 'source_node_id': conn_grp['source_node_id'],
                             'target_node_id': conn_grp['target_node_id']})

        c_df.set_index('edge_type_id', inplace=True)

        nodes_df = pd.merge(left=c_df,
                            right=cm_df,
                            how='left',
                            left_index=True,
                            right_index=True)  # use 'model_id' key to merge, for right table the "model_id" is an index
        return nodes_df
    
    for connection, conn_models_file, conns_file in zip(connections, edge_types_fpaths, edges_h5_fpaths):
        connections_dict[connection] = get_connection_table(conn_models_file,conns_file)

    return connections_dict

def connection_totals(nodes=None,edges=None,populations=[]):
    if not nodes:
        nodes = load_nodes()
    if not edges:
        edges = load_connections()

    total_cell_types = len(list(set(nodes[populations[0]]["node_type_id"])))
    nodes_hip = pd.DataFrame(nodes[populations[0]])
    pop_names = nodes_hip.pop_name.unique()

    e_matrix = np.zeros((total_cell_types,total_cell_types))

    for i, key in enumerate(list(edges)):
        if i==0:#TODO TAKE OUT FROM TESTING
            continue
        
        for j, row in edges[key].iterrows():
            source = row["source_node_id"]
            target = row["target_node_id"]
            source_node_type = nodes[populations[0]].iloc[source]["node_type_id"]
            target_node_type = nodes[populations[0]].iloc[target]["node_type_id"]

            source_index = int(source_node_type - 100)
            target_index = int(target_node_type - 100)

            e_matrix[source_index,target_index]+=1
            
    return e_matrix, pop_names

def percent_connectivity(nodes=None, edges=None, conn_totals=None, pop_names=None,populations=[]):
    if nodes == None:
        nodes = load_nodes()
    if edges == None:
        edges = load_connections()
    if conn_totals == None:
        conn_totals,pop_names=connection_totals(nodes=nodes,edges=edges,populations=populations)

    #total_cell_types = len(list(set(nodes[populations[0]]["node_type_id"])))
    vc = nodes[populations[0]].apply(pd.Series.value_counts)
    vc = vc["node_type_id"].dropna().sort_index()
    vc = list(vc)

    max_connect = np.ones((len(vc),len(vc)),dtype=np.float)

    for a, i in enumerate(vc):
        for b, j in enumerate(vc):
            max_connect[a,b] = i*j
    ret = conn_totals/max_connect
    ret = ret*100
    ret = np.around(ret, decimals=1)

    return ret, pop_names
    
    
def connection_average_synapses():
    return

def connection_divergence_average(nodes=None, edges=None,populations=[],convergence=False):
    """
    For each cell in source count # of connections in target and average
    """
    if nodes == None:
        nodes = load_nodes()
    if edges == None:
        edges = load_connections()

    nodes_hip = pd.DataFrame(nodes[populations[0]])
    pop_names = nodes_hip.pop_name.unique()

    nodes = nodes[list(nodes)[1]]
    edges = edges[list(edges)[1]]

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