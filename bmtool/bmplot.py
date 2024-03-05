"""
Want to be able to take multiple plot names in and plot them all at the same time, to save time
https://stackoverflow.com/questions/458209/is-there-a-way-to-detach-matplotlib-plots-so-that-the-computation-can-continue
"""
from .util import util

import argparse,os,sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython
from IPython.display import display, HTML
import statistics
import pandas as pd
import os
import sys

from .util.util import CellVarsFile #, missing_units
from bmtk.analyzer.utils import listify

use_description = """

Plot BMTK models easily.

python -m bmtool.plot 
"""

def is_notebook() -> bool:
    """
    Used to tell if inside jupyter notebook or not. This is used to tell if we should use plt.show or not
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def total_connection_matrix(config=None,title=None,sources=None, targets=None, sids=None, tids=None,no_prepend_pop=False,save_file=None,synaptic_info='0',include_gap=True):
    """
    Generates connection plot displaying total connection or other stats
    config: A BMTK simulation config 
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier 
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    save_file: If plot should be saved
    synaptic_info: '0' for total connections, '1' for mean and stdev connections, '2' for all synapse .mod files used, '3' for all synapse .json files used
    include_gap: Determines if connectivity shown should include gap junctions + chemical synapses. False will only include chemical
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []
    text,num, source_labels, target_labels = util.connection_totals(config=config,nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop,synaptic_info=synaptic_info,include_gap=include_gap)

    if title == None or title=="":
        title = "Total Connections"
    if synaptic_info=='1':
        title = "Mean and Stdev # of Conn on Target"    
    if synaptic_info=='2':
        title = "All Synapse .mod Files Used"
    if synaptic_info=='3':
        title = "All Synapse .json Files Used"
    plot_connection_info(text,num,source_labels,target_labels,title, syn_info=synaptic_info, save_file=save_file)
    return
    
def percent_connection_matrix(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None,method = 'total',include_gap=True):
    """
    Generates a plot showing the percent connectivity of a network
    config: A BMTK simulation config 
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier 
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    method: what percent to displace on the graph 'total','uni',or 'bi' for total connections, unidirectional connections or bidirectional connections
    save_file: If plot should be saved
    include_gap: Determines if connectivity shown should include gap junctions + chemical synapses. False will only include chemical
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []
    text,num, source_labels, target_labels = util.percent_connections(config=config,nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop,method=method,include_gap=include_gap)
    if title == None or title=="":
        title = "Percent Connectivity"


    plot_connection_info(text,num,source_labels,target_labels,title, save_file=save_file)
    return

def probability_connection_matrix(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, 
                            no_prepend_pop=False,save_file=None, dist_X=True,dist_Y=True,dist_Z=True,bins=8,line_plot=False,verbose=False,include_gap=True):
    """
    Generates probability graphs
    need to look into this more to see what it does
    needs model_template to be defined to work
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []

    throwaway, data, source_labels, target_labels = util.connection_probabilities(config=config,nodes=None,
        edges=None,sources=sources,targets=targets,sids=sids,tids=tids,
        prepend_pop=not no_prepend_pop,dist_X=dist_X,dist_Y=dist_Y,dist_Z=dist_Z,num_bins=bins,include_gap=include_gap)
    if not data.any():
        return
    if data[0][0]==-1:
        return
    #plot_connection_info(data,source_labels,target_labels,title, save_file=save_file)

    #plt.clf()# clears previous plots
    np.seterr(divide='ignore', invalid='ignore')
    num_src, num_tar = data.shape
    fig, axes = plt.subplots(nrows=num_src, ncols=num_tar, figsize=(12,12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for x in range(num_src):
        for y in range(num_tar):
            ns = data[x][y]["ns"]
            bins = data[x][y]["bins"]
            
            XX = bins[:-1]
            YY = ns[0]/ns[1]

            if line_plot:
                axes[x,y].plot(XX,YY)
            else:
                axes[x,y].bar(XX,YY)

            if x == num_src-1:
                axes[x,y].set_xlabel(target_labels[y])
            if y == 0:
                axes[x,y].set_ylabel(source_labels[x])

            if verbose:
                print("Source: [" + source_labels[x] + "] | Target: ["+ target_labels[y] +"]")
                print("X:")
                print(XX)
                print("Y:")
                print(YY)

    tt = "Distance Probability Matrix"
    if title:
        tt = title
    st = fig.suptitle(tt, fontsize=14)
    fig.text(0.5, 0.04, 'Target', ha='center')
    fig.text(0.04, 0.5, 'Source', va='center', rotation='vertical')
    notebook = is_notebook
    if notebook == False:
        fig.show()

    return

def convergence_connection_matrix(config=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None,convergence=True,method='mean+std',include_gap=True):
    """
    Generates connection plot displaying convergence data
    config: A BMTK simulation config 
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier 
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    save_file: If plot should be saved
    method: 'mean','min','max','stdev' or 'mean+std' connvergence plot 
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    return divergence_connection_matrix(config,title ,sources, targets, sids, tids, no_prepend_pop, save_file ,convergence, method,include_gap=include_gap)

def divergence_connection_matrix(config=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None,convergence=False,method='mean+std',include_gap=True):
    """
    Generates connection plot displaying divergence data
    config: A BMTK simulation config 
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier 
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    save_file: If plot should be saved
    method: 'mean','min','max','stdev', and 'mean+std' for divergence plot
    """
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []

    syn_info, data, source_labels, target_labels = util.connection_divergence(config=config,nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop,convergence=convergence,method=method,include_gap=include_gap)

    
    #data, labels = util.connection_divergence_average(config=config,nodes=nodes,edges=edges,populations=populations)

    if title == None or title=="":

        if method == 'min':
            title = "Minimum "
        elif method == 'max':
            title = "Maximum "
        elif method == 'std':
            title = "Standard Deviation "
        elif method == 'mean':
            title = "Mean "
        else: 
            title = 'Mean + Std '

        if convergence:
            title = title + "Synaptic Convergence"
        else:
            title = title + "Synaptic Divergence"
    plot_connection_info(syn_info,data,source_labels,target_labels,title, save_file=save_file)
    return

def connection_histogram(config=None,nodes=None,edges=None,sources=[],targets=[],sids=[],tids=[],prepend_pop=True,synaptic_info='0',
                      source_cell = None,target_cell = None,include_gap=True):
    """
    Generates histogram of number of connections individual cells in a population receieve from another population
    config: A BMTK simulation config 
    sources: network name(s) to plot
    targets: network name(s) to plot
    sids: source node identifier 
    tids: target node identifier
    no_prepend_pop: dictates if population name is displayed before sid or tid when displaying graph
    source_cell: where connections are coming from
    target_cell: where connections on coming onto
    save_file: If plot should be saved
    """
    def connection_pair_histogram(**kwargs):
        edges = kwargs["edges"] 
        source_id_type = kwargs["sid"]
        target_id_type = kwargs["tid"]
        source_id = kwargs["source_id"]
        target_id = kwargs["target_id"]
        if source_id == source_cell and target_id == target_cell:
            temp = edges[(edges[source_id_type] == source_id) & (edges[target_id_type]==target_id)]
            if include_gap == False:
                temp = temp[temp['is_gap_junction'] != True]
            node_pairs = temp.groupby('target_node_id')['source_node_id'].count()
            conn_mean = statistics.mean(node_pairs.values)
            conn_std = statistics.stdev(node_pairs.values)
            conn_median = statistics.median(node_pairs.values)
            label = "mean {:.2f} std ({:.2f}) median {:.2f}".format(conn_mean,conn_std,conn_median)
            plt.hist(node_pairs.values,density=True,bins='auto',stacked=True,label=label)
            plt.legend()
            plt.xlabel("# of conns from {} to {}".format(source_cell,target_cell))
            plt.ylabel("Density")
            plt.show()
        else: # dont care about other cell pairs so pass
            pass

    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []
    util.relation_matrix(config,nodes,edges,sources,targets,sids,tids,prepend_pop,relation_func=connection_pair_histogram,synaptic_info=synaptic_info)

def edge_histogram_matrix(config=None,sources = None,targets=None,sids=None,tids=None,no_prepend_pop=None,edge_property = None,time = None,time_compare = None,report=None,title=None,save_file=None):
    """
    write about function here
    """
    
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []

    if time_compare:
        time_compare = int(time_compare)

    data, source_labels, target_labels = util.edge_property_matrix(edge_property,nodes=None,edges=None,config=config,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop,report=report,time=time,time_compare=time_compare)

    # Fantastic resource
    # https://stackoverflow.com/questions/7941207/is-there-a-function-to-make-scatterplot-matrices-in-matplotlib 
    num_src, num_tar = data.shape
    fig, axes = plt.subplots(nrows=num_src, ncols=num_tar, figsize=(12,12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for x in range(num_src):
        for y in range(num_tar):
            axes[x,y].hist(data[x][y])

            if x == num_src-1:
                axes[x,y].set_xlabel(target_labels[y])
            if y == 0:
                axes[x,y].set_ylabel(source_labels[x])

    tt = edge_property + " Histogram Matrix"
    if title:
        tt = title
    st = fig.suptitle(tt, fontsize=14)
    fig.text(0.5, 0.04, 'Target', ha='center')
    fig.text(0.04, 0.5, 'Source', va='center', rotation='vertical')
    plt.draw()

def plot_connection_info(text, num, source_labels,target_labels, title, syn_info='0', save_file=None):
    """
    write about function here
    """
    
    #num = pd.DataFrame(num).fillna('nc').to_numpy() # replace nan with nc * does not work with imshow
    
    num_source=len(source_labels)
    num_target=len(target_labels)
    matplotlib.rc('image', cmap='viridis')
    
    fig1, ax1 = plt.subplots(figsize=(num_source,num_target))
    im1 = ax1.imshow(num)
    #fig.colorbar(im, ax=ax,shrink=0.4)
    # We want to show all ticks...
    ax1.set_xticks(list(np.arange(len(target_labels))))
    ax1.set_yticks(list(np.arange(len(source_labels))))
    # ... and label them with the respective list entries
    ax1.set_xticklabels(target_labels)
    ax1.set_yticklabels(source_labels,size=12, weight = 'semibold')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor", size=12, weight = 'semibold')
    
    # Loop over data dimensions and create text annotations.
    for i in range(num_source):
        for j in range(num_target):
            edge_info = text[i,j]
            if syn_info =='2' or syn_info =='3':
                if num_source > 8 and num_source <20:
                    fig_text = ax1.text(j, i, edge_info,
                            ha="center", va="center", color="k",rotation=37.5, size=8, weight = 'semibold')
                elif num_source > 20:
                    fig_text = ax1.text(j, i, edge_info,
                            ha="center", va="center", color="k",rotation=37.5, size=7, weight = 'semibold')
                else:
                    fig_text = ax1.text(j, i, edge_info,
                            ha="center", va="center", color="k",rotation=37.5, size=11, weight = 'semibold')
            else:
                fig_text = ax1.text(j, i, edge_info,
                            ha="center", va="center", color="k", size=11, weight = 'semibold')

    ax1.set_ylabel('Source', size=11, weight = 'semibold')
    ax1.set_xlabel('Target', size=11, weight = 'semibold')
    ax1.set_title(title,size=20, weight = 'semibold')
    #plt.tight_layout()
    notebook = is_notebook()
    if notebook == False:
        fig1.show()
    if save_file:
        plt.savefig(save_file)
    return

def raster_old(config=None,title=None,populations=['hippocampus']):
    """
    old function probs dep
    """
    conf = util.load_config(config)
    spikes_path = os.path.join(conf["output"]["output_dir"],conf["output"]["spikes_file"])
    nodes = util.load_nodes_from_config(config)
    plot_spikes(nodes,spikes_path)
    return

def raster(config=None,title=None,population=None,group_key='pop_name'):
    """
    old function probs dep or more to new spike module?
    """
    conf = util.load_config(config)
    
    cells_file = conf["networks"]["nodes"][0]["nodes_file"]
    cell_types_file = conf["networks"]["nodes"][0]["node_types_file"]
    spikes_path = os.path.join(conf["output"]["output_dir"],conf["output"]["spikes_file"])

    from bmtk.analyzer.visualization import spikes
    spikes.plot_spikes(cells_file,cell_types_file,spikes_path,population=population,group_key=group_key)
    return

def plot_spikes(nodes, spikes_file,save_file=None):   
    """
    old function probs dep
    """
    import h5py

    spikes_h5 = h5py.File(spikes_file, 'r')
    spike_gids = np.array(spikes_h5['/spikes/gids'], dtype=np.uint)
    spike_times = np.array(spikes_h5['/spikes/timestamps'], dtype=np.float)
        
    spikes = np.rot90(np.vstack((spike_gids,spike_times))) # Make array [[gid spiketime],[gid2 spiketime2]]

    #spikes = spikes[spikes[:,0].argsort()] # Sort by cell number

    """
    Author: Tyler Banks
    Loop through all spike files, create a list of when each cell spikes, plot.
    """
   
    cell_types = ['EC','CA3e','CA3o','CA3b','DGg','DGh','DGb']
    cell_nums = [30,63,8,8,384,32,32]
    d = [[] for _ in range(sum(cell_nums))]
    
    color_picker=['red','orange','yellow','green','blue','purple','black']
    colors = []
    offset=0
    
    for i, row in enumerate(spikes):
        d[int(row[0])+offset].append(row[1])
            
    for i, n in enumerate(cell_nums):
        for _ in range(n):
            colors.append(color_picker[i])
        
    fig, axs = plt.subplots(1,1)
    axs.eventplot(d,colors=colors)
    axs.set_title('Hipp BMTK')
    axs.set_ylabel('Cell Number')
    axs.set_xlabel('Time (ms)')
    axs.legend(cell_types[::-1])
    
    leg = axs.get_legend()
    for i,c in enumerate(color_picker):
        leg.legendHandles[-i-1].set_color(c)
    
    #splt.savefig('raster3_after_pycvode_fixes.png')
    if save_file:
        plt.savefig(save_file)
    
    plt.draw()
    
    return
    
def plot_3d_positions(config=None,populations_list=None,group_by=None,title=None,save_file=None):
    """
    plots a 3D graph of all cells with x,y,z location
    config: A BMTK simulation config 
    populations_list: Which network(s) to plot 
    group_by: How to name cell groups
    title: plot title
    save_file: If plot should be saved
    """
    
    if not config:
        raise Exception("config not defined")
    if populations_list == None:
        populations_list = "all"
    group_keys = group_by
    if title == None:
        title = "3D positions"

    nodes = util.load_nodes_from_config(config)
    
    if 'all' in populations_list:
        populations = list(nodes)
    else:
        populations = populations_list.split(",")

    group_keys = group_keys.split(",")
    group_keys += (len(populations)-len(group_keys)) * ["node_type_id"] #Extend the array to default values if not enough given
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    handles = []
    for nodes_key,group_key in zip(list(nodes),group_keys):
        if 'all' not in populations and nodes_key not in populations:
            continue
            
        nodes_df = nodes[nodes_key]

        if group_key is not None:
            if group_key not in nodes_df:
                raise Exception('Could not find column {}'.format(group_key))
            groupings = nodes_df.groupby(group_key)

            n_colors = nodes_df[group_key].nunique()
            color_norm = colors.Normalize(vmin=0, vmax=(n_colors-1))
            scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
            color_map = [scalar_map.to_rgba(i) for i in range(0, n_colors)]
        else:
            groupings = [(None, nodes_df)]
            color_map = ['blue']

        for color, (group_name, group_df) in zip(color_map, groupings):
            if "pos_x" not in group_df: #could also check model type == virtual
                continue #can't plot them if there isn't an xy coordinate (may be virtual)
            h = ax.scatter(group_df["pos_x"],group_df["pos_y"],group_df["pos_z"],color=color,label=group_name)
            handles.append(h)
    if not handles:
        return
    plt.title(title)
    plt.legend(handles=handles)
    
    plt.draw()

    if save_file:
        plt.savefig(save_file)
    notebook = is_notebook
    if notebook == False:
        plt.show()

    return

def cell_rotation_3d(config=None, populations_list=None, group_by=None, title=None, save_file=None, quiver_length=None, arrow_length_ratio=None, group=None, max_cells=1000000):
    from scipy.spatial.transform import Rotation as R
    if not config:
        raise Exception("config not defined")

    if populations_list is None:
        populations_list = ["all"]

    group_keys = group_by.split(",") if group_by else []

    if title is None:
        title = "Cell rotations"

    nodes = util.load_nodes_from_config(config)

    if 'all' in populations_list:
        populations = list(nodes)
    else:
        populations = populations_list

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    handles = []

    for nodes_key, group_key in zip(list(nodes), group_keys):
        if 'all' not in populations and nodes_key not in populations:
            continue

        nodes_df = nodes[nodes_key]

        if group_key is not None:
            if group_key not in nodes_df.columns:
                raise Exception(f'Could not find column {group_key}')
            groupings = nodes_df.groupby(group_key)

            n_colors = nodes_df[group_key].nunique()
            color_norm = colors.Normalize(vmin=0, vmax=(n_colors - 1))
            scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
            color_map = [scalar_map.to_rgba(i) for i in range(n_colors)]
        else:
            groupings = [(None, nodes_df)]
            color_map = ['blue']

        cells_plotted = 0
        for color, (group_name, group_df) in zip(color_map, groupings):
            if group and group_name not in group.split(","):
                continue

            if "pos_x" not in group_df or "rotation_angle_xaxis" not in group_df:
                continue

            if cells_plotted >= max_cells:
                continue

            if len(group_df) + cells_plotted > max_cells:
                total_remaining = max_cells - cells_plotted
                group_df = group_df[:total_remaining]

            cells_plotted += len(group_df)

            X = group_df["pos_x"]
            Y = group_df["pos_y"]
            Z = group_df["pos_z"]
            U = group_df["rotation_angle_xaxis"].values
            V = group_df["rotation_angle_yaxis"].values
            W = group_df["rotation_angle_zaxis"].values

            if U is None:
                U = np.zeros(len(X))
            if V is None:
                V = np.zeros(len(Y))
            if W is None:
                W = np.zeros(len(Z))

            # Create rotation matrices from Euler angles
            rotations = R.from_euler('xyz', np.column_stack((U, V, W)), degrees=False)

            # Define initial vectors 
            init_vectors = np.column_stack((np.ones(len(X)), np.zeros(len(Y)), np.zeros(len(Z))))

            # Apply rotations to initial vectors
            rots = np.dot(rotations.as_matrix(), init_vectors.T).T

            # Extract x, y, and z components of the rotated vectors
            rot_x = rots[:, 0]
            rot_y = rots[:, 1]
            rot_z = rots[:, 2]

            h = ax.quiver(X, Y, Z, rot_x, rot_y, rot_z, color=color, label=group_name, arrow_length_ratio=arrow_length_ratio, length=quiver_length)
            ax.scatter(X, Y, Z, color=color, label=group_name)
            ax.set_xlim([min(X), max(X)])
            ax.set_ylim([min(Y), max(Y)])
            ax.set_zlim([min(Z), max(Z)])
            handles.append(h)

    if not handles:
        return

    plt.title(title)
    plt.legend(handles=handles)
    plt.draw()

    if save_file:
        plt.savefig(save_file)
    notebook = is_notebook
    if notebook == False:
        plt.show()

def plot_network_graph(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None,edge_property='model_template'):
    if not config:
        raise Exception("config not defined")
    if not sources or not targets:
        raise Exception("Sources or targets not defined")
    sources = sources.split(",")
    targets = targets.split(",")
    if sids:
        sids = sids.split(",")
    else:
        sids = []
    if tids:
        tids = tids.split(",")
    else:
        tids = []
    throw_away, data, source_labels, target_labels = util.connection_graph_edge_types(config=config,nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop,edge_property=edge_property)

    if title == None or title=="":
        title = "Network Graph"
    
    import networkx as nx

    net_graph = nx.MultiDiGraph() #or G = nx.MultiDiGraph()
    
    edges = []
    edge_labels = {}
    for node in list(set(source_labels+target_labels)):
        net_graph.add_node(node)

    for s, source in enumerate(source_labels):
        for t, target in enumerate(target_labels):
            relationship = data[s][t]
            for i, relation in enumerate(relationship):
                edge_labels[(source,target)]=relation
                edges.append([source,target])

    net_graph.add_edges_from(edges)
    #pos = nx.spring_layout(net_graph,k=0.50,iterations=20)
    pos = nx.shell_layout(net_graph)
    plt.figure()
    nx.draw(net_graph,pos,edge_color='black', width=1,linewidths=1,\
        node_size=500,node_color='white',arrowstyle='->',alpha=0.9,\
        labels={node:node for node in net_graph.nodes()})

    nx.draw_networkx_edge_labels(net_graph,pos,edge_labels=edge_labels,font_color='red')
    plt.show()

    return

def plot_report(config_file=None, report_file=None, report_name=None, variables=None, gids=None):
    if report_file is None:
        report_name, report_file = _get_cell_report(config_file, report_name)

    var_report = CellVarsFile(report_file)
    variables = listify(variables) if variables is not None else var_report.variables
    gids = listify(gids) if gids is not None else var_report.gids
    time_steps = var_report.time_trace

    def __units_str(var):
        units = var_report.units(var)
        if units == CellVarsFile.UNITS_UNKNOWN:
            units = missing_units.get(var, '')
        return '({})'.format(units) if units else ''

    n_plots = len(variables)
    if n_plots > 1:
        # If more than one variale to plot do so in different subplots
        f, axarr = plt.subplots(n_plots, 1)
        for i, var in enumerate(variables):
            for gid in gids:
                axarr[i].plot(time_steps, var_report.data(gid=gid, var_name=var), label='gid {}'.format(gid))

            axarr[i].legend()
            axarr[i].set_ylabel('{} {}'.format(var, __units_str(var)))
            if i < n_plots - 1:
                axarr[i].set_xticklabels([])

        axarr[i].set_xlabel('time (ms)')

    elif n_plots == 1:
        # For plotting a single variable
        plt.figure()
        for gid in gids:
            plt.plot(time_steps, var_report.data(gid=gid, var_name=variables[0]), label='gid {}'.format(gid))
        plt.ylabel('{} {}'.format(variables[0], __units_str(variables[0])))
        plt.xlabel('time (ms)')
        plt.legend()
    else:
        return

    plt.show()

def plot_report_default(config, report_name, variables, gids):

    if variables:
        variables = variables.split(',')
    if gids:
        gids = [int(i) for i in gids.split(',')]    

    if report_name:
         cfg = util.load_config(config)
         report_file = os.path.join(cfg['output']['output_dir'],report_name+'.h5')
    plot_report(config_file=config, report_file=report_file, report_name=report_name, variables=variables, gids=gids);
    
    return

# The following code was developed by Matthew Stroud 7/15/21 neural engineering supervisor: Satish Nair
# This is an extension of bmtool: a development of Tyler Banks. 
# The goal of the sim_setup() function is to output relevant simulation information that can be gathered by providing only the main configuration file.


def sim_setup(config_file='simulation_config.json',network=None):
    if "JPY_PARENT_PID" in os.environ:
        print("Inside a notebook:")
        get_ipython().run_line_magic('matplotlib', 'tk')

    
    # Output tables that contain the cells involved in the configuration file given. Also returns the first biophysical network found
    bio=plot_basic_cell_info(config_file)
    if network == None:
        network=bio

    print("Please wait. This may take a while depending on your network size...")
    # Plot connection probabilities
    plt.close(1)
    probability_connection_matrix(config=config_file,sources=network,targets=network, no_prepend_pop=True,sids= 'pop_name', tids= 'pop_name', bins=10,line_plot=True,verbose=False)
    # Gives current clamp information
    plot_I_clamps(config_file)
    # Plot spike train info
    plot_inspikes(config_file)
    # Using bmtool, print total number of connections between cell groups
    total_connection_matrix(config=config_file,sources='all',targets='all',sids='pop_name',tids='pop_name',title='All Connections found', size_scalar=2, no_prepend_pop=True, synaptic_info='0')
    # Plot 3d positions of the network
    plot_3d_positions(populations='all',config=config_file,group_by='pop_name',title='3D Positions',save_file=None)

def plot_I_clamps(fp):
    print("Plotting current clamp info...")
    clamps = util.load_I_clamp_from_config(fp)
    if not clamps:
        print("     No current clamps were found.")
        return
    time=[]
    num_clamps=0
    fig, ax = plt.subplots()
    ax = plt.gca()
    for clinfo in clamps:
        simtime=len(clinfo[0])*clinfo[1]
        time.append(np.arange(0,simtime,clinfo[1]).tolist())

        line,=ax.plot(time[num_clamps],clinfo[0],drawstyle='steps')
        line.set_label('I Clamp to: '+str(clinfo[2]))
        plt.legend()
        num_clamps=num_clamps+1

def plot_basic_cell_info(config_file):
    print("Network and node info:")
    nodes=util.load_nodes_from_config(config_file)
    if not nodes:
        print("No nodes were found.")
        return
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    bio=[]
    i=0
    j=0
    for j in nodes:
        node=nodes[j]
        node_type_id=node['node_type_id']
        num_cells=len(node['node_type_id'])
        if node['model_type'][0]=='virtual':
            CELLS=[]
            count=1
            for i in range(num_cells-1):
                if(node_type_id[i]==node_type_id[i+1]):
                    count+=1
                else:
                    node_type=node_type_id[i]
                    pop_name=node['pop_name'][i]
                    model_type=node['model_type'][i]
                    CELLS.append([node_type,pop_name,model_type,count])
                    count=1
            else:
                node_type=node_type_id[i]
                pop_name=node['pop_name'][i]
                model_type=node['model_type'][i]
                CELLS.append([node_type,pop_name,model_type,count])
                count=1
            df1 = pd.DataFrame(CELLS, columns = ["node_type","pop_name","model_type","count"])
            print(j+':')
            notebook = is_notebook()
            if notebook == True:
                display(HTML(df1.to_html()))
            else:
                print(df1)
        elif node['model_type'][0]=='biophysical':
            CELLS=[]
            count=1
            node_type_id=node['node_type_id']
            num_cells=len(node['node_type_id'])
            for i in range(num_cells-1):
                if(node_type_id[i]==node_type_id[i+1]):
                    count+=1
                else:
                    node_type=node_type_id[i]
                    pop_name=node['pop_name'][i]
                    model_type=node['model_type'][i]
                    model_template=node['model_template'][i]
                    morphology=node['morphology'][i] if node['morphology'][i] else ''
                    CELLS.append([node_type,pop_name,model_type,model_template,morphology,count])
                    count=1
            else:
                node_type=node_type_id[i]
                pop_name=node['pop_name'][i]
                model_type=node['model_type'][i]
                model_template=node['model_template'][i]
                morphology=node['morphology'][i] if node['morphology'][i] else ''
                CELLS.append([node_type,pop_name,model_type,model_template,morphology,count])
                count=1
            df2 = pd.DataFrame(CELLS, columns = ["node_type","pop_name","model_type","model_template","morphology","count"])
            print(j+':')
            bio.append(j)
            notebook = is_notebook()
            if notebook == True:
                display(HTML(df2.to_html()))
            else:
                print(df2)
    if len(bio)>0:      
        return bio[0]        

def plot_inspikes(fp):
    
    print("Plotting spike Train info...")
    trains = util.load_inspikes_from_config(fp)
    if not trains:
        print("No spike trains were found.")
    num_trains=len(trains)

    time=[]
    node=[]
    fig, ax = plt.subplots(num_trains, figsize=(12,12),squeeze=False)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    pos=0
    for tr in trains:
        node_group=tr[0][2]
        if node_group=='':
            node_group='Defined by gids (y-axis)'
        time=[]
        node=[]
        for sp in tr:
            node.append(sp[1])
            time.append(sp[0])

        #plotting spike train
        
        ax[pos,0].scatter(time,node,s=1)
        ax[pos,0].title.set_text('Input Spike Train to: '+node_group)
        plt.xticks(rotation = 45)
        if num_trains <=4:
            ax[pos,0].xaxis.set_major_locator(plt.MaxNLocator(20))
        if num_trains <=9 and num_trains >4:
            ax[pos,0].xaxis.set_major_locator(plt.MaxNLocator(4))
        elif num_trains <9:
            ax[pos,0].xaxis.set_major_locator(plt.MaxNLocator(2))
        #fig.suptitle('Input Spike Train to: '+node_group, fontsize=14)
        fig.show()
        pos+=1
