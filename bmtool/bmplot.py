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
import math
import pandas as pd
import h5py
import os
import sys
import time

from .util.util import CellVarsFile #, missing_units
from bmtk.analyzer.utils import listify

use_description = """

Plot BMTK models easily.

python -m bmtool.plot 
"""

def connection_matrix(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, size_scalar=1,no_prepend_pop=False,save_file=None,synaptic_info='0'):
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
    text,num, source_labels, target_labels = util.connection_totals(config=config,nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop,synaptic_info=synaptic_info)

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
    
def percent_connection_matrix(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None):
    text,num, source_labels, target_labels = util.connection_totals(config=config,nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop)

    if title == None or title=="":
        title = "Percent Connectivity"

    plot_connection_info(text,num,source_labels,target_labels,title, save_file=save_file)
    return

def probability_connection_matrix(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, 
                            no_prepend_pop=False,save_file=None, dist_X=True,dist_Y=True,dist_Z=True,bins=8,line_plot=False,verbose=False):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
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
        prepend_pop=not no_prepend_pop,dist_X=dist_X,dist_Y=dist_Y,dist_Z=dist_Z,num_bins=bins)
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
    fig.show()

    return

def convergence_connection_matrix(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None,convergence=True,method='mean'):
    return divergence_connection_matrix(config,nodes ,edges ,title ,sources, targets, sids, tids, no_prepend_pop, save_file ,convergence, method)

def divergence_connection_matrix(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None,convergence=False,method='mean'):
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

    syn_info, data, source_labels, target_labels = util.connection_divergence(config=config,nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop,convergence=convergence,method=method)

    
    #data, labels = util.connection_divergence_average(config=config,nodes=nodes,edges=edges,populations=populations)

    if title == None or title=="":

        if method == 'min':
            title = "Minimum "
        elif method == 'max':
            title = "Maximum "
        elif method == 'std':
            title = "Standard Deviation "
        else:
            title = "Mean "

        if convergence:
            title = title + "Synaptic Convergence"
        else:
            title = title + "Synaptic Divergence"

    plot_connection_info(data,data,source_labels,target_labels,title, save_file=save_file)
    return

def edge_histogram_matrix(**kwargs):
    config = kwargs["config"]
    sources = kwargs["sources"]
    targets = kwargs["targets"]
    sids = kwargs["sids"]
    tids = kwargs["tids"]
    no_prepend_pop = kwargs["no_prepend_pop"]
    edge_property = kwargs["edge_property"]
    time = int(kwargs["time"])
    time_compare = kwargs["time_compare"]
    report = kwargs["report"]

    title = kwargs["title"]

    save_file = kwargs["save_file"] 
    
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
    
    fig1.show()

    if save_file:
        plt.savefig(save_file)
    return

def raster_old(config=None,title=None,populations=['hippocampus']):
    conf = util.load_config(config)
    spikes_path = os.path.join(conf["output"]["output_dir"],conf["output"]["spikes_file"])
    nodes = util.load_nodes_from_config(config)
    plot_spikes(nodes,spikes_path)
    return

def raster(config=None,title=None,population=None,group_key='pop_name'):
    conf = util.load_config(config)
    
    cells_file = conf["networks"]["nodes"][0]["nodes_file"]
    cell_types_file = conf["networks"]["nodes"][0]["node_types_file"]
    spikes_path = os.path.join(conf["output"]["output_dir"],conf["output"]["spikes_file"])

    from bmtk.analyzer.visualization import spikes
    spikes.plot_spikes(cells_file,cell_types_file,spikes_path,population=population,group_key=group_key)
    return

def plot_spikes(nodes, spikes_file,save_file=None):   
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
    
def plot_3d_positions(**kwargs):
    populations_list = kwargs["populations"]
    config = kwargs["config"]
    group_keys = kwargs["group_by"]
    title = kwargs["title"]
    save_file = kwargs["save_file"]

    nodes = util.load_nodes_from_config(config)
    
    if 'all' in populations_list:
        populations = list(nodes)
    else:
        populations = populations_list.split(",")

    group_keys = group_keys.split(",")
    group_keys += (len(populations)-len(group_keys)) * ["node_type_id"] #Extend the array to default values if not enough given
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
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

    return


def cell_rotation_3d(**kwargs):
    populations_list = kwargs["populations"]
    config = kwargs["config"]
    group_keys = kwargs["group_by"]
    title = kwargs.get("title")
    save_file = kwargs["save_file"]
    quiver_length = kwargs["quiver_length"]
    arrow_length_ratio = kwargs["arrow_length_ratio"]
    group = kwargs["group"]
    max_cells = kwargs.get("max_cells",999999999)
    init_vector = kwargs.get("init_vector","1,0,0")

    nodes = util.load_nodes_from_config(config)

    if 'all' in populations_list:
        populations = list(nodes)
    else:
        populations = populations_list.split(",")

    group_keys = group_keys.split(",")
    group_keys += (len(populations)-len(group_keys)) * ["node_type_id"] #Extend the array to default values if not enough given
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
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

        cells_plotted = 0
        for color, (group_name, group_df) in zip(color_map, groupings):
            # if we selected a group and it's not in the list continue
            if group and group_name not in group.split(","):
                continue

            if "pos_x" not in group_df: #could also check model type == virtual
                continue #can't plot them if there isn't an xy coordinate (may be virtual)

            # if we exceed the max cells, stop plotting or limit
            if cells_plotted >= max_cells:
                continue
            if len(group_df) + cells_plotted > max_cells:
                total_remaining = max_cells - cells_plotted
                group_df = group_df[:total_remaining]
            cells_plotted += len(group_df)

            X = group_df["pos_x"]
            Y = group_df["pos_y"]
            Z = group_df["pos_z"]
            U = group_df.get("rotation_angle_xaxis") 
            V = group_df.get("rotation_angle_yaxis")
            W = group_df.get("rotation_angle_zaxis")
            if U is None:
                U = np.zeros(len(X))
            if V is None:
                V = np.zeros(len(Y))
            if W is None:
                W = np.zeros(len(Z))
            
            #Convert to arrow direction
            from scipy.spatial.transform import Rotation as R
            uvw = pd.DataFrame([U,V,W]).T
            init_vector = init_vector.split(',')
            init_vector = np.repeat([init_vector],len(X),axis=0)
            
            # To get the final cell orientation after rotation, 
            # you need to use function Rotaion.apply(init_vec), 
            # where init_vec is a vector of the initial orientation of a cell
            #rots = R.from_euler('xyz', uvw).apply(init_vector.astype(float))
            #rots = R.from_euler('xyz', pd.DataFrame([rots[:,0],rots[:,1],rots[:,2]]).T).as_rotvec().T

            rots = R.from_euler('zyx', uvw).apply(init_vector.astype(float)).T
            h = ax.quiver(X, Y, Z, rots[0],rots[1],rots[2],color=color,label=group_name, arrow_length_ratio = arrow_length_ratio, length=quiver_length)

            #h = ax.quiver(X, Y, Z, rots[0],rots[1],rots[2],color=color,label=group_name, arrow_length_ratio = arrow_length_ratio, length=quiver_length)
            ax.scatter(X,Y,Z,color=color,label=group_name)
            handles.append(h)
    if not handles:
        return
    plt.title(title)
    plt.legend(handles=handles)
    
    plt.draw()

    if save_file:
        plt.savefig(save_file)

    return

def plot_network_graph(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None,edge_property='model_template'):
    
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
    data, source_labels, target_labels = util.connection_graph_edge_types(nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop,edge_property=edge_property)

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
    connection_matrix(config=config_file,sources='all',targets='all',sids='pop_name',tids='pop_name',title='All Connections found', size_scalar=2, no_prepend_pop=True, synaptic_info='0')
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

def plot_basic_cell_info(config_file,notebook=0):
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
                    morphology=node['morphology'][i] if node.get('morphology') else ''
                    CELLS.append([node_type,pop_name,model_type,model_template,morphology,count])
                    count=1
            else:
                node_type=node_type_id[i]
                pop_name=node['pop_name'][i]
                model_type=node['model_type'][i]
                model_template=node['model_template'][i]
                morphology=node['morphology'][i] if node.get('morphology') else ''
                CELLS.append([node_type,pop_name,model_type,model_template,morphology,count])
                count=1
            df2 = pd.DataFrame(CELLS, columns = ["node_type","pop_name","model_type","model_template","morphology","count"])
            print(j+':')
            bio.append(j)
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
