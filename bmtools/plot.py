"""
Want to be able to take multiple plot names in and plot them all at the same time, to save time
https://stackoverflow.com/questions/458209/is-there-a-way-to-detach-matplotlib-plots-so-that-the-computation-can-continue
"""
from . import util

import argparse,os,sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

use_description = """

Plot BMTK models easily.

python -m bmtools.plot 
"""

def conn_matrix(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None):
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
    data, source_labels, target_labels = util.connection_totals(nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop)

    if title == None or title=="":
        title = "Total Connections"

    plot_connection_info(data,source_labels,target_labels,title, save_file=save_file)
    return
    
def percent_conn_matrix(config=None,nodes=None,edges=None,title=None,populations=['hippocampus'], save_file=None):
    data, labels = util.percent_connectivity(nodes=None,edges=None,populations=populations)

    if title == None or title=="":
        title = "Percent Connectivity"

    plot_connection_info(data,labels,title, save_file=save_file)
    return


def divergence_conn_matrix(config=None,nodes=None,edges=None,title=None,sources=None, targets=None, sids=None, tids=None, no_prepend_pop=False,save_file=None,convergence=False):
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
    data, source_labels, target_labels = util.connection_divergence_average(nodes=None,edges=None,sources=sources,targets=targets,sids=sids,tids=tids,prepend_pop=not no_prepend_pop,convergence=convergence)

    
    #data, labels = util.connection_divergence_average(config=config,nodes=nodes,edges=edges,populations=populations)

    if title == None or title=="":
        if convergence:
            title = "Average Synaptic Convergence"
        else:
            title = "Average Synaptic Divergence"

    plot_connection_info(data,source_labels,target_labels,title, save_file=save_file)
    return

def raster(config=None,title=None,populations=['hippocampus']):
    conf = util.load_config(config)
    spikes_path = os.path.join(conf["output"]["output_dir"],conf["output"]["spikes_file"])
    nodes = util.load_nodes_from_config(config)
    plot_spikes(nodes,spikes_path)
    return

def plot_connection_info(data, source_labels,target_labels, title, save_file=None):
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    
    # We want to show all ticks...
    ax.set_xticks(list(np.arange(len(target_labels))))
    ax.set_yticks(list(np.arange(len(source_labels))))
    # ... and label them with the respective list entries
    ax.set_xticklabels(target_labels)
    ax.set_yticklabels(source_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(source_labels)):
        for j in range(len(target_labels)):
            text = ax.text(j, i, data[i, j],
                        ha="center", va="center", color="w")
    ax.set_ylabel('Source')
    ax.set_xlabel('Target')
    ax.set_title(title)
    fig.tight_layout()
    plt.draw()

    if save_file:
        plt.savefig(save_file)

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
    
    plt.savefig('raster3_after_pycvode_fixes.png')
    if save_file:
        plt.savefig(save_file)
    
    plt.show()
    
    return
    
def plot_3d_positions(**kwargs):
    import h5py
    #A = nodes_table(nodes_file='network/hippocampus_nodes.h5', population='hippocampus')

    #dset = np.array(A['positions'].values)
    #pos_df = pd.DataFrame(list(dset))
    #pos_ds = pos_df.values

    f = h5py.File('network/hippocampus_nodes.h5')
    pos = (f['nodes']['hippocampus']['0']['positions'])
    post = [list(i) for i in list(pos)]
    pos_ds = np.array(post)
    #print(pos_ds[:30,:])

    #inpTotal = 30 # EC
    #excTotal = 63 # CA3 principal
    #CA3oTotal = 8
    #CA3bTotal  = 8
    #DGexcTotal = 384 
    #DGbTotal = 32
    #DGhTotal = 32

    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.scatter(pos_ds[0:599,0],pos_ds[0:599,1],pos_ds[0:599,2],color='red')
    #ax.scatter(pos_ds[600:699,0],pos_ds[600:699,1],pos_ds[600:699,2],color='blue')
    ec = ax.scatter(pos_ds[0:29,0],pos_ds[0:29,1],pos_ds[0:29,2],color='red',label='EC')

    ca3e = ax.scatter(pos_ds[30:92,0],pos_ds[30:92,1],pos_ds[30:92,2],color='blue',label='CA3e')
    ca3o = ax.scatter(pos_ds[93:100,0],pos_ds[93:100,1],pos_ds[93:100,2],color='orange',label='CA3o')
    ca3b = ax.scatter(pos_ds[101:108,0],pos_ds[101:108,1],pos_ds[101:108,2],color='yellow',label='CA3b')

    dgg = ax.scatter(pos_ds[109:492,0],pos_ds[109:492,1],pos_ds[109:492,2],color='green',label='DGg')
    dgh = ax.scatter(pos_ds[493:524,0],pos_ds[493:524,1],pos_ds[493:524,2],color='black',label='DGh')
    dgb = ax.scatter(pos_ds[525:556,0],pos_ds[525:556,1],pos_ds[525:556,2],color='purple',label='DGb')

    plt.title('Hippocampus')
    plt.legend(handles=[ec,ca3e,ca3o,ca3b,dgg,dgh,dgb])
    plt.draw()
    
def new_plot_3d_positions(**kwargs):
    
    return

if __name__ == '__main__':
    parser = util.get_argparse(use_description)
    
    functions = {}
    #TODO: Can add an interactive mode function that loads nodes and edges before hand to save time
    base_params = [
        {
            "dest":["--config"],
            "help":"simulation config file (default: simulation_config.json)",
            "default":"simulation_config.json"
        },
        {
            "dest":["--no_display"],
            "action":"store_true",
            "help":"When set there will be no plot displayed, useful for saving plots",
            "default":False
        }
    ]
    connection_params = base_params + [
        {
            "dest":["--title"],
            "help":"change the plot's title"
        },
        {
            "dest":["--save_file"],
            "help":"save plot to path supplied",
            "default":None
        },
        {
            "dest":["--sources"],
            "help":"comma separated list of source node types [default:all]",
            "default":"all"
        },
        {
            "dest":["--targets"],
            "help":"comma separated list of target node types [default:all]",
            "default":"all"
        },
        {
            "dest":["--sids"],
            "help":"comma separated list of source node identifiers [default:node_type_id]",
            "default":None
        },
        {
            "dest":["--tids"],
            "help":"comma separated list of target node identifiers [default:node_type_id]",
            "default":None
        },
        {
            "dest":["--no_prepend_pop"],
            "help":"When set don't prepend the population name to the unique ids [default:False]",
            "action":"store_true",
            "default":False
        }
    ]

    functions["positions"] = {
        "function":plot_3d_positions, 
        "description":"Plot cell positions for a given set of populations",
        "args":
        [
            {
                "dest":["--title"],
                "help":"change the plot's title"
            },
            {
                "dest":["--populations"],
                "nargs":"+",
                "required":False
            }
        ]
    }
    conn_args = connection_params[:]
    functions["connection_total"] = {
        "function":conn_matrix, 
        "description":"Plot the total connection matrix for a given set of populations",
        "args":conn_args
    }
    perc_args = connection_params[:]
    functions["connection_percent"] = {
        "function":percent_conn_matrix, 
        "description":"Plot the connection percentage matrix for a given set of populations",
        "disabled":True,
        "args":perc_args
    }
    div_args = connection_params[:]
    functions["connection_divergence"] = {
        "function":divergence_conn_matrix, 
        "description":"Plot the connection percentage matrix for a given set of populations",
        "disabled":False,
        "args": div_args       
    }
    conv_args = connection_params[:]
    functions["connection_convergence"] = {
        "function":divergence_conn_matrix, 
        "description":"Plot the connection convergence matrix for a given set of populations",
        "disabled":False,
        "args": conv_args +
        [
            {
                "dest":["--convergence"],
                "default":True,
                "help":argparse.SUPPRESS
            }
        ]
    }
    functions["raster"] = {
        "function":raster, 
        "description":"Plot the spike raster for a given set of populations",
        "args": base_params + [
            {
                "dest":["--title"],
                "help":"change the plot's title"
            }
        ]
    }
    
    #parser.add_argument('--config',help="simulation config file (default: simulation_config.json) [MUST be first argument]",default='simulation_config.json')
    #parser.add_argument('--no-display', action="store_true", default=False, help="When set there will be no plot displayed, useful for saving plots")
    subparser = parser.add_subparsers()
    for k in list(functions):
        if functions[k].get("disabled") and functions[k]["disabled"]:
            continue
        sp = subparser.add_parser(k,help=functions[k]["description"])
        sp.add_argument('--handler', default=functions[k]["function"], help=argparse.SUPPRESS)
        if functions[k].get("args"):
            for a in functions[k]["args"]:
                #import pdb
                #pdb.set_trace()
                dest = a["dest"]
                #a.pop('dest',None)
                b = {key:value for key, value in a.items() if key not in ["dest"]}
                sp.add_argument(*dest,**b)

    if not len(sys.argv) > 1:
        parser.print_help()
    else:
        args = parser.parse_args() 
        v = vars(args)
        handling_func = v['handler']
        no_display = False 
        if v.get('no_display'):
            no_display = v['no_display']
        v.pop('handler', None)
        v.pop('no_display',None)
        handling_func(**v)
    
        if not no_display:
            plt.show()
