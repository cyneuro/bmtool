"""
Want to be able to take multiple plot names in and plot them all at the same time, to save time
https://stackoverflow.com/questions/458209/is-there-a-way-to-detach-matplotlib-plots-so-that-the-computation-can-continue
"""
from . import util

import argparse,os,sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from bmtk.analyzer.cell_vars import CellVarsFile, missing_units
from bmtk.analyzer.utils import listify

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

    fig = plt.figure()
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
    
    plt.title(title)
    plt.legend(handles=handles)
    plt.draw()

    if save_file:
        plt.savefig(save_file)

    return

def plot_3d_positions_old(**kwargs):
    import h5py

    f = h5py.File('network/hippocampus_nodes.h5')
    pos = (f['nodes']['hippocampus']['0']['positions'])
    post = [list(i) for i in list(pos)]
    pos_ds = np.array(post)

    fig = plt.figure()
    ax = Axes3D(fig)

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
            "dest":["--no-display"],
            "action":"store_true",
            "help":"When set there will be no plot displayed, useful for saving plots",
            "default":False
        }
    ]
    connection_params = base_params + [
        {
            "dest":["--title"],
            "default":None,
            "help":"change the plot's title"
        },
        {
            "dest":["--save-file"],
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
            "dest":["--no-prepend-pop"],
            "help":"When set don't prepend the population name to the unique ids [default:False]",
            "action":"store_true",
            "default":False
        }
    ]

    functions["positions"] = {
        "function":plot_3d_positions, 
        "description":"Plot cell positions for a given set of populations",
        "args": base_params + 
        [
            {
                "dest":["--title"],
                "help":"change the plot's title",
                "default":"Cell 3D Positions"
            },
            {
                "dest":["--populations"],
                "help":"comma separated list of populations to plot (default:all)",
                "default":"all",
                "required":False
            },
            {
                "dest":["--group-by"],
                "default":"node_type_id",
                "help":"comma separated list of identifiers (default: node_type_id) (pop_name is a good one)",
                "required":False
            },
            {
                "dest":["--save-file"],
                "help":"save plot to path supplied (default:None)",
                "default":None
            }
        ]
    }
    functions["positions-old"] = {
        "function":plot_3d_positions_old, 
        "description":"Plot cell positions for hipp model",
        "disabled":True,
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
    functions["connection-total"] = {
        "function":conn_matrix, 
        "description":"Plot the total connection matrix for a given set of populations",
        "args":conn_args
    }
    perc_args = connection_params[:]
    functions["connection-percent"] = {
        "function":percent_conn_matrix, 
        "description":"Plot the connection percentage matrix for a given set of populations",
        "disabled":True,
        "args":perc_args
    }
    div_args = connection_params[:]
    functions["connection-divergence"] = {
        "function":divergence_conn_matrix, 
        "description":"Plot the connection percentage matrix for a given set of populations",
        "disabled":False,
        "args": div_args       
    }
    conv_args = connection_params[:]
    functions["connection-convergence"] = {
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
    edge_hist_args = connection_params[:]
    functions["edge-histogram-matrix"] = {
        "function":edge_histogram_matrix,
        "description":"Plot the connection weight matrix for a given set of populations",
        "disabled":False,
        "args": edge_hist_args +
        [
            {
                "dest":["--edge-property"],
                "default":"syn_weight",
                "help":"Parameter you want to plot (default:syn_weight)"
            },
            {
                "dest":["--report"],
                "default":None,
                "help":"For variables that were collected post simulation run, specify the report the variable is contained in, specified in your simulation config (default:None)"
            },
            {
                "dest":["--time"],
                "default":-1,
                "help":"Time in (ms) that you want the data point to be collected from. Only used in conjunction with the --report parameter."
            },
            {
                "dest":["--time-compare"],
                "default":None,
                "help":"Time in (ms) that you want to compare with the --time parameter. Requires --time and --report parameters."
            }
        ]
    }

    graph_args = connection_params[:]
    functions["network-graph"] = {
        "function":plot_network_graph, 
        "description":"Plot the connection graph for supplied targets (default:all)",
        "disabled":False,
        "args": graph_args +
        [
            {
                "dest":["--edge-property"],
                "default":'model_template',
                "help":"Edge property to define connections (default:model_template)"
            }
        ]
    }
    functions["raster"] = {
        "function":raster, 
        "description":"Plot the spike raster for a given population",
        "args": base_params + [
            {
                "dest":["--title"],
                "help":"change the plot's title"
            },
            {
                "dest":["--population"],
                "help":"population name"
            },
            {
                "dest":["--group-key"],
                "help":"change key to group cells by (default: pop_name)",
                "default":"pop_name"
            }
        ]
    }
    functions["raster-old"] = {
        "function":raster_old,
        "disabled":True, 
        "description":"Plot the spike raster for hipp model",
        "args": base_params + [
            {
                "dest":["--title"],
                "help":"change the plot's title"
            }
        ]
    }
    functions["report"] = {
        "function":plot_report_default,
        "description":"Plot the specified report using BMTK's default report plotter",
        "args": base_params + [
            {
                "dest":["--report-name"],
                "help":"Name of the report specified in your simulation config you want to consider",
                "default":None
            },
            {
                "dest":["--variables"],
                "help":"Comma separated list of variables to plot",
                "default":None
            },
            {
                "dest":["--gids"],
                "help":"Cell numbers you want to plot",
                "default":None
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
                dest = a["dest"]
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
