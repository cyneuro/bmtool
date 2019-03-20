"""
Want to be able to take multiple plot names in and plot them all at the same time, to save time
https://stackoverflow.com/questions/458209/is-there-a-way-to-detach-matplotlib-plots-so-that-the-computation-can-continue
"""
from . import util

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

use_description = """
Plot BMTK models easily.

python -m bmtool.plot 
"""

def conn_matrix(nodes=None,edges=None,title="Total Connections",populations=['hippocampus']):
    data, labels = util.connection_totals(nodes=None,edges=None,populations=populations)
    plot_connection_info(data,labels,title)
    return
    
    
def plot_connection_info(data, labels, title):
    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, data[i, j],
                        ha="center", va="center", color="w")
    ax.set_ylabel('Source')
    ax.set_xlabel('Target')
    ax.set_title(title)
    fig.tight_layout()
    plt.draw()
    return
    
def plot_spikes(cells_file, cell_models_file, spikes_file, population=None):
    
    import h5py
    cm_df = pd.read_csv(cell_models_file, sep=' ')
    cm_df.set_index('node_type_id', inplace=True)

    cells_h5 = h5py.File(cells_file, 'r')
    # TODO: Use sonata api
    if population is None:
        if len(cells_h5['/nodes']) > 1:
            raise Exception('Multiple populations in nodes file. Please specify one to plot using population param')
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
    
    plt.show()
    
    return
    
def plot_3d_positions():
    import h5py
    #A = nodes_table(nodes_file='network/hippocampus_nodes.h5', population='hippocampus')

    #dset = np.array(A['positions'].values)
    #pos_df = pd.DataFrame(list(dset))
    #pos_ds = pos_df.values

    f = h5py.File('network/hippocampus_nodes.h5')
    pos = (f['nodes']['hippocampus']['0']['positions'])
    post = [list(i) for i in list(pos)]
    pos_ds = np.array(post)

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
    
    
if __name__ == '__main__':
    parser = util.get_argparse(use_description)

    functions = {}
    functions["positions"] = {"func":conn_matrix, "description":"yeah"}
    functions["con_matrix"] = {"func":plot_3d_positions, "description":"yeah"}
    functions["raster"] = {"func":plot_spikes, "description":"yeah"
    
    parser.add_argument('--config',help='simulation config file (default: simulation_config.json)',default='simulation_config.json')
    for k in list(functions):
        parser.add_argument('--'+k,action='store_true',help=functions[k]["description"])
    #parser.add_argument('--multiple', nargs='+', help='display multiple default plots quickly',choices=list(functions))
    
    util.verify_parse(parser)
    args = parser.parse_args()
    
    for key in list(functions):
        if(eval("args."+key)):
            functions[key]["func"]()
                
    plt.show()