import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from .. import util

def cell_rotation_3d(**kwargs):
    populations_list = kwargs["populations"]
    config = kwargs["config"]
    group_keys = kwargs["group_by"]
    title = kwargs["title"]
    save_file = kwargs["save_file"]
    quiver_length = kwargs["quiver_length"]
    arrow_length_ratio = kwargs["arrow_length_ratio"]
    group = kwargs["group"]
    max_cells = kwargs["max_cells"]

    nodes = util.load_nodes_from_config(config)

    if 'all' in populations_list:
        populations = list(nodes)
    else:
        populations = populations_list.split(",")

    group_keys = group_keys.split(",")
    group_keys += (len(populations)-len(group_keys)) * ["node_type_id"] #Extend the array to default values if not enough given
    fig = plt.figure(figsize=(10,10))
    # ax = Axes3D(fig) # Old way
    ax = fig.add_subplot(111, projection='3d')
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
            rots = R.from_euler('zyx', uvw).as_rotvec().T
            h = ax.quiver(X, Y, Z, rots[0],rots[1],rots[2],color=color,label=group_name, arrow_length_ratio = arrow_length_ratio, length=quiver_length)
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
