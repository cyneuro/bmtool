import networkx as nx
import pandas as pd
import bmtool.util.util as u
import pandas as pd


def generate_graph(config,source,target):
    """
    returns a graph object
    config: A BMTK simulation config 
    source: network name 
    target: network name
    """
    nodes,edges = u.load_nodes_edges_from_config(config)
    nodes_source = nodes[source]
    nodes_target = nodes[target]
    if source != target:
        # Concatenate the DataFrames if source and target are different nodes
        nodes = pd.concat([nodes_source, nodes_target])
    else:
        nodes = nodes[source]
    edge_to_grap = source+"_to_"+target
    edges = edges[edge_to_grap]

    # Create an empty graph
    G = nx.DiGraph()

    # Add nodes to the graph with their positions and labels
    for index, node_data in nodes.iterrows():
        G.add_node(index, pos=(node_data['pos_x'], node_data['pos_y'], node_data['pos_z']), label=node_data['pop_name'])

    # Add edges to the graph
    for _, row in edges.iterrows():
        G.add_edge(row['source_node_id'], row['target_node_id'])

    return G

# import plotly.graph_objects as go
# def plot_graph(Graph=None,show_edges = False,title = None):
#     """
#     Generate an interactive plot of the network
#     Graph: A Graph object
#     show_edges: Boolean to show edges in graph plot
#     title: A string for the title of the graph
    
#     """

#     # Extract node positions
#     node_positions = nx.get_node_attributes(Graph, 'pos')
#     node_x = [data[0] for data in node_positions.values()]
#     node_y = [data[1] for data in node_positions.values()]
#     node_z = [data[2] for data in node_positions.values()]

#     # Create edge traces
#     edge_x = []
#     edge_y = []
#     edge_z = []
#     for edge in Graph.edges():
#         x0, y0, z0 = node_positions[edge[0]]
#         x1, y1, z1 = node_positions[edge[1]]
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])
#         edge_z.extend([z0, z1, None])

#     # Create edge trace
#     edge_trace = go.Scatter3d(
#         x=edge_x,
#         y=edge_y,
#         z=edge_z,
#         line=dict(width=1, color='#888'),
#         hoverinfo='none',
#         mode='lines',
#         opacity=0.2)

#     # Create node trace
#     node_trace = go.Scatter3d(
#         x=node_x,
#         y=node_y,
#         z=node_z,
#         mode='markers',
#         hoverinfo='text',
#         marker=dict(
#             showscale=True,
#             colorscale='YlGnBu',  # Adjust color scale here
#             reversescale=True,
#             color=[len(list(Graph.neighbors(node))) for node in Graph.nodes()],  # Assign color data here
#             size=5,  # Adjust the size of the nodes here
#             colorbar=dict(
#                 thickness=15,
#                 title='Node Connections',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             line_width=2,
#             cmin=0,  # Adjust color scale range here
#             cmax=max([len(list(Graph.neighbors(node))) for node in Graph.nodes()])   # Adjust color scale range here
#         ))

#     # Define hover text for nodes
#     node_hover_text = [f'Node ID: {node_id}<br>Population Name: {node_data["label"]}<br># of Connections: {len(list(Graph.neighbors(node_id)))}' for node_id, node_data in Graph.nodes(data=True)]
#     node_trace.hovertext = node_hover_text

#     # Create figure
#     if show_edges:
#         graph_prop = [edge_trace,node_trace]
#     else:
#         graph_prop = [node_trace]

#     if title == None:
#         title = '3D plot'
    
#     fig = go.Figure(data=graph_prop,
#                     layout=go.Layout(
#                         title=title,
#                         titlefont_size=16,
#                         showlegend=False,
#                         hovermode='closest',
#                         margin=dict(b=20, l=5, r=5, t=40),
#                         scene=dict(
#                             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                             zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
#                         ),
#                         width=800,
#                         height=800
#                     ))

#     # Show figure
#     fig.show()


def export_node_connections_to_csv(Graph, filename):
    """
    Generates a CSV file with node type and all outgoing connections that node has.
    
    Parameters:
    Graph: a DiGraph object (directed graph)
    filename: A string for the name of output, must end in .csv
    """
    # Create an empty dictionary to store the connections for each node
    node_connections = {}

    # Iterate over each node in the graph
    for node in Graph.nodes():
        # Initialize a dictionary to store the outgoing connections for the current node
        connections = {}
        node_label = Graph.nodes[node]['label']

        # Iterate over each presuccessor (ingoing neighbor) of the current node
        for successor in Graph.predecessors(node):
            # Get the label of the successor node
            successor_label = Graph.nodes[successor]['label']

            # Increment the connection count for the current node and successor label
            connections[f'{successor_label} incoming Connections'] = connections.get(f'{successor_label} incoming Connections', 0) + 1

        # Add the connections information for the current node to the dictionary
        connections['Node Label'] = node_label
        node_connections[node] = connections

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(node_connections).fillna(0).T

    # Reorder columns so that 'Node Label' is the leftmost column
    cols = df.columns.tolist()
    cols = ['Node Label'] + [col for col in cols if col != 'Node Label']
    df = df[cols]

    # Write the DataFrame to a CSV file
    df.to_csv(filename)