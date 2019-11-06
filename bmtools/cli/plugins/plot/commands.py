import click
import logging
import os

from clint.textui import puts, colored, indent
from .plot import (conn_matrix, percent_conn_matrix,
                divergence_conn_matrix,plot_3d_positions,
                edge_histogram_matrix,plot_network_graph,
                raster,plot_report_default)
import matplotlib.pyplot as plt

@click.group('plot')
@click.option('--config', type=click.Path(), default='./simulation_config.json', help='Configuration file to use, default: "simulation_config.json"')
@click.option('--no-display', is_flag=True, default=False, help='When set there will be no plot displayed, useful for saving plots')
@click.pass_context
def cli(ctx, config, no_display):

    config_path = os.path.abspath(os.path.expanduser(config)).replace("\\","/")
    if not os.path.exists(config_path):
        click.echo(colored.red("Config file not found: " + config))

    ctx.obj["config"] = config_path
    ctx.obj["display"] = not no_display


@click.group('connection', help='Display information related to neuron connections')
@click.option('--title', type=click.STRING, default=None, help="change the plot's title")
@click.option('--save-file', type=click.STRING, default=None, help="save plot to path supplied")
@click.option('--sources', type=click.STRING, default='all', help="comma separated list of source node types [default:all]")
@click.option('--targets', type=click.STRING, default='all', help="comma separated list of target node types [default:all]")
@click.option('--sids', type=click.STRING, default=None, help="comma separated list of source node identifiers [default:node_type_id]")
@click.option('--tids', type=click.STRING, default=None, help="comma separated list of target node identifiers [default:node_type_id]")
@click.option('--no-prepend-pop', is_flag=True, default=False, help="When set don't prepend the population name to the unique ids [default:False]")
@click.pass_context
def connection(ctx,title,save_file,sources,targets,sids,tids,no_prepend_pop):

    ctx.obj["connection"] = {
        'title':title,
        'save_file':save_file,
        'sources':sources,
        'targets':targets,
        'sids':sids,
        'tids':tids,
        'no_prepend_pop':no_prepend_pop
    }
    
@connection.command('total',help="total connection matrix for a given set of populations")
@click.pass_context
def connection_total(ctx):
    conn_matrix(ctx.obj['config'],**ctx.obj['connection'])
    if ctx.obj['display']:
        plt.show()

@connection.command('percent',help="percentage matrix for a given set of populations")
@click.pass_context
def connection_percent(ctx):
    percent_conn_matrix(ctx.obj['config'],**ctx.obj['connection'])
    if ctx.obj['display']:
        plt.show()

@connection.command('divergence',help="divergence matrix for a given set of populations")
@click.pass_context
def connection_divergence(ctx):
    divergence_conn_matrix(ctx.obj['config'],**ctx.obj['connection'])
    if ctx.obj['display']:
        plt.show()

@connection.command('convergence',help="convergence matrix for a given set of populations")
@click.pass_context
def connection_convergence(ctx):
    divergence_conn_matrix(ctx.obj['config'],**ctx.obj['connection'],convergence=True)
    if ctx.obj['display']:
        plt.show()

@connection.command('property-histogram-matrix',help="connection property matrix for a given set of populations")
@click.option('--edge-property', type=click.STRING, default='syn_weight', help="Parameter you want to plot (default:syn_weight)")
@click.option('--report', type=click.STRING, default=None, help="For variables that were collected post simulation run, specify the report the variable is contained in, specified in your simulation config (default:None)")
@click.option('--time', type=click.STRING, default=-1, help="Time in (ms) that you want the data point to be collected from. Only used in conjunction with the --report parameter.")
@click.option('--time-compare', type=click.STRING, default=None, help="Time in (ms) that you want to compare with the --time parameter. Requires --time and --report parameters.")
@click.pass_context
def connection_property_histogram_matrix(ctx, edge_property, report, time, time_compare):
    edge_histogram_matrix(config = ctx.obj['config'],
                        **ctx.obj['connection'],
                        edge_property=edge_property,
                        report=report,
                        time=time,
                        time_compare=time_compare)
    if ctx.obj['display']:
        plt.show()

@connection.command('network-graph',help="connection graph for supplied targets (default:all)")
@click.option('--edge-property', type=click.STRING, default='model_template', help="Edge property to define connections [default:model_template]")
@click.pass_context
def connection_network_graph(ctx,edge_property):
    plot_network_graph(ctx.obj['config'],**ctx.obj['connection'],edge_property=edge_property)
    if ctx.obj['display']:
        plt.show()

@cli.command('positions', help="Plot cell positions for a given set of populations")
@click.option('--title', type=click.STRING, default='Cell 3D Positions', help="change the plot's title")
@click.option('--populations', type=click.STRING, default='all', help="comma separated list of populations to plot [default:all]")
@click.option('--group-by', type=click.STRING, default='node_type_id', help="comma separated list of identifiers [default: node_type_id] (pop_name is a good one)")
@click.option('--save-file', type=click.STRING, default=None, help="save plot to path supplied [default:None]")
@click.pass_context
def plot_positions(ctx, title, populations, group_by, save_file):
    plot_3d_positions(config=ctx.obj['config'],
                    title=title,
                    populations=populations,
                    group_by=group_by,
                    save_file=save_file)
    if ctx.obj['display']:
        plt.show()

@cli.command('raster', help="Plot the spike raster for a given population")
@click.option('--title', type=click.STRING, default='Raster Plot', help="change the plot's title")
@click.option('--population', type=click.STRING, default=None, help="population name")
@click.option('--group-key', type=click.STRING, default='pop_name', help="change key to group cells by [default: pop_name]")
@click.pass_context
def plot_raster(ctx, title, population, group_key):
    raster(config=ctx.obj['config'],
                    title=title,
                    population=population,
                    group_key=group_key)
    if ctx.obj['display']:
        plt.show()

@cli.command('report', help="Plot the specified report using BMTK's default report plotter")
@click.option('--report-name', type=click.STRING, default=None, help="Name of the report specified in your simulation config you want to consider")
@click.option('--variables', type=click.STRING, default=None, help="Comma separated list of variables to plot")
@click.option('--gids', type=click.STRING, default=None, help="Cell numbers you want to plot")
@click.pass_context
def plot_report(ctx, report_name, variables, gids):
    plot_report_default(config=ctx.obj['config'],
                    report_name=report_name,
                    variables=variables,
                    gids=gids)
    if ctx.obj['display']:
        plt.show()

cli.add_command(connection)

if __name__ == "__main__":
    cli()
