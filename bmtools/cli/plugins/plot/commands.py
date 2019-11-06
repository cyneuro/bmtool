import click
import logging
import os

from clint.textui import puts, colored, indent
from .plot import conn_matrix, percent_conn_matrix, divergence_conn_matrix
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
    
@connection.command('total',help="Plot the total connection matrix for a given set of populations")
@click.pass_context
def connection_total(ctx):
    conn_matrix(ctx.obj['config'],**ctx.obj['connection'])
    if ctx.obj['display']:
        plt.show()

@connection.command('percent',help="Plot the connection percentage matrix for a given set of populations")
@click.pass_context
def connection_percent(ctx):
    percent_conn_matrix(ctx.obj['config'],**ctx.obj['connection'])
    if ctx.obj['display']:
        plt.show()

@connection.command('divergence',help="Plot the connection divergence matrix for a given set of populations")
@click.pass_context
def connection_divergence(ctx):
    divergence_conn_matrix(ctx.obj['config'],**ctx.obj['connection'])
    if ctx.obj['display']:
        plt.show()

@connection.command('convergence',help="Plot the connection convergence matrix for a given set of populations")
@click.pass_context
def connection_convergence(ctx):
    divergence_conn_matrix(ctx.obj['config'],**ctx.obj['connection'],convergence=True)
    if ctx.obj['display']:
        plt.show()

cli.add_command(connection)

if __name__ == "__main__":
    cli()
