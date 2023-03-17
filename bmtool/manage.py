import click
import logging
import os

from clint.textui import puts, colored, indent

from .debug import commands as debug_commands
from . import plot_commands
from .util import commands as util_commands

@click.group()
@click.option('--verbose', is_flag=True, default=False, help='Verbose printing')
@click.pass_context
def cli(ctx, verbose):
    
    

    if verbose:
        click.echo(colored.green('Verbose printing mode is on.'))

    ctx_obj = {}
    ctx_obj["verbose"] = verbose

    ctx.obj = ctx_obj


cli.add_command(debug_commands.cli)
cli.add_command(plot_commands.cli)
cli.add_command(util_commands.cli)

if __name__ == "__main__":
    cli()