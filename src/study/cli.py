"""all commands here"""
import click

from .serv import fine_tune_train


@click.group()
def cli():
    """all clicks here"""


cli.add_command(fine_tune_train)
