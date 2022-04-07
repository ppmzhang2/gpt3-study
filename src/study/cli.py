"""all commands here"""
import click

from .serv import fine_tune_train
from .serv import prompt_generate


@click.group()
def cli():
    """all clicks here"""


cli.add_command(fine_tune_train)
cli.add_command(prompt_generate)
