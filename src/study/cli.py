"""all commands here"""
import click

from .serv import decode
from .serv import encode
from .serv import encode_with_pretrained
from .serv import fine_tune_train
from .serv import prompt_generate
from .serv import tokenize


@click.group()
def cli():
    """all clicks here"""


cli.add_command(decode)
cli.add_command(encode)
cli.add_command(encode_with_pretrained)
cli.add_command(fine_tune_train)
cli.add_command(prompt_generate)
cli.add_command(tokenize)
