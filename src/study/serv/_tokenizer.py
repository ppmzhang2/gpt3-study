"""tokenizer commands"""
import click
from transformers import AutoTokenizer

from ..tokenizer import Tokenizer


@click.command()
@click.option(
    "--model-name",
    type=click.STRING,
    required=True,
    help="model name",
)
@click.option(
    "--words",
    type=click.STRING,
    required=True,
    help="string to encode e.g. EleutherAI/gpt-neo-1.3B",
)
def encode_with_pretrained(model_name: str, words: str):
    """encode string as IDs with pre-trained model"""
    tkn = AutoTokenizer.from_pretrained(model_name)
    ids = tkn.encode(words)
    print(ids)


@click.command()
@click.option(
    "--words",
    type=click.STRING,
    required=True,
    help="string to encode",
)
def encode(words: str):
    """encode string as IDs with handmade Tokenizer"""
    ids = Tokenizer.encode(words)
    print(ids)


@click.command()
@click.option(
    "--words",
    type=click.STRING,
    required=True,
    help="string to encode",
)
def tokenize(words: str):
    """tokenize string with handmade Tokenizer"""
    ids = Tokenizer.str_to_tokens(words)
    print(ids)
