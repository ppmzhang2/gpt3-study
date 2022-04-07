"""generators"""
import logging

import click
import joblib

from ..models import prompt_predict
from ._utils import get_txt

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-path",
    type=click.STRING,
    required=True,
    help="flat file for fine-tune training",
)
@click.option(
    "--model-path",
    type=click.STRING,
    required=True,
    help="path of saved model",
)
@click.option(
    "--tokenizer-path",
    type=click.STRING,
    required=True,
    help="path of saved tokenizer",
)
@click.option(
    "--question",
    type=click.STRING,
    required=True,
    help="question to ask",
)
def prompt_generate(
    data_path: str,
    model_path: str,
    tokenizer_path: str,
    question: str,
):
    """fing-tune training"""
    tokenizer = joblib.load(tokenizer_path)
    model = joblib.load(model_path)

    seq = get_txt(data_path, tokenizer.eos_token)
    prompt = ''.join(seq)
    inputs = prompt + question
    LOGGER.info(f'{inputs=}')

    gen_tokens = prompt_predict(
        model,
        tokenizer,
        inputs,
        do_sample=True,
        temperature=0.6,
        max_new_tokens=20,
    )
    print(tokenizer.batch_decode(gen_tokens)[0])
