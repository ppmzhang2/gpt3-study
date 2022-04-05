"""fine-tune training
https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo.py
"""
import csv

import click
import joblib
import torch
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import IntervalStrategy
from transformers import TrainingArguments

from ..datasets import SeqTxtDataset
from ..models import train_seq_model

_BOS_TOKEN = '<|bos|>'
_EOS_TOKEN = '<|eos|>'
_PAD_TOKEN = '<|pad|>'

_training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    logging_steps=5000,
    save_strategy=IntervalStrategy.NO,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
)

torch.manual_seed(42)


def _get_data(pth: str) -> list[str]:
    with open(pth, newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        return [i[0] for i in csv_reader]


@click.command()
@click.option(
    "--data-path",
    type=click.STRING,
    required=True,
    help="flat file for fine-tune training",
)
@click.option(
    "--valid-ratio",
    type=click.FLOAT,
    required=True,
)
@click.option(
    "--model-path",
    type=click.STRING,
    required=True,
    help="path to save trained model",
)
@click.option(
    "--model-name",
    type=click.STRING,
    required=True,
    default='EleutherAI/gpt-neo-1.3B',
)
def fine_tune_train(
    data_path: str,
    valid_ratio: float,
    model_path: str,
    model_name: str,
):
    """fing-tune training"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        bos_token=_BOS_TOKEN,
        eos_token=_EOS_TOKEN,
        pad_token=_PAD_TOKEN,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    data = _get_data(data_path)
    dataset = SeqTxtDataset(
        data,
        tokenizer,
        max_length=max([len(i) for i in data]),
    )
    val_size = int(valid_ratio * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_size, val_size])
    model = train_seq_model(model, _training_args, train_dataset, val_dataset)
    joblib.dump(model, model_path)
