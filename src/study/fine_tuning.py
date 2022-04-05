"""fine-tuning
https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo.py
"""
import pandas as pd
import torch
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import IntervalStrategy
from transformers import TrainingArguments

from .datasets import SeqTxtDataset
from .models import prompt_predict
from .models import train_seq_model

_BOS_TOKEN = '<|bos|>'
_EOS_TOKEN = '<|eos|>'
_PAD_TOKEN = '<|pad|>'

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neo-1.3B",
    bos_token=_BOS_TOKEN,
    eos_token=_EOS_TOKEN,
    pad_token=_PAD_TOKEN,
)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
model.resize_token_embeddings(len(tokenizer))

descriptions = pd.read_csv('./netflix_titles.csv')['description']

max_length = max(
    [len(tokenizer.encode(description)) for description in descriptions])

dataset = SeqTxtDataset(descriptions, tokenizer, max_length=max_length)
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(
    dataset, [train_size, len(dataset) - train_size])
training_args = TrainingArguments(
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

model = train_seq_model(model, training_args, train_dataset, val_dataset)

sample_outputs = prompt_predict(
    model,
    tokenizer,
    _BOS_TOKEN,
    do_sample=True,
    top_k=50,
    max_length=300,
    top_p=0.95,
    temperature=1.9,
    num_return_sequences=20,
)
for i, sample_output in enumerate(sample_outputs):
    print(f"{i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")
# gen_text = tokenizer.batch_decode(gen_tokens)[0]
# print(gen_text)
