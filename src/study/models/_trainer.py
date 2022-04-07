"""model training functions
"""
import torch
from transformers import Trainer
from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel

from ..datasets import SeqTxtDataset


def train_seq_model(
    model: PreTrainedModel,
    training_args: TrainingArguments,
    train_dataset: SeqTxtDataset,
    eval_dataset: SeqTxtDataset,
) -> PreTrainedModel:
    """train a sequence model"""
    Trainer(model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=lambda data: {
                'input_ids': torch.stack([f[0] for f in data]),
                'attention_mask': torch.stack([f[1] for f in data]),
                'labels': torch.stack([f[0] for f in data])
            }).train()
    return model
