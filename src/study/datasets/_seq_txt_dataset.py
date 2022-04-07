"""text sequence dataset"""
from collections.abc import Iterable

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class SeqTxtDataset(Dataset):
    """Text Sequence Dataset"""

    def __init__(
        self,
        txt_list: Iterable[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer(
                # add only the EOS token according to generator performance
                txt + tokenizer.eos_token,
                # tokenizer.bos_token + txt + tokenizer.eos_token,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(
                torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
