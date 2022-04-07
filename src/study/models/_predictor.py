"""generate outputs from a sequence model"""
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def prompt_predict(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    **kwargs,
):
    """generate outputs by input prompt"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return model.generate(
        input_ids,
        bos_token=tokenizer.bos_token,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
        **kwargs,
    )
