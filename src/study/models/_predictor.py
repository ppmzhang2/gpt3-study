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
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )
