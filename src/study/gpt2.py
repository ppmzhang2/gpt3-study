"""hugging-face demo

Example of Causal Language Model using pipeline
"""
import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import top_k_top_p_filtering

# Creating tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Creating context for sequence
context_sequence = "I have never watched anything like this, and it was"

# Applying tokenizer on sequence
tokens = tokenizer.encode(context_sequence, return_tensors="pt")
last_logits = model(tokens).logits[:, -1, :]

# Applying top k top p filtering
flt = top_k_top_p_filtering(last_logits, top_k=50, top_p=1.0)

# Finding probabilities using softmax function
probabilities = nn.functional.softmax(flt, dim=-1)

# Applying multinomial
final_token = torch.multinomial(probabilities, num_samples=1)

# Applying cat function
output = torch.cat([tokens, final_token], dim=-1)

# Decoding
answer = tokenizer.decode(output.tolist()[0])

# Printing answer
print(answer)
