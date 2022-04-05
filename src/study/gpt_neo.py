"""GPT-neo demo

reference:

- https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
- https://www.johnfaben.com/blog/gpt-3-translations

"""
import joblib
from transformers import GPT2Tokenizer
from transformers import GPTNeoForCausalLM

model = (joblib.load('./models/model.pkl')
         or GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B"))
tokenizer = (joblib.load('./models/tokenizer.pkl')
             or GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B"))

prompt = """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been great so far"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredible"
Sentiment:"""

prompt = """sea otter => loutre de mer
###
peppermint => menthe poivrée
###
plush giraffe => girafe peluche
###
cheese =>"""

prompt = """
Me: Le singe est dans l'arbre
Her: The monkey is in the tree
###
Me: la plume de ma tante est sur la table
Her: My aunt's pen is on the table
###
Me: j'aime bien le jambon
Her: I like the chair
###
Me: Qu'est-ce que c'est que ca?
Her: What do you mean?
###
Me: Comment tu t'appeles?
Her: I am called Bob
###
Me: Où est le garçon?
Her: Where is the boy?
###
Me: Qui est le president des Etats-Unis?
Her:
"""

# tokenizer("###", return_tensors='pt').input_ids = 21017

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.6,
    max_new_tokens=20,
    eos_token_id=21017,
    pad_token_id=21017,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)
