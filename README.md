# GPT-3 Study

## Commands

Fine-tune training:

```sh
study fine-tune-train --data-path='./data/netflix_types.txt' \
    --valid-ratio=0.1 --model-path='./data/gpt_neo_3b_trained.model' \
    --tokenizer-path='./data/gpt_neo_3b_trained.tkn' \
    --model-name='EleutherAI/gpt-neo-2.7B'
```

In-context learning:

```sh
# sentiment classification
study prompt-generate --data-path='./data/prompt_twitter_cls.txt' \
    --model-path='./data/gpt_neo_3b.model' \
    --tokenizer-path='./data/gpt_neo_3b.tkn' \
    --question='Tweet: This new music video was incredible. Sentiment:'
# translation
study prompt-generate --data-path='./data/prompt_translate.txt' \
    --model-path='./data/gpt_neo_3b.model' \
    --tokenizer-path='./data/gpt_neo_3b.tkn' \
    --question='Me: Qui est le president des Etats-Unis? Her:'
# movie classification
study prompt-generate --data-path='./data/prompt_netflix_type.txt' \
    --model-path='./data/gpt_neo_3b.model' \
    --tokenizer-path='./data/gpt_neo_3b.tkn' \
    --question='Description: This documentary delves into the mystique behind the blues-rock trio and explores how the enigmatic band created their iconic look and sound. Type:'
```

## Reference

- https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
- https://www.johnfaben.com/blog/gpt-3-translations
- https://github.com/dredwardhyde/gpt-neo-fine-tuning-example/blob/main/gpt_neo.py
