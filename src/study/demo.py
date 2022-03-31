"""a demo"""
import openai

from . import cfg


def completion_task(query: str) -> str:
    """get completion text"""
    openai.api_key = cfg.OPENAI_KEY
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=query,
        top_p=1,
    )
    return response.choices[0].text
