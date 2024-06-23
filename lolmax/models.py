import getpass
import os

from langchain.chat_models.base import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity

import copy


def read_file(path):
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return open(path, "r").read().rstrip()


def fetch_api_token(env_keys, paths, prompt=False):
    accessors = []
    accessors.extend([{
        "method": "env",
        "key": k,
        "fn": lambda: os.getenv(k)
    } for k in env_keys])
    accessors.extend([{
        "method": "file",
        "path": k,
        "fn": lambda: read_file(k)
    } for k in paths])
    if prompt:
        accessors.append({
            "method": "prompt",
            "fn": getpass.getpass("Enter your API token:")
        })

    api_key = None
    for accessor in accessors:
        api_key = accessor["fn"]()
        if api_key is not None:
            return api_key


MODELS = [
    {
        "id":
        "gpt-4-default",
        "model":
        "gpt-4",
        "openai_api_key":
        fetch_api_token(["OPENAI_API_KEY"], ["~/.config/openai.token"]),
    },
    {
        "id":
        "anthropic",
        "model":
        "claude-3-sonnet-20240229",
        "api_key":
        fetch_api_token(["ANTHROPIC_API_KEY"], ["~/.config/anthropic.token"]),
    },
    {
        "id":
        "mistral",
        "model":
        "mistral-small-latest",
        "api_key":
        fetch_api_token(["MISTRAL_API_KEY"], ["~/.config/mistralai.token"]),
    },
    {
        # This one seems to require some manual intervention for project setup
        "id": "gemini-default",
        "model": "gemini",
        # Nope...nice try though
        # need to RTFM: https://python.langchain.com/v0.2/docs/integrations/chat/google_vertex_ai_palm/
        # "google_api_key":
        # fetch_api_token(["GOOGLE_API_KEY"], ["~/.config/google_gemini.token"])
    },
]
MODELS_BY_ID = {m["id"]: m for m in MODELS}


def load_model_by_id(model_id):
    config = MODELS_BY_ID.get(model_id)
    if not config:
        raise f"No config found for model_id {model_id}"

    config = copy.copy(config)
    config.pop("id")

    return init_chat_model(**config)


def openai_chat(api_key=None, model="gpt-4"):
    api_key = api_key or fetch_api_token(["OPENAI_API_KEY"],
                                         ["~/.config/openai.token"])
    return ChatOpenAI(api_key=api_key, model=model)


def gemini_chat(api_key: str | None = None, model: str = "gemini-pro"):
    api_key = api_key or fetch_api_token(["GOOGLE_API_KEY"],
                                         ["~/.config/google_gemini.token"])
    # HACK: this client seems to require passing api key by environment variable, so we mutate the environment
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    return ChatGoogleGenerativeAI(model=model)


# Supported models: https://docs.perplexity.ai/docs/model-cards
def perplexity_chat(api_key: str | None = None,
                    model: str = "llama-3-sonar-large-32k-chat"):
    api_key = api_key or fetch_api_token(["PPLX_API_KEY"],
                                         ["~/.config/perplexity.token"])
    return ChatPerplexity(pplx_api_key=api_key, model=model)


# Available models (as of 2024.05.29):
# + claude-3-opus-20240229
# + claude-3-sonnet-20240229
# + claude-3-haiku-20240307
def anthropic_chat(api_key=None, model="claude-3-sonnet-20240229"):
    api_key = api_key or fetch_api_token(["ANTHROPIC_API_KEY"],
                                         ["~/.config/anthropic.token"])
    return ChatAnthropic(api_key=api_key, model_name=model)


def mistral_chat(api_key=None, model="mistral-small"):
    api_key = api_key or fetch_api_token(["MISTRAL_API_KEY"],
                                         ["~/.config/mistralai.token"])
    return ChatMistralAI(api_key=api_key, model_name=model)
