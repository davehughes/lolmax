import getpass
import os

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI


def read_file(path):
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return open(path, "r").read().rstrip()


def fetch_api_token(env_keys, paths, prompt=False):
    accessors = []
    accessors.extend(
        [{"method": "env", "key": k, "fn": lambda: os.getenv(k)} for k in env_keys]
    )
    accessors.extend(
        [{"method": "file", "path": k, "fn": lambda: read_file(k)} for k in paths]
    )
    if prompt:
        accessors.append(
            {"method": "prompt", "fn": getpass.getpass("Enter your API token:")}
        )

    api_key = None
    for accessor in accessors:
        api_key = accessor["fn"]()
        if api_key is not None:
            return api_key


def print_chat_stream(client, prompt):
    for chunk in client.stream(prompt):
        print(chunk)


def openai_chat(api_key=None, model="gpt-4"):
    api_key = api_key or fetch_api_token(["OPENAI_API_KEY"], ["~/.config/openai.token"])
    return ChatOpenAI(api_key=api_key, model=model)


def gemini_chat(api_key: str | None = None, model: str = "gemini-pro"):
    api_key = api_key or fetch_api_token(
        ["GOOGLE_API_KEY"], ["~/.config/google_gemini.token"]
    )
    # HACK: this client seems to require passing api key by environment variable, so we mutate the environment
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    return ChatGoogleGenerativeAI(model=model)


# Available models (as of 2024.05.29):
# + claude-3-opus-20240229
# + claude-3-sonnet-20240229
# + claude-3-haiku-20240307
def anthropic_chat(api_key=None, model="claude-3-sonnet-20240229"):
    api_key = api_key or fetch_api_token(
        ["ANTHROPIC_API_KEY"], ["~/.config/anthropic.token"]
    )
    return ChatAnthropic(api_key=api_key, model_name=model)


def mistral_chat(api_key=None, model="mistral-small"):
    api_key = api_key or fetch_api_token(
        ["MISTRAL_API_KEY"], ["~/.config/mistralai.token"]
    )
    return ChatMistralAI(api_key=api_key, model_name=model)
