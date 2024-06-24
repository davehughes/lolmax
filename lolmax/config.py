import copy
import importlib
import os

import yaml
from langchain.chat_models.base import init_chat_model


class Config:
    # TODO: XDG properly
    @classmethod
    def load_config(
        cls, config_path="lolmax/config.yml", tokens_path="lolmax/tokens.yml"
    ):
        with open(os.path.expanduser(os.path.join("~/.config", config_path)), "r") as f:
            config = yaml.safe_load(f)

        with open(os.path.expanduser(os.path.join("~/.config", tokens_path)), "r") as f:
            tokens = yaml.safe_load(f)

        return cls(config, tokens)

    def __init__(self, config, tokens):
        self.config = config
        self.tokens = tokens

    def info(self):
        models = []
        for name, model_config in self.config.get("models", {}).items():
            model_info = copy.deepcopy(model_config)
            model_info["name"] = name
            remove_keys = [k for k in model_info.keys() if k.endswith("api_key")]
            for k in remove_keys:
                model_info.pop(k)

            models.append(model_info)

        return {"models": models, "effects": self.config.get("effects", {})}

    def get_model(self, model_id):
        if model_id not in self.config.get("models", {}):
            raise Exception(f"Unrecognized model: {model_id}")

        model = copy.deepcopy(self.config.get("models", {}).get(model_id))

        # Replace API key names with values from the tokens file
        for key in model.keys():
            if key.endswith("api_key"):
                model[key] = self.tokens.get(model[key])

        if "description" in model:
            model.pop("description")

        if "model_class" in model:
            # HACK: fall back to 'manual' loading in cases where LangChain doesn't support providers
            path_segments = model.pop("model_class").split(".")
            module_name = ".".join(path_segments[:-1])
            class_name = path_segments[-1]
            module = importlib.import_module(module_name)
            Chat = getattr(module, class_name)
            return Chat(**model)
        else:
            return init_chat_model(**model)

    def get_effects(self, effect_ids):
        all_effects = self.config.get("effects", {})
        effect_configs = [
            all_effects.get(eid) for eid in effect_ids if all_effects.get(eid)
        ]
        return [copy.deepcopy(e) for e in effect_configs]


# def openai_chat(api_key=None, model="gpt-4"):
#     api_key = api_key or fetch_api_token(["OPENAI_API_KEY"], ["~/.config/openai.token"])
#     return ChatOpenAI(api_key=api_key, model=model)
#
#
# def gemini_chat(api_key: str | None = None, model: str = "gemini-pro"):
#     api_key = api_key or fetch_api_token(
#         ["GOOGLE_API_KEY"], ["~/.config/google_gemini.token"]
#     )
#     # HACK: this client seems to require passing api key by environment variable, so we mutate the environment
#     if api_key:
#         os.environ["GOOGLE_API_KEY"] = api_key
#
#     return ChatGoogleGenerativeAI(model=model)
#
#
# # Supported models: https://docs.perplexity.ai/docs/model-cards
# def perplexity_chat(
#     api_key: str | None = None, model: str = "llama-3-sonar-large-32k-chat"
# ):
#     api_key = api_key or fetch_api_token(
#         ["PPLX_API_KEY"], ["~/.config/perplexity.token"]
#     )
#     return ChatPerplexity(pplx_api_key=api_key, model=model)
#
#
# # Available models (as of 2024.05.29):
# # + claude-3-opus-20240229
# # + claude-3-sonnet-20240229
# # + claude-3-haiku-20240307
# def anthropic_chat(api_key=None, model="claude-3-sonnet-20240229"):
#     api_key = api_key or fetch_api_token(
#         ["ANTHROPIC_API_KEY"], ["~/.config/anthropic.token"]
#     )
#     return ChatAnthropic(api_key=api_key, model_name=model)
#
#
# def mistral_chat(api_key=None, model="mistral-small"):
#     api_key = api_key or fetch_api_token(
#         ["MISTRAL_API_KEY"], ["~/.config/mistralai.token"]
#     )
#     return ChatMistralAI(api_key=api_key, model_name=model)
