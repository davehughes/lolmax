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
