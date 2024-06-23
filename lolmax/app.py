from dataclasses import dataclass
import json

from flask import Flask, Response, request
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pydantic.v1.error_wrappers

from lolmax import models

app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    prompt = ChatPrompt.extract(request.json)
    return Response(prompt.stream(), mimetype="application/jsonlines+json")


@app.route("/invoke")
def invoke():
    client_prompt = ChatPrompt.extract(request.json)
    result = client_prompt.client.invoke(client_prompt.messages)
    return Response(result.content, mimetype="text/plain")


@dataclass
class ChatPrompt:
    model: str
    effects: list[str]
    messages: list[str]

    @property
    def client(self):
        client_factory = initialize_models().get(self.model)
        return client_factory()

    def stream(self):
        try:
            for chunk in self.client.stream(self.messages):
                serializable_chunk = {
                    "id": chunk.id,
                    "content": chunk.content,
                }
                yield f"{json.dumps(serializable_chunk)}\n"
        except pydantic.v1.error_wrappers.ValidationError as e:
            serializable_error = {
                "id":
                "error",
                "content":
                f"An error occurred while generating a response:\n{json.dumps(e.errors())}",
            }
            yield f"{json.dumps(serializable_error)}\n"
        except Exception as e:
            __import__('ipdb').set_trace()
            serializable_error = {
                "content":
                f"An error occurred while generating a response:\n{e.message}",
                "id": "error",
            }
            yield f"{json.dumps(serializable_error)}\n"

    @property
    def converted_messages(self):
        converted = []
        role_to_message_type = {
            'user': HumanMessage,
            'assistant': AIMessage,
            'system': SystemMessage,
        }
        for message in self.messages:
            message_type = role_to_message_type.get(message['role'])
            if not message_type:
                print(f"Unrecognized message type for message: {message}")
            converted.append(message_type(message['content']))

        return converted

    @staticmethod
    def extract(d):
        return ChatPrompt(
            model=d.get("model", "perplexity"),
            effects=d.get("effects", []),
            messages=d.get("messages"),
        )


def initialize_models():
    return {
        "openai": models.openai_chat,
        "gemini": models.gemini_chat,
        "anthropic": models.anthropic_chat,
        "mistral": models.mistral_chat,
        "perplexity": models.perplexity_chat,
        "default": models.openai_chat,
    }


def main():
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
