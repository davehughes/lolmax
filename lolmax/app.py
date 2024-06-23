from dataclasses import dataclass
import json

from flask import Flask, Response, request
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pydantic.v1.error_wrappers

from lolmax import models

app = Flask(__name__)


# curl -X POST localhost:8000 \
#   -H "Accept: text/plain" \
#   -H "Content-Type: application/json" \
#   --date '{"model": "perplexity", "messages": [{"role": "user", "content": "hey, what's up?"}]}'
@app.route("/chat", methods=["POST"])
def chat():
    prompt = ChatPrompt.extract(request.json)
    mimetype = request.accept_mimetypes.best_match([
        "application/jsonlines+json",
        "application/json",
        "text/plain",
    ])
    if mimetype == "text/plain":
        return Response(prompt.stream_text(), mimetype="text/plain")
    else:
        # default, but should implies any of: ["application/jsonlines+json", "application/json"]
        return Response(prompt.stream_ndjson(),
                        mimetype="application/jsonlines+json")


@app.route("/invoke", methods=["POST"])
def invoke():
    prompt = ChatPrompt.extract(request.json)
    result = prompt.invoke()
    mimetype = request.accept_mimetypes.best_match([
        "application/json",
        "text/plain",
    ])
    if mimetype == "text/plain":
        return Response(result["content"], mimetype="text/plain")
    else:
        return Response(json.dumps(result), mimetype="application/json")


@dataclass
class ChatPrompt:
    model: str
    effects: list[str]
    messages: list[str]

    @property
    def client(self):
        client_factory = initialize_models().get(self.model)
        return client_factory()

    def invoke(self):
        result = self.client.invoke(self.messages)
        return {
            "id": result.id,
            "content": result.content,
        }

    def stream_objects(self):
        try:
            for chunk in self.client.stream(self.messages):
                yield {
                    "id": chunk.id,
                    "content": chunk.content,
                }
        except pydantic.v1.error_wrappers.ValidationError as e:
            yield {
                "id":
                "error",
                "content":
                f"An error occurred while generating a response:\n{json.dumps(e.errors())}",
            }
        except Exception as e:
            __import__('ipdb').set_trace()
            yield {
                "content":
                f"An error occurred while generating a response:\n{e.message}",
                "id": "error",
            }

    def stream_ndjson(self):
        for object in self.stream_objects():
            yield f"{json.dumps(object)}\n"

    def stream_text(self):
        for object in self.stream_objects():
            yield object["content"]

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
