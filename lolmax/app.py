from dataclasses import dataclass

from flask import Flask, Response, request
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from lolmax import models

app = Flask(__name__)


@app.route("/stream", methods=["POST"])
def stream():
    client_prompt = ChatPrompt.extract(request.json)

    def generate():
        for chunk in client_prompt.client.stream(client_prompt.messages):
            print(f"chunk: {chunk}")
            yield chunk.content

    return Response(generate(), mimetype="text/plain")


@app.route("/invoke")
def invoke():
    client_prompt = ChatPrompt.extract(request.json)
    result = client_prompt.client.invoke(client_prompt.messages)
    return Response(result.content, mimetype="text/plain")


@dataclass
class ChatPrompt:
    model: str
    role: str
    messages: list[str]

    @property
    def client(self):
        client_factory = initialize_models().get(self.model)
        return client_factory()

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
            model=d.get("model", "gpt4"),
            role=d.get("role", "default"),
            messages=d.get("messages"),
        )


def initialize_models():
    return {
        "openai": models.openai_chat,
        "gemini": models.gemini_chat,
        "anthropic": models.anthropic_chat,
        "mistral": models.mistral_chat,
        "default": models.openai_chat,
    }


def main():
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
