from dataclasses import dataclass

from flask import Flask, Response, request

from lolmax import models

app = Flask(__name__)


@app.route("/stream")
def stream():
    client_prompt = extract_prompt_from_request()

    def generate():
        for chunk in client_prompt.client.stream(client_prompt.prompt):
            yield chunk.content

    return Response(generate(), mimetype="text/plain")


@app.route("/invoke")
def invoke():
    client_prompt = extract_prompt_from_request()
    result = client_prompt.client.invoke(client_prompt.prompt)
    return Response(result.content, mimetype="text/plain")


@dataclass
class ClientPrompt:
    model: str
    prompt: str

    @property
    def client(self):
        client_factory = initialize_models().get(self.model)
        return client_factory()


def initialize_models():
    return {
        "openai": models.openai_chat,
        "gemini": models.gemini_chat,
        "anthropic": models.anthropic_chat,
        "mistral": models.mistral_chat,
    }


def extract_prompt_from_request() -> ClientPrompt:
    return ClientPrompt(
        model=request.args.get("model", "gpt4"),
        prompt=request.args.get("prompt"),
    )


def main():
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
