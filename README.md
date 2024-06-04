LoLMax
---
A half-baked simple server for running AI chats and other invocations through LangChain. Hoping someday this will be ok
enough to use as a backend for a plugin similar to [vim-ai](https://github.com/madox2/vim-ai).

Quickstart
```
python -m venv .venv
source .venv/bin/activate
poetry install

# set up api tokens in files/environment (see lolmax/models.py for specifics)

# run the flask app
lolmax-app

# curl to send a basic prompt to a defined model
curl -G localhost:8000/stream --data-urlencode "model=anthropic" --data-urlencode "prompt=what are some good things to
ask an AI chatbot?"
```
