LoLMax
---
A simple server for running AI chats and other invocations through LangChain. Serves as the backend
for my [vim-ai-lolmax](https://github.com/davehughes/vim-ai-lolmax) plugin.

Quickstart
```
python -m venv .venv
source .venv/bin/activate
poetry install

# set up api tokens in files/environment (see lolmax/models.py for specifics)

# run the flask app
lolmax-app

# curl to send a basic prompt to a defined model
curl -X POST localhost:8000/chat \
  -H "Accept: text/plain" \
  -H "Content-Type: application/json" \
  --date '{"model": "perplexity", "messages": [{"role": "user", "content": "hey, what's up?"}]}'
```
