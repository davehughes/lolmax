LoLMax
---
A simple server for running AI chats and other invocations through LangChain. Serves as the backend
for my [vim-ai-lolmax](https://github.com/davehughes/vim-ai-lolmax) plugin.

Quickstart
```
python -m venv .venv
source .venv/bin/activate
poetry install

# set up api tokens in files/environment (see lolmax/config.py for specifics)

# run the flask app
lolmax-app

# curl to send a basic prompt to a defined model
curl -X POST localhost:8000/chat \
  -H "Accept: text/plain" \
  -H "Content-Type: application/json" \
  --date '{"model": "perplexity", "messages": [{"role": "user", "content": "hey, what's up?"}]}'

# fetch info on defined models/effects/etc.
curl -X GET localhost:8000/info | jq '.'
```

Sample configuration
---
TODO: XDG-friendly configuration. Currently, this is hard-coded to load config from `~/.config/lolmax`.
```
# ~/.config/lolmax/config.yml
---
models:
  gpt-4-default:
    description: "..."
    model: gpt-4
    api_key: openai

  claude:
    description: "..."
    model: claude-3-sonnet-20240229
    api_key: anthropic

  mistral:
    description: "..."
    model: mistral-small-latest
    model_provider: mistralai
    api_key: mistral

  pplx-llama3-large:
    model: llama-3-sonar-large-32k-chat
    model_class: langchain_community.chat_models.ChatPerplexity
    pplx_api_key: perplexity
```

Tokens are stored in a separate file and used to replace `api_key` references in the main config file:
```
# ~/.config/lolmax/tokens.yml
---
openai: <openai token>
anthropic: <anthropic token>
...
```
