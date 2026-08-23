# Contributing

Issues and pull requests are welcome.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,tokenizers]'
ruff check .
pytest
```

Please include tests for behavioural changes. In particular, changes to conversation boundaries,
token budgeting, streaming or authentication should include a regression test.

## Design principles

- Safe defaults over convenient public exposure.
- Explicit errors over silent loss of conversation history.
- Provider-neutral OpenAI compatibility where practical.
- No complete transcripts in persistent state.
- Small, reviewable dependencies.
- Honest documentation about the lossy nature of summarisation.

Do not include model weights, API keys, private conversations, generated state databases or local
network addresses in a contribution.
