# AI CLI Client

> **Python CLI for OpenAI/OpenRouter-compatible APIs** · Rich terminal UI · prompt history · configurable endpoint/model
>
> Repository codename: `AI-CLI-assist-`.

AI CLI Client is a small Python command-line application for working with OpenAI-compatible chat APIs directly from a terminal. It is intentionally much smaller than the production platforms in this profile and is kept as a tooling/CLI example.

## Highlights

- installable Python package;
- `ai-cli` console entry point;
- OpenAI/OpenRouter-style endpoint support;
- interactive terminal UX with Rich and prompt-toolkit;
- environment-based configuration;
- lightweight dependency set suitable for local developer tooling.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Configure credentials through environment variables or a local environment file that is excluded from Git.

```env
OPENROUTER_API_KEY=...
AI_CLI_API_KEY=...
```

Run:

```bash
ai-cli
```

## Package structure

```text
pyproject.toml
src/
└── ai_cli/
```

## Stack

- Python 3.8+
- requests
- python-dotenv
- Rich
- prompt-toolkit
- yaspin
- tqdm

## Portfolio note

This repository represents developer tooling rather than a production SaaS/product backend. The larger pinned projects on this profile are better examples of backend architecture, databases, payments and production operations.
