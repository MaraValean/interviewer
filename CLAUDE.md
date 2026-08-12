# AI Interviewer

A small app that conducts short, AI-powered interviews on a chosen topic,
then summarizes and analyzes the responses. Inspired by Anthropic's Interviewer.

## What it does

1. User picks a topic.
2. App asks 3–5 adaptive questions, where each next question is informed by prior answers.
3. Collects answers interactively.
4. Produces an AI summary including themes, sentiment, and key points.
5. Saves the full transcript, summary, and analysis as JSON.

## Stack & conventions

- Python 3.11+
- Poetry for dependency and environment management.
- Pydantic v2 for all application data structures.
- Do not pass untyped/raw dictionaries between modules.
- FastAPI for the web interface.
- A CLI entry point is also provided.
- Groq SDK for LLM calls.
- Type hints on every function.
- Docstrings on public functions, classes, and Pydantic models.
- Small, single-responsibility modules.
- Prefer clarity and maintainability over cleverness.

## Project structure

- `models.py` — Pydantic models (`Interview`, `Question`, `Answer`, `Summary`, `Analysis`, `QuestionMetrics`, `EngagementMetrics`)
- `interviewer.py` — all LLM interaction; prompt strings live here as named constants
- `engagement.py` — pure-Python computation of engagement metrics (word count and response time per answer, plus averages); no LLM calls
- `storage.py` — save/load interviews as JSON in `transcripts/`
- `cli.py` — terminal interface
- `main.py` — FastAPI application
- `config.py` — application settings using `pydantic-settings`; reads `GROQ_API_KEY` from the environment

## Rules

- Never hardcode secrets or API keys.
- Read `GROQ_API_KEY` from the environment through `config.py`.
- All prompts must be defined as named constants.
- Do not embed prompt strings directly inside business logic.
- Keep prompt definitions separate from business logic so prompts can be tuned independently.
- Ask before adding new dependencies.
- When a design decision is unclear, present the available options and wait for approval rather than guessing.
- Do not make unrelated changes to the codebase.
- Engagement stats (word count, response time) are computed locally in `engagement.py` and must never live in `interviewer.py`, which is LLM-only. The `Interview` model exposes them via a separate `engagement: EngagementMetrics | None` field, distinct from `analysis`.

## Interview business rules

- Each interview must contain between 3 and 5 questions.
- Questions must remain relevant to the chosen topic.
- Questions must be open-ended.
- Never repeat a question within the same interview.
- Each question after the first should consider the candidate's previous answers when appropriate.
- The interview should feel conversational and adaptive rather than like a fixed questionnaire.

## Development workflow

Before making significant changes:

1. Inspect the existing code and understand the relevant modules.
2. Explain the proposed approach briefly.
3. Ask for approval if the change involves a significant architectural or dependency decision.

When implementing changes:

- Follow the existing project structure and conventions.
- Keep changes focused on the requested task.
- Add or update tests when behavior changes.
- Run relevant tests after making changes.
- Do not modify unrelated files.

## Testing

- Use `pytest`.
- New business logic should have corresponding tests where practical.
- Tests must not require a real Groq API call unless explicitly requested.
- Mock or isolate LLM calls in unit tests.

## Git

- Do not commit secrets, `.env` files, or generated transcripts containing sensitive data.
- Keep commits focused and descriptive.