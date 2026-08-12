# AI Interviewer

A small app that conducts a short, AI-powered interview on a topic you choose,
then summarizes and analyzes your answers. Inspired by [Anthropic's
Interviewer](https://www.anthropic.com/news/anthropic-interviewer).

Pick a topic (e.g. "AI in the workplace", "productivity tools", "gardening")
and the app:

1. Sketches a loose plan of thematic angles worth exploring.
2. Asks 3–5 adaptive, open-ended questions — each one aware of the plan and
   everything you've said so far, so it can dig deeper or skip ahead rather
   than working through a fixed script.
3. Collects your answers interactively, in the terminal or a browser.
4. Generates a brief narrative summary, plus a sentiment/keyword analysis and
   engagement stats (average word count and response time per answer).
5. Saves the full transcript, summary, and analysis as a JSON file.

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/)
- A [Groq](https://console.groq.com/) API key (free tier, no card required)

## Setup

```bash
poetry install
cp .env.example .env
```

Then edit `.env` and add your key:

```
GROQ_API_KEY=gsk_your_key_here
```

## Running it

**Terminal:**

```bash
poetry run python cli.py
```

**Browser:**

```bash
poetry run uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/.

Either way, the finished interview is saved as a timestamped JSON file in
`transcripts/`.

## Running the tests

```bash
poetry run pytest
```

No API key is required to run the tests — every LLM call is mocked.

## Project structure

| File | Purpose |
|---|---|
| `models.py` | Pydantic data models |
| `interviewer.py` | All LLM interaction (prompts, Groq calls, response parsing) |
| `engagement.py` | Local word-count/response-time stats — no LLM calls |
| `transcript.py` | Shared plain-text transcript formatting |
| `storage.py` | Save/load interviews as JSON |
| `cli.py` | Terminal interface |
| `main.py` + `static/index.html` | FastAPI app and browser UI |
| `config.py` | Environment-based settings (`GROQ_API_KEY`) |

See `CLAUDE.md` for the full set of project conventions.
