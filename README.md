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
poetry run python -m cli.cli
```

**Browser:**

```bash
poetry run uvicorn web.main:app --reload
```

Then open http://127.0.0.1:8000/.

Either way, the finished interview is saved as a timestamped JSON file in
`transcripts/`.

## Running it with Docker

```bash
docker build -t ai-interviewer .
docker run --rm -p 8000:8000 --env-file .env ai-interviewer
```

Then open http://127.0.0.1:8000/. The image runs the web interface by
default; to run the terminal interface in a container instead:

```bash
docker run --rm -it --env-file .env ai-interviewer python -m cli.cli
```

Transcripts are written inside the container's filesystem and are lost when
the container is removed unless you mount a volume:

```bash
docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/transcripts:/app/transcripts" ai-interviewer
```

## Running the tests

```bash
poetry run pytest
```

No API key is required to run the tests — every LLM call is mocked.

## Project structure

| Path | Purpose |
|---|---|
| `core/models.py` | Pydantic data models |
| `core/interviewer.py` | All LLM interaction (prompts, Groq calls, response parsing) |
| `core/engagement.py` | Local word-count/response-time stats — no LLM calls |
| `core/transcript.py` | Shared plain-text transcript formatting |
| `core/storage.py` | Save/load interviews as JSON |
| `core/config.py` | Environment-based settings (`GROQ_API_KEY`) |
| `cli/cli.py` | Terminal interface |
| `web/main.py`, `web/schemas.py`, `web/static/index.html` | FastAPI app, its request/response schemas, and the browser UI |

`core/` has no dependency on `cli/` or `web/` — both interfaces are built on
top of it, not the other way around. See `CLAUDE.md` for the full set of
project conventions.
