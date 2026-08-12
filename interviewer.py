"""LLM interaction and response parsing for the AI Interviewer application."""

import json

from groq import Groq
from groq.types.chat import ChatCompletion

from models import Analysis

MODEL = "openai/gpt-oss-20b"
MAX_TOKENS = 1024

# Groq's strict json_schema response format requires additionalProperties:false
# on the schema; Analysis.model_json_schema() doesn't set that, so it's patched
# in here rather than in models.py, keeping this a call-parameter concern only.
_ANALYSIS_JSON_SCHEMA = Analysis.model_json_schema()
_ANALYSIS_JSON_SCHEMA["additionalProperties"] = False
ANALYSIS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "Analysis", "schema": _ANALYSIS_JSON_SCHEMA, "strict": True},
}

QUESTION_SYSTEM_PROMPT = """You are a skilled qualitative research interviewer. 
Your job is to interview one person about a given topic through a natural, 
adaptive conversation.

Guidelines:

- Ask ONE open-ended question at a time. Never ask multiple questions at once.
- Each question should build on what the person has already said, digging into 
  specifics, motivations, or examples they've raised.
- Stay on the topic. Do not drift into unrelated areas.
- Be warm and conversational, not robotic or survey-like.
- Never repeat a question that has already been asked.
- Do not summarize, comment on, or praise their answers — just ask the next question.

Output ONLY the question text, with no preamble, numbering, or quotation marks."""

QUESTION_USER_PROMPT = """Topic: {topic}

Conversation so far:
{history}

This will be question number {question_number} of a {max_questions}-question interview.
Write the next interview question."""

SUMMARY_SYSTEM_PROMPT = """You are a research analyst. Given an interview 
transcript, write a brief, neutral summary for a researcher.

Cover, in 3–5 short sentences:

- The main themes the person raised.
- Their overall sentiment toward the topic.
- The most notable or specific points they made.

Be faithful to what they actually said. Do not invent details or add opinions. 
Write in plain prose, no bullet points."""

SUMMARY_USER_PROMPT = """Topic: {topic}

Transcript:
{transcript}

Write the summary."""

ANALYSIS_SYSTEM_PROMPT = """You are a research analyst analyzing a qualitative
interview transcript.

Identify the main themes, the participant's overall sentiment, and the most
important insights from the interview.

Return ONLY valid JSON matching the expected Analysis structure.
Do not include markdown, explanations, or any text outside the JSON."""

ANALYSIS_USER_PROMPT = """Transcript:
{transcript}

Analyze the interview and return the requested JSON."""


_client: Groq | None = None


def _get_client() -> Groq:
    """Return a cached Groq client, creating it on first use."""
    global _client
    if _client is None:
        from config import settings

        _client = Groq(api_key=settings.groq_api_key.get_secret_value())
    return _client


def generate_next_question(
    topic: str,
    history: str,
    question_number: int,
    max_questions: int,
) -> str:
    """Ask the LLM for the next interview question, given the conversation so far."""
    user_prompt = QUESTION_USER_PROMPT.format(
        topic=topic,
        history=history,
        question_number=question_number,
        max_questions=max_questions,
    )
    response = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _extract_text(response)


def generate_summary(
    topic: str,
    transcript: str,
) -> str:
    """Ask the LLM for a brief narrative summary of the interview transcript."""
    user_prompt = SUMMARY_USER_PROMPT.format(topic=topic, transcript=transcript)
    response = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _extract_text(response)


def analyze(
    transcript: str,
) -> Analysis:
    """Ask the LLM to analyze the transcript and return a validated Analysis."""
    user_prompt = ANALYSIS_USER_PROMPT.format(transcript=transcript)
    response = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ANALYSIS_RESPONSE_FORMAT,
    )
    json_text = _strip_code_fences(_extract_text(response))

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Analysis response was not valid JSON: {exc}\nResponse: {json_text}") from exc

    return Analysis.model_validate(data)


def _extract_text(response: ChatCompletion) -> str:
    """Extract the text content from a Groq chat completion response."""
    return response.choices[0].message.content.strip()


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence (e.g. ```json ... ```), if present."""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
