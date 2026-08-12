"""LLM interaction and response parsing for the AI Interviewer application."""

import json

from groq import BadRequestError, Groq
from groq.types.chat import ChatCompletion, ChatCompletionMessageParam

from models import Analysis, InterviewPlan

MODEL = "openai/gpt-oss-20b"
MAX_TOKENS = 1024

# openai/gpt-oss-20b occasionally leaks its reasoning trace into the response
# instead of clean JSON when a strict json_schema response_format is used,
# which Groq surfaces as a BadRequestError with this code. It's transient —
# the same request typically succeeds immediately on retry — so structured
# calls get one retry via _create_structured() instead of failing outright.
_STRUCTURED_OUTPUT_PARSE_ERROR_CODE = "output_parse_failed"
_STRUCTURED_OUTPUT_MAX_ATTEMPTS = 2

# Groq's strict json_schema response format requires additionalProperties:false
# on the schema; Analysis.model_json_schema() doesn't set that, so it's patched
# in here rather than in models.py, keeping this a call-parameter concern only.
_ANALYSIS_JSON_SCHEMA = Analysis.model_json_schema()
_ANALYSIS_JSON_SCHEMA["additionalProperties"] = False
ANALYSIS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "Analysis", "schema": _ANALYSIS_JSON_SCHEMA, "strict": True},
}

_PLAN_JSON_SCHEMA = InterviewPlan.model_json_schema()
_PLAN_JSON_SCHEMA["additionalProperties"] = False
PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "InterviewPlan", "schema": _PLAN_JSON_SCHEMA, "strict": True},
}

PLAN_SYSTEM_PROMPT = """You are a skilled qualitative research interviewer preparing for an interview.

Given a topic, sketch a short list of distinct thematic angles worth exploring —
not literal questions, just brief phrases naming a facet of the topic (e.g.
"their first hands-on experience", "how it's changed their daily routine",
"concerns or skepticism they have", "what they'd want to see improve").

The angles should be genuinely different from each other and cover a range of
facets: concrete experiences, motivations, challenges, emotional reactions, and
outlook. Treat this as a loose guide for the interview, not a rigid script —
the interviewer may skip, reorder, or combine angles based on how the
conversation actually goes.

Return ONLY valid JSON matching the expected InterviewPlan structure."""

PLAN_USER_PROMPT = """Topic: {topic}

Sketch {max_questions} distinct thematic angles worth exploring in an interview about this topic."""

QUESTION_SYSTEM_PROMPT = """You are a skilled qualitative research interviewer.
Your job is to interview one person about a given topic through a natural,
adaptive conversation.

Guidelines:

- Ask ONE open-ended question at a time. Never ask multiple questions at once.
- Ask a single, focused question — do not stack multiple questions together
  with "and" or "or" inside one sentence.
- Each question should build on what the person has already said, digging into
  specifics, motivations, or examples they've raised.
- You will be given a loose interview plan (a list of angles worth exploring).
  Use it as a guide for what to explore next, but adapt to the actual
  conversation — skip an angle if it's already been covered, and don't ask
  about it verbatim if the person already addressed it in passing.
- Stay on the topic. Do not drift into unrelated areas.
- Be warm and conversational, not robotic or survey-like.
- Never repeat a question that has already been asked.
- Do not summarize, comment on, or praise their answers — just ask the next question.

Output ONLY the question text, with no preamble, numbering, or quotation marks."""

QUESTION_USER_PROMPT = """Topic: {topic}

Interview plan (loose guide — skip anything already covered):
{plan}

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

Refer to the interviewee using they/them pronouns, since their gender is not stated.
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

Refer to the participant using they/them pronouns, since their gender is not stated.

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


def _create_structured(
    messages: list[ChatCompletionMessageParam],
    response_format: dict,
) -> ChatCompletion:
    """Call chat.completions.create with a strict json_schema response_format.

    Retries once if Groq reports a structured-output parse failure — this is a
    transient quirk of the reasoning model, not a defect in the request (the
    same input reliably succeeds on a second attempt).
    """
    last_error: BadRequestError | None = None
    for _ in range(_STRUCTURED_OUTPUT_MAX_ATTEMPTS):
        try:
            return _get_client().chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
                response_format=response_format,
            )
        except BadRequestError as exc:
            body_error = exc.body.get("error", {}) if isinstance(exc.body, dict) else {}
            if body_error.get("code") != _STRUCTURED_OUTPUT_PARSE_ERROR_CODE:
                raise
            last_error = exc
    assert last_error is not None
    raise last_error


def generate_plan(
    topic: str,
    max_questions: int,
) -> InterviewPlan:
    """Ask the LLM for a loose outline of thematic angles, before any questions are asked."""
    user_prompt = PLAN_USER_PROMPT.format(topic=topic, max_questions=max_questions)
    response = _create_structured(
        messages=[
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=PLAN_RESPONSE_FORMAT,
    )
    json_text = _strip_code_fences(_extract_text(response))

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Plan response was not valid JSON: {exc}\nResponse: {json_text}") from exc

    return InterviewPlan.model_validate(data)


def generate_next_question(
    topic: str,
    history: str,
    question_number: int,
    max_questions: int,
    plan: InterviewPlan,
) -> str:
    """Ask the LLM for the next interview question, given the plan and conversation so far."""
    user_prompt = QUESTION_USER_PROMPT.format(
        topic=topic,
        plan=_format_plan(plan),
        history=history,
        question_number=question_number,
        max_questions=max_questions,
    )
    response = _get_client().chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        reasoning_effort="low",
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
        reasoning_effort="low",
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
    response = _create_structured(
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


def _format_plan(plan: InterviewPlan) -> str:
    """Render an InterviewPlan's angles as a bullet list for prompt insertion."""
    return "\n".join(f"- {angle}" for angle in plan.angles)


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
