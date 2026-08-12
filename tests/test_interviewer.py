"""Unit tests for interviewer.py.

Every test replaces interviewer._get_client() with a mock, so no test ever
constructs a real Groq client or touches config.py — no API key, real or
fake, is required to run this suite.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import interviewer
from models import Analysis, SentimentLabel


@dataclass
class FakeMessage:
    """Stand-in for a Groq ChatCompletionMessage."""

    content: str


@dataclass
class FakeChoice:
    """Stand-in for a Groq Choice."""

    message: FakeMessage


@dataclass
class FakeChatCompletion:
    """Stand-in for a Groq ChatCompletion response."""

    choices: list[FakeChoice]


def fake_response(text: str) -> FakeChatCompletion:
    """Build a FakeChatCompletion whose single choice's message has the given text."""
    return FakeChatCompletion(choices=[FakeChoice(message=FakeMessage(content=text))])


@pytest.fixture
def mock_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace interviewer._get_client with a stub returning a MagicMock client."""
    client = MagicMock()
    monkeypatch.setattr(interviewer, "_get_client", lambda: client)
    return client


def test_generate_next_question_returns_stripped_text(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response(
        "  What draws you to this topic?  "
    )

    question = interviewer.generate_next_question(
        topic="AI in the workplace",
        history="Q1: What's your role?\nA1: I'm an engineer.",
        question_number=2,
        max_questions=4,
    )

    assert question == "What draws you to this topic?"


def test_generate_next_question_sends_expected_prompt(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response("next question")

    interviewer.generate_next_question(
        topic="AI in the workplace",
        history="Q1: ...\nA1: ...",
        question_number=2,
        max_questions=4,
    )

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["model"] == interviewer.MODEL
    assert kwargs["messages"] == [
        {"role": "system", "content": interviewer.QUESTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": interviewer.QUESTION_USER_PROMPT.format(
                topic="AI in the workplace",
                history="Q1: ...\nA1: ...",
                question_number=2,
                max_questions=4,
            ),
        },
    ]


def test_generate_next_question_with_empty_history(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response("What's your role?")

    question = interviewer.generate_next_question(
        topic="productivity tools", history="", question_number=1, max_questions=3
    )

    assert question == "What's your role?"
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["messages"][1] == {
        "role": "user",
        "content": interviewer.QUESTION_USER_PROMPT.format(
            topic="productivity tools", history="", question_number=1, max_questions=3
        ),
    }


def test_generate_summary_returns_stripped_text(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response(
        "  They felt mostly positive about AI tools.  "
    )

    summary = interviewer.generate_summary(
        topic="AI in the workplace", transcript="Q: ...\nA: ..."
    )

    assert summary == "They felt mostly positive about AI tools."


def test_generate_summary_sends_expected_prompt(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response("summary text")

    interviewer.generate_summary(topic="AI in the workplace", transcript="Q: ...\nA: ...")

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["messages"] == [
        {"role": "system", "content": interviewer.SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": interviewer.SUMMARY_USER_PROMPT.format(
                topic="AI in the workplace", transcript="Q: ...\nA: ..."
            ),
        },
    ]


def test_extract_text_strips_whitespace() -> None:
    assert interviewer._extract_text(fake_response("  hello  \n")) == "hello"


def test_strip_code_fences_no_fence_returns_unchanged() -> None:
    text = '{"a": 1}'
    assert interviewer._strip_code_fences(text) == text


def test_strip_code_fences_json_tagged_fence() -> None:
    fenced = '```json\n{"a": 1}\n```'
    assert interviewer._strip_code_fences(fenced) == '{"a": 1}'


def test_strip_code_fences_plain_fence() -> None:
    fenced = '```\n{"a": 1}\n```'
    assert interviewer._strip_code_fences(fenced) == '{"a": 1}'


def test_strip_code_fences_preserves_backticks_inside_json_string_values() -> None:
    fenced = '```json\n{"key_points": ["uses `pip install`"]}\n```'
    assert interviewer._strip_code_fences(fenced) == '{"key_points": ["uses `pip install`"]}'


VALID_ANALYSIS_JSON = (
    '{"overall_sentiment": "positive", "sentiment_score": 0.6, '
    '"keywords": ["ai", "productivity"], "key_points": ["enjoys automation"]}'
)


def test_analyze_success_parses_fenced_json(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response(
        f"```json\n{VALID_ANALYSIS_JSON}\n```"
    )

    result = interviewer.analyze("some transcript")

    assert isinstance(result, Analysis)
    assert result.overall_sentiment == SentimentLabel.POSITIVE
    assert result.sentiment_score == 0.6
    assert result.keywords == ["ai", "productivity"]
    assert result.key_points == ["enjoys automation"]


def test_analyze_success_parses_unfenced_json(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response(VALID_ANALYSIS_JSON)

    result = interviewer.analyze("some transcript")

    assert isinstance(result, Analysis)


def test_analyze_sends_expected_prompt(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response(VALID_ANALYSIS_JSON)

    interviewer.analyze("some transcript text")

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["messages"] == [
        {"role": "system", "content": interviewer.ANALYSIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": interviewer.ANALYSIS_USER_PROMPT.format(transcript="some transcript text"),
        },
    ]
    assert kwargs["response_format"] == interviewer.ANALYSIS_RESPONSE_FORMAT


def test_generate_next_question_does_not_request_structured_output(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response("a question")

    interviewer.generate_next_question(
        topic="t", history="", question_number=1, max_questions=3
    )

    _, kwargs = mock_client.chat.completions.create.call_args
    assert "response_format" not in kwargs


def test_analyze_raises_value_error_on_invalid_json(mock_client: MagicMock) -> None:
    mock_client.chat.completions.create.return_value = fake_response("not json at all")

    with pytest.raises(ValueError, match="not valid JSON"):
        interviewer.analyze("some transcript")


def test_analyze_raises_validation_error_on_bad_sentiment_label(mock_client: MagicMock) -> None:
    bad_json = '{"overall_sentiment": "ecstatic", "sentiment_score": 0.5, "keywords": [], "key_points": []}'
    mock_client.chat.completions.create.return_value = fake_response(bad_json)

    with pytest.raises(ValidationError):
        interviewer.analyze("some transcript")


def test_analyze_raises_validation_error_on_out_of_range_score(mock_client: MagicMock) -> None:
    bad_json = '{"overall_sentiment": "positive", "sentiment_score": 5.0, "keywords": [], "key_points": []}'
    mock_client.chat.completions.create.return_value = fake_response(bad_json)

    with pytest.raises(ValidationError):
        interviewer.analyze("some transcript")
