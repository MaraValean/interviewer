"""Unit tests for transcript.py. Pure Python — no mocking needed, no API key."""

from datetime import datetime

from models import Answer, Interview, Question
from transcript import format_transcript

T0 = datetime(2026, 1, 1, 12, 0, 0)


def make_answered_question(question_id: int, question_text: str, answer_text: str) -> Question:
    """Build a Question with an Answer already attached."""
    return Question(
        id=question_id,
        text=question_text,
        asked_at=T0,
        answer=Answer(question_id=question_id, text=answer_text, answered_at=T0),
    )


def test_format_transcript_with_no_questions() -> None:
    interview = Interview(topic="test")
    assert format_transcript(interview) == ""


def test_format_transcript_skips_unanswered_questions() -> None:
    unanswered = Question(id=1, text="Q1?", asked_at=T0)
    interview = Interview(topic="test", questions=[unanswered])
    assert format_transcript(interview) == ""


def test_format_transcript_single_answered_question() -> None:
    question = make_answered_question(1, "What tools do you use?", "Mostly Notion.")
    interview = Interview(topic="test", questions=[question])
    assert format_transcript(interview) == "Q: What tools do you use?\nA: Mostly Notion."


def test_format_transcript_joins_multiple_questions_with_blank_line() -> None:
    q1 = make_answered_question(1, "Q1?", "A1.")
    q2 = make_answered_question(2, "Q2?", "A2.")
    interview = Interview(topic="test", questions=[q1, q2])
    assert format_transcript(interview) == "Q: Q1?\nA: A1.\n\nQ: Q2?\nA: A2."


def test_format_transcript_ignores_unanswered_question_mixed_in() -> None:
    answered = make_answered_question(1, "Q1?", "A1.")
    unanswered = Question(id=2, text="Q2?", asked_at=T0)
    interview = Interview(topic="test", questions=[answered, unanswered])
    assert format_transcript(interview) == "Q: Q1?\nA: A1."
