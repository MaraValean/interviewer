"""Unit tests for engagement.py. Pure Python — no mocking needed, no API key."""

from datetime import datetime, timedelta

import engagement
from models import Answer, Interview, Question

T0 = datetime(2026, 1, 1, 12, 0, 0)


def make_answered_question(
    question_id: int, text: str, asked_at: datetime, response_delay_seconds: float
) -> Question:
    """Build a Question with an Answer submitted response_delay_seconds after it was asked."""
    return Question(
        id=question_id,
        text=text,
        asked_at=asked_at,
        answer=Answer(
            question_id=question_id,
            text=text,
            answered_at=asked_at + timedelta(seconds=response_delay_seconds),
        ),
    )


def test_compute_engagement_with_no_questions() -> None:
    interview = Interview(topic="test")

    result = engagement.compute_engagement(interview)

    assert result.per_question == []
    assert result.average_word_count == 0.0
    assert result.average_response_time_seconds == 0.0


def test_compute_engagement_skips_unanswered_questions() -> None:
    unanswered = Question(id=1, text="Q1?", asked_at=T0)
    interview = Interview(topic="test", questions=[unanswered])

    result = engagement.compute_engagement(interview)

    assert result.per_question == []
    assert result.average_word_count == 0.0
    assert result.average_response_time_seconds == 0.0


def test_compute_engagement_single_answered_question() -> None:
    question = make_answered_question(1, "four words here now", T0, response_delay_seconds=10.0)
    interview = Interview(topic="test", questions=[question])

    result = engagement.compute_engagement(interview)

    assert len(result.per_question) == 1
    assert result.per_question[0].question_id == 1
    assert result.per_question[0].word_count == 4
    assert result.per_question[0].response_time_seconds == 10.0
    assert result.average_word_count == 4.0
    assert result.average_response_time_seconds == 10.0


def test_compute_engagement_multiple_answered_questions_averages() -> None:
    q1 = make_answered_question(1, "four words here now", T0, response_delay_seconds=10.0)
    q2 = make_answered_question(2, "two words", T0 + timedelta(seconds=20), response_delay_seconds=6.0)
    interview = Interview(topic="test", questions=[q1, q2])

    result = engagement.compute_engagement(interview)

    assert [m.word_count for m in result.per_question] == [4, 2]
    assert [m.response_time_seconds for m in result.per_question] == [10.0, 6.0]
    assert result.average_word_count == 3.0
    assert result.average_response_time_seconds == 8.0


def test_compute_engagement_ignores_unanswered_question_mixed_in() -> None:
    answered = make_answered_question(1, "two words", T0, response_delay_seconds=5.0)
    unanswered = Question(id=2, text="Q2?", asked_at=T0 + timedelta(seconds=30))
    interview = Interview(topic="test", questions=[answered, unanswered])

    result = engagement.compute_engagement(interview)

    assert len(result.per_question) == 1
    assert result.per_question[0].question_id == 1
    assert result.average_word_count == 2.0
    assert result.average_response_time_seconds == 5.0


def test_compute_engagement_preserves_question_order() -> None:
    q1 = make_answered_question(1, "a", T0, response_delay_seconds=1.0)
    q2 = make_answered_question(2, "b", T0 + timedelta(seconds=5), response_delay_seconds=2.0)
    q3 = make_answered_question(3, "c", T0 + timedelta(seconds=10), response_delay_seconds=3.0)
    interview = Interview(topic="test", questions=[q1, q2, q3])

    result = engagement.compute_engagement(interview)

    assert [m.question_id for m in result.per_question] == [1, 2, 3]
