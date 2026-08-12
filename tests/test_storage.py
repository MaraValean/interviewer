"""Unit tests for storage.py. All tests write into a temp directory, never
into the project's real transcripts/.
"""

import re
from datetime import datetime
from pathlib import Path

import pytest

import storage
from models import (
    Analysis,
    Answer,
    EngagementMetrics,
    Interview,
    InterviewStatus,
    Question,
    QuestionMetrics,
    SentimentLabel,
    Summary,
)


@pytest.fixture(autouse=True)
def isolated_transcripts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point storage.TRANSCRIPTS_DIR at a temp directory for every test."""
    transcripts_dir = tmp_path / "transcripts"
    monkeypatch.setattr(storage, "TRANSCRIPTS_DIR", transcripts_dir)
    return transcripts_dir


def make_interview(topic: str = "productivity tools") -> Interview:
    """Build a minimal valid Interview: just a topic, everything else defaulted."""
    return Interview(topic=topic)


def make_completed_interview() -> Interview:
    """Build an Interview with transcript, summary, analysis, and engagement populated."""
    question = Question(
        id=1,
        text="What tools do you use?",
        asked_at=datetime.now(),
        answer=Answer(question_id=1, text="Notion.", answered_at=datetime.now()),
    )
    return Interview(
        topic="productivity tools",
        status=InterviewStatus.COMPLETED,
        questions=[question],
        completed_at=datetime.now(),
        summary=Summary(
            overview="They rely on Notion.",
            key_points=["Uses Notion"],
            themes=["tooling"],
            overall_sentiment=SentimentLabel.POSITIVE,
            generated_at=datetime.now(),
        ),
        analysis=Analysis(
            overall_sentiment=SentimentLabel.POSITIVE,
            sentiment_score=0.5,
            keywords=["notion"],
            key_points=["Uses Notion"],
        ),
        engagement=EngagementMetrics(
            per_question=[QuestionMetrics(question_id=1, word_count=2, response_time_seconds=5.0)],
            average_word_count=2.0,
            average_response_time_seconds=5.0,
        ),
    )


def test_save_interview_creates_transcripts_dir(isolated_transcripts_dir: Path) -> None:
    assert not isolated_transcripts_dir.exists()

    storage.save_interview(make_interview())

    assert isolated_transcripts_dir.is_dir()


def test_save_interview_returns_path_inside_transcripts_dir(isolated_transcripts_dir: Path) -> None:
    path = storage.save_interview(make_interview())

    assert path.parent == isolated_transcripts_dir


def test_save_interview_names_file_with_timestamp_and_id() -> None:
    interview = make_interview()

    path = storage.save_interview(interview)

    assert path.suffix == ".json"
    assert re.match(r"^\d{8}_\d{6}_" + re.escape(str(interview.id)) + r"\.json$", path.name)


def test_save_interview_writes_model_dump_json_content() -> None:
    interview = make_completed_interview()

    path = storage.save_interview(interview)

    assert path.read_text() == interview.model_dump_json(indent=2)


def test_load_interview_round_trips_a_completed_interview() -> None:
    interview = make_completed_interview()

    path = storage.save_interview(interview)
    loaded = storage.load_interview(path)

    assert loaded == interview


def test_load_interview_round_trips_a_minimal_interview() -> None:
    interview = make_interview()

    path = storage.save_interview(interview)
    loaded = storage.load_interview(path)

    assert loaded == interview


def test_two_interviews_do_not_collide() -> None:
    path_a = storage.save_interview(make_interview(topic="a"))
    path_b = storage.save_interview(make_interview(topic="b"))

    assert path_a != path_b
    assert storage.load_interview(path_a).topic == "a"
    assert storage.load_interview(path_b).topic == "b"
