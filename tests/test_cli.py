"""Unit tests for cli.py. All LLM calls, storage, and stdin are mocked."""

import random
from pathlib import Path

import pytest

import cli
import interviewer
import storage
from models import Analysis, InterviewPlan, InterviewStatus, SentimentLabel

FAKE_ANALYSIS = Analysis(
    overall_sentiment=SentimentLabel.POSITIVE,
    sentiment_score=0.6,
    keywords=["notion", "productivity"],
    key_points=["Uses Notion", "Likes calendar blocking"],
)

FAKE_PLAN = InterviewPlan(angles=["their first experience", "how it's changed over time"])


@pytest.fixture(autouse=True)
def isolated_transcripts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point storage.TRANSCRIPTS_DIR at a temp directory so no test touches the real transcripts/."""
    transcripts_dir = tmp_path / "transcripts"
    monkeypatch.setattr(storage, "TRANSCRIPTS_DIR", transcripts_dir)
    return transcripts_dir


@pytest.fixture(autouse=True)
def mock_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace all interviewer functions with deterministic fakes. No API key needed."""
    monkeypatch.setattr(interviewer, "generate_plan", lambda topic, max_questions: FAKE_PLAN)
    monkeypatch.setattr(
        interviewer,
        "generate_next_question",
        lambda topic, history, question_number, max_questions, plan: f"Question {question_number}?",
    )
    monkeypatch.setattr(interviewer, "generate_summary", lambda topic, transcript: "A short summary.")
    monkeypatch.setattr(interviewer, "analyze", lambda transcript: FAKE_ANALYSIS)


def feed_input(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> None:
    """Make input() return each of responses in order, one per call."""
    responses_iter = iter(responses)
    monkeypatch.setattr("builtins.input", lambda *args: next(responses_iter))


def test_run_interview_asks_the_chosen_number_of_questions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(random, "randint", lambda a, b: 3)
    feed_input(monkeypatch, ["productivity tools", "answer 1", "answer 2", "answer 3"])

    cli.run_interview()

    out = capsys.readouterr().out
    assert "Q1: Question 1?" in out
    assert "Q2: Question 2?" in out
    assert "Q3: Question 3?" in out
    assert "Q4:" not in out


def test_run_interview_respects_a_five_question_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(random, "randint", lambda a, b: 5)
    feed_input(monkeypatch, ["topic"] + [f"answer {i}" for i in range(1, 6)])

    cli.run_interview()

    out = capsys.readouterr().out
    assert "Q5:" in out
    assert "Q6:" not in out


def test_run_interview_prints_summary_analysis_and_engagement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(random, "randint", lambda a, b: 3)
    feed_input(
        monkeypatch, ["productivity tools", "answer one", "answer two words", "three word answer"]
    )

    cli.run_interview()

    out = capsys.readouterr().out
    assert "--- Summary ---" in out
    assert "A short summary." in out
    assert "--- Analysis ---" in out
    assert "Sentiment: positive (+0.60)" in out
    assert "Keywords: notion, productivity" in out
    assert "- Uses Notion" in out
    assert "--- Engagement ---" in out
    assert "Average word count:" in out
    assert "Average response time:" in out
    assert "Saved to" in out


def test_run_interview_saves_a_completed_interview(
    monkeypatch: pytest.MonkeyPatch, isolated_transcripts_dir: Path
) -> None:
    monkeypatch.setattr(random, "randint", lambda a, b: 3)
    feed_input(monkeypatch, ["productivity tools", "answer one", "answer two", "answer three"])

    cli.run_interview()

    saved_files = list(isolated_transcripts_dir.glob("*.json"))
    assert len(saved_files) == 1

    interview = storage.load_interview(saved_files[0])
    assert interview.topic == "productivity tools"
    assert interview.status == InterviewStatus.COMPLETED
    assert interview.completed_at is not None
    assert interview.plan == FAKE_PLAN
    assert len(interview.questions) == 3
    assert [q.answer.text for q in interview.questions] == ["answer one", "answer two", "answer three"]
    assert interview.summary == "A short summary."
    assert interview.analysis == FAKE_ANALYSIS
    assert interview.engagement is not None
    assert len(interview.engagement.per_question) == 3


def test_run_interview_passes_growing_history_to_generate_next_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random, "randint", lambda a, b: 3)
    feed_input(monkeypatch, ["productivity tools", "first answer", "second answer", "third answer"])

    seen_histories: list[str] = []

    def fake_generate_next_question(
        topic: str, history: str, question_number: int, max_questions: int, plan: InterviewPlan
    ) -> str:
        seen_histories.append(history)
        return f"Question {question_number}?"

    monkeypatch.setattr(interviewer, "generate_next_question", fake_generate_next_question)

    cli.run_interview()

    assert seen_histories[0] == ""
    assert "first answer" in seen_histories[1]
    assert "first answer" in seen_histories[2] and "second answer" in seen_histories[2]


def test_run_interview_asked_at_precedes_answered_at_for_every_question(
    monkeypatch: pytest.MonkeyPatch, isolated_transcripts_dir: Path
) -> None:
    monkeypatch.setattr(random, "randint", lambda a, b: 3)
    feed_input(monkeypatch, ["productivity tools", "answer one", "answer two", "answer three"])

    cli.run_interview()

    interview = storage.load_interview(next(isolated_transcripts_dir.glob("*.json")))
    for question in interview.questions:
        assert question.answer is not None
        assert question.answer.answered_at >= question.asked_at
