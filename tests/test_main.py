"""Unit tests for main.py. All LLM calls and storage are mocked."""

import random
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import interviewer
import main
import storage
from models import Analysis, Answer, Interview, InterviewPlan, InterviewStatus, Question, SentimentLabel

FAKE_ANALYSIS = Analysis(
    overall_sentiment=SentimentLabel.POSITIVE,
    sentiment_score=0.6,
    keywords=["notion", "productivity"],
    key_points=["Uses Notion", "Likes calendar blocking"],
)

FAKE_PLAN = InterviewPlan(angles=["their first experience", "how it's changed over time"])


@pytest.fixture(autouse=True)
def clear_sessions() -> Iterator[None]:
    """Reset main._sessions before and after every test, since it's module-level state."""
    main._sessions.clear()
    yield
    main._sessions.clear()


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


@pytest.fixture(autouse=True)
def fixed_question_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix the randomly chosen question count to 3, for deterministic tests."""
    monkeypatch.setattr(random, "randint", lambda a, b: 3)


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


def test_index_serves_the_web_ui(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>AI Interviewer</title>" in response.text
    assert '"/interviews"' in response.text


def test_start_interview_returns_id_and_first_question(client: TestClient) -> None:
    response = client.post("/interviews", json={"topic": "productivity tools"})

    assert response.status_code == 200
    body = response.json()
    assert "interview_id" in body
    assert body["question"] == "Question 1?"


def test_start_interview_creates_a_session_with_one_asked_question(client: TestClient) -> None:
    response = client.post("/interviews", json={"topic": "productivity tools"})
    interview_id = response.json()["interview_id"]

    session = main._sessions[UUID(interview_id)]
    assert session.interview.topic == "productivity tools"
    assert session.max_questions == 3
    assert session.interview.plan == FAKE_PLAN
    assert len(session.interview.questions) == 1
    assert session.interview.questions[0].answer is None


def test_submit_answer_returns_next_question(client: TestClient) -> None:
    start = client.post("/interviews", json={"topic": "productivity tools"})
    interview_id = start.json()["interview_id"]

    response = client.post(f"/interviews/{interview_id}/answer", json={"text": "answer one"})

    assert response.status_code == 200
    assert response.json() == {"done": False, "question": "Question 2?", "total_questions": 3}


def test_submit_answer_on_final_question_returns_done_with_no_question(client: TestClient) -> None:
    start = client.post("/interviews", json={"topic": "productivity tools"})
    interview_id = start.json()["interview_id"]

    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer one"})
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer two"})
    response = client.post(f"/interviews/{interview_id}/answer", json={"text": "answer three"})

    assert response.status_code == 200
    assert response.json() == {"done": True, "question": None, "total_questions": 3}


def test_submit_answer_unknown_interview_id_returns_404(client: TestClient) -> None:
    response = client.post(
        "/interviews/00000000-0000-0000-0000-000000000000/answer", json={"text": "hi"}
    )

    assert response.status_code == 404


def test_submit_answer_after_completion_returns_400(client: TestClient) -> None:
    start = client.post("/interviews", json={"topic": "productivity tools"})
    interview_id = start.json()["interview_id"]
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer one"})
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer two"})
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer three"})

    response = client.post(f"/interviews/{interview_id}/answer", json={"text": "one more"})

    assert response.status_code == 400


def test_submit_answer_on_already_answered_question_returns_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "already answered" guard protects a state the normal request flow can't
    reach on its own, so it's tested by reaching into _sessions directly rather
    than through the API sequence.
    """
    interview = Interview(topic="test")
    interview.questions.append(
        Question(
            id=1,
            text="Q1?",
            asked_at=datetime.now(),
            answer=Answer(question_id=1, text="already answered", answered_at=datetime.now()),
        )
    )
    interview_id = interview.id
    main._sessions[interview_id] = main._Session(interview=interview, max_questions=3)

    client = TestClient(main.app)
    response = client.post(f"/interviews/{interview_id}/answer", json={"text": "again"})

    assert response.status_code == 400


def test_get_summary_before_completion_returns_409(client: TestClient) -> None:
    start = client.post("/interviews", json={"topic": "productivity tools"})
    interview_id = start.json()["interview_id"]

    response = client.get(f"/interviews/{interview_id}/summary")

    assert response.status_code == 409


def test_get_summary_unknown_interview_id_returns_404(client: TestClient) -> None:
    response = client.get("/interviews/00000000-0000-0000-0000-000000000000/summary")

    assert response.status_code == 404


def test_get_summary_after_completion(client: TestClient) -> None:
    start = client.post("/interviews", json={"topic": "productivity tools"})
    interview_id = start.json()["interview_id"]
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer one"})
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer two"})
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer three"})

    response = client.get(f"/interviews/{interview_id}/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "A short summary."
    assert body["analysis"]["overall_sentiment"] == "positive"
    assert body["analysis"]["keywords"] == ["notion", "productivity"]
    assert len(body["engagement"]["per_question"]) == 3
    assert body["engagement"]["average_word_count"] == 2.0


def test_full_interview_flow_saves_a_completed_interview(
    client: TestClient, isolated_transcripts_dir: Path
) -> None:
    start = client.post("/interviews", json={"topic": "productivity tools"})
    interview_id = start.json()["interview_id"]
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer one"})
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer two"})
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer three"})

    saved_files = list(isolated_transcripts_dir.glob("*.json"))
    assert len(saved_files) == 1

    interview = storage.load_interview(saved_files[0])
    assert interview.status == InterviewStatus.COMPLETED
    assert interview.completed_at is not None
    assert [q.answer.text for q in interview.questions] == ["answer one", "answer two", "answer three"]
    assert interview.summary == "A short summary."
    assert interview.analysis == FAKE_ANALYSIS


def test_history_passed_to_generate_next_question_grows(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_histories: list[str] = []

    def fake_generate_next_question(
        topic: str, history: str, question_number: int, max_questions: int, plan: InterviewPlan
    ) -> str:
        seen_histories.append(history)
        return f"Question {question_number}?"

    monkeypatch.setattr(interviewer, "generate_next_question", fake_generate_next_question)

    start = client.post("/interviews", json={"topic": "productivity tools"})
    interview_id = start.json()["interview_id"]
    client.post(f"/interviews/{interview_id}/answer", json={"text": "first answer"})
    client.post(f"/interviews/{interview_id}/answer", json={"text": "second answer"})

    assert seen_histories[0] == ""
    assert "first answer" in seen_histories[1]
    assert "first answer" in seen_histories[2] and "second answer" in seen_histories[2]


def test_answered_at_is_not_before_asked_at(client: TestClient) -> None:
    start = client.post("/interviews", json={"topic": "productivity tools"})
    interview_id = start.json()["interview_id"]
    client.post(f"/interviews/{interview_id}/answer", json={"text": "answer one"})

    session = main._sessions[UUID(interview_id)]
    first_question = session.interview.questions[0]
    assert first_question.answer is not None
    assert first_question.answer.answered_at >= first_question.asked_at
