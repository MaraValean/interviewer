"""FastAPI application exposing the AI Interviewer over HTTP."""

import random
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import engagement
import interviewer
import storage
from models import Analysis, Answer, EngagementMetrics, Interview, InterviewStatus, Question

app = FastAPI(title="AI Interviewer")


@app.get("/")
def index() -> FileResponse:
    """Serve the browser UI, which talks to the JSON endpoints below via fetch()."""
    return FileResponse("static/index.html")


@dataclass
class _Session:
    """An in-progress interview plus the question count chosen for it.

    Bundled together (rather than two separate dicts keyed by the same id)
    so the two can never drift out of sync with each other.
    """

    interview: Interview
    max_questions: int


# In-memory store of interviews, keyed by id. Lives only in this process's
# memory: it starts empty on every server startup and is lost on restart.
# See the note at the bottom of this file on why that's fine for a demo but
# not for production, and what to use instead.
_sessions: dict[UUID, _Session] = {}


class StartInterviewRequest(BaseModel):
    """Request body for POST /interviews."""

    topic: str


class StartInterviewResponse(BaseModel):
    """Response body for POST /interviews."""

    interview_id: UUID
    question: str
    total_questions: int


class AnswerRequest(BaseModel):
    """Request body for POST /interviews/{id}/answer."""

    text: str


class AnswerResponse(BaseModel):
    """Response body for POST /interviews/{id}/answer.

    question is None exactly when done is True.
    """

    done: bool
    question: str | None = None
    total_questions: int


class SummaryResponse(BaseModel):
    """Response body for GET /interviews/{id}/summary."""

    summary: str
    analysis: Analysis
    engagement: EngagementMetrics


@app.post("/interviews")
def start_interview(request: StartInterviewRequest) -> StartInterviewResponse:
    """Start a new interview and return its id plus the first question."""
    max_questions = random.randint(3, 5)
    question_text = interviewer.generate_next_question(
        topic=request.topic, history="", question_number=1, max_questions=max_questions
    )

    # asked_at is stamped here, at the moment this question is about to be
    # handed back in the HTTP response — see the timing note below.
    interview = Interview(topic=request.topic)
    interview.questions.append(Question(id=1, text=question_text, asked_at=datetime.now()))

    _sessions[interview.id] = _Session(interview=interview, max_questions=max_questions)

    return StartInterviewResponse(
        interview_id=interview.id, question=question_text, total_questions=max_questions
    )


@app.post("/interviews/{interview_id}/answer")
def submit_answer(interview_id: UUID, request: AnswerRequest) -> AnswerResponse:
    """Record an answer to the current question, then return the next question or done=True."""
    session = _get_session(interview_id)
    interview = session.interview

    if interview.status == InterviewStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Interview is already completed.")

    current_question = interview.questions[-1]
    if current_question.answer is not None:
        raise HTTPException(status_code=400, detail="Current question has already been answered.")

    # answered_at is stamped here, at the moment this request's body has
    # arrived — see the timing note below.
    current_question.answer = Answer(
        question_id=current_question.id, text=request.text, answered_at=datetime.now()
    )

    if len(interview.questions) >= session.max_questions:
        _complete_interview(interview)
        return AnswerResponse(done=True, question=None, total_questions=session.max_questions)

    next_question_number = len(interview.questions) + 1
    question_text = interviewer.generate_next_question(
        topic=interview.topic,
        history=_format_transcript(interview),
        question_number=next_question_number,
        max_questions=session.max_questions,
    )
    interview.questions.append(
        Question(id=next_question_number, text=question_text, asked_at=datetime.now())
    )

    return AnswerResponse(
        done=False, question=question_text, total_questions=session.max_questions
    )


@app.get("/interviews/{interview_id}/summary")
def get_summary(interview_id: UUID) -> SummaryResponse:
    """Return the summary, analysis, and engagement stats for a completed interview."""
    interview = _get_session(interview_id).interview
    if interview.status != InterviewStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Interview is not completed yet.")

    assert interview.summary is not None
    assert interview.analysis is not None
    assert interview.engagement is not None
    return SummaryResponse(
        summary=interview.summary, analysis=interview.analysis, engagement=interview.engagement
    )


def _get_session(interview_id: UUID) -> _Session:
    """Look up a session by interview id, or raise 404."""
    session = _sessions.get(interview_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Interview not found.")
    return session


def _complete_interview(interview: Interview) -> None:
    """Mark an interview completed, generate summary/analysis/engagement, and save it."""
    interview.status = InterviewStatus.COMPLETED
    interview.completed_at = datetime.now()

    transcript = _format_transcript(interview)
    interview.summary = interviewer.generate_summary(topic=interview.topic, transcript=transcript)
    interview.analysis = interviewer.analyze(transcript)
    interview.engagement = engagement.compute_engagement(interview)

    storage.save_interview(interview)


def _format_transcript(interview: Interview) -> str:
    """Render the answered questions so far as a plain-text Q/A transcript."""
    pairs = [
        f"Q: {question.text}\nA: {question.answer.text}"
        for question in interview.questions
        if question.answer is not None
    ]
    return "\n\n".join(pairs)


# --- Timing note ---
# response_time_seconds (computed by engagement.compute_engagement from
# asked_at/answered_at) means something different here than in cli.py.
# In the CLI, asked_at and answered_at bracket a single blocking input()
# call inside one process, so the gap is close to the user's actual typing
# time. Here, asked_at is stamped when POST /interviews or POST .../answer
# returns a question, and answered_at is stamped on a later, separate
# POST .../answer request — so the gap is wall-clock time across two HTTP
# round trips, including network latency and however long the client sits
# on the question before submitting. Same field, same formula, genuinely
# different thing being measured.