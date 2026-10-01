"""HTTP request/response schemas for main.py's endpoints.

These are wire contracts, not domain models — see models.py for the actual
domain objects. Some fields here (SummaryResponse.analysis/engagement) reuse
domain models directly, where the API shape happens to match the domain
shape; the rest exist because the API shape is deliberately thinner than (or
otherwise different from) any single domain object.
"""

from uuid import UUID

from pydantic import BaseModel

from core.models import Analysis, EngagementMetrics


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
