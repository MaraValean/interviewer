"""Pydantic data models for the AI Interviewer application."""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class InterviewStatus(str, Enum):
    """Lifecycle state of an interview."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class SentimentLabel(str, Enum):
    """Coarse sentiment classification used by Analysis."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class Answer(BaseModel):
    """A user's response to a single interview question."""

    question_id: int = Field(description="ID of the Question this answer responds to.")
    text: str = Field(description="Raw text of the user's answer.")
    answered_at: datetime = Field(description="Timestamp when the answer was submitted.")


class Question(BaseModel):
    """A single interview question and, once answered, its paired Answer."""

    id: int = Field(description="1-based position of this question within the interview.")
    text: str = Field(description="The open-ended question text.")
    asked_at: datetime = Field(description="Timestamp when the question was generated.")
    answer: Answer | None = Field(
        default=None, description="The user's answer, or None if not yet answered."
    )


class InterviewPlan(BaseModel):
    """A loose outline of thematic angles to explore, generated once at interview start.

    Angles are short phrases naming a facet of the topic, not literal questions —
    generate_next_question treats this as a guide to adapt around, not a script.
    """

    angles: list[str] = Field(
        description="Short thematic directions to explore during the interview, in no particular order."
    )


class Analysis(BaseModel):
    """Bonus structured analysis of the interview: sentiment score and extracted keywords.

    Field names and types mirror the exact JSON shape expected from the LLM's
    structured output, so this model doubles as the parsing/validation boundary
    for that response.
    """

    overall_sentiment: SentimentLabel = Field(
        description="One of positive, neutral, negative, or mixed."
    )
    sentiment_score: float = Field(
        ge=-1.0, le=1.0, description="Sentiment polarity from -1.0 (negative) to 1.0 (positive)."
    )
    keywords: list[str] = Field(description="Keywords or key phrases extracted from the interview.")
    key_points: list[str] = Field(description="Notable points extracted from the interview.")


class QuestionMetrics(BaseModel):
    """Locally computed engagement stats for a single answered question."""

    question_id: int = Field(description="ID of the Question these metrics were computed for.")
    word_count: int = Field(description="Number of words in the answer text.")
    response_time_seconds: float = Field(
        description="Seconds elapsed between the question being asked and the answer being submitted."
    )


class EngagementMetrics(BaseModel):
    """Locally computed engagement statistics, derived from timestamps and answer text.

    Unlike Analysis, this model requires no LLM call — every field is computed
    directly from data already on the Interview's questions and answers.
    """

    per_question: list[QuestionMetrics] = Field(
        description="Word count and response time for each answered question, in order."
    )
    average_word_count: float = Field(description="Mean word count across all answers.")
    average_response_time_seconds: float = Field(
        description="Mean response time across all answered questions, in seconds."
    )


class Interview(BaseModel):
    """Root aggregate for a single interview: transcript, summary text, and analysis."""

    id: UUID = Field(
        default_factory=uuid4, description="Unique interview identifier; also the storage filename."
    )
    topic: str = Field(description="Topic chosen by the user at the start of the interview.")
    status: InterviewStatus = Field(
        default=InterviewStatus.IN_PROGRESS, description="Current lifecycle state."
    )
    plan: InterviewPlan | None = Field(
        default=None, description="Loose outline of angles generated at interview start."
    )
    questions: list[Question] = Field(
        default_factory=list,
        description="Questions asked so far, each carrying its own answer once answered.",
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp when the interview was started."
    )
    completed_at: datetime | None = Field(
        default=None, description="Timestamp when the interview was completed."
    )
    summary: str | None = Field(
        default=None, description="Narrative summary text, generated once the interview is completed."
    )
    analysis: Analysis | None = Field(
        default=None, description="Generated once the interview is completed."
    )
    engagement: EngagementMetrics | None = Field(
        default=None, description="Locally computed engagement stats; no LLM call required."
    )
