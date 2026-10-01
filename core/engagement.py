"""Pure-Python computation of engagement metrics. No LLM calls."""

from core.models import EngagementMetrics, Interview, QuestionMetrics


def compute_engagement(interview: Interview) -> EngagementMetrics:
    """Compute word-count and response-time engagement metrics for an interview.

    Only answered questions are included. If there are none, returns zeroed
    averages instead of dividing by zero.
    """
    per_question = [
        QuestionMetrics(
            question_id=question.id,
            word_count=len(question.answer.text.split()),
            response_time_seconds=(question.answer.answered_at - question.asked_at).total_seconds(),
        )
        for question in interview.questions
        if question.answer is not None
    ]

    if not per_question:
        return EngagementMetrics(
            per_question=[], average_word_count=0.0, average_response_time_seconds=0.0
        )

    average_word_count = sum(m.word_count for m in per_question) / len(per_question)
    average_response_time_seconds = sum(m.response_time_seconds for m in per_question) / len(
        per_question
    )

    return EngagementMetrics(
        per_question=per_question,
        average_word_count=average_word_count,
        average_response_time_seconds=average_response_time_seconds,
    )
