"""Render an Interview's answered questions as a plain-text Q/A transcript."""

from core.models import Interview


def format_transcript(interview: Interview) -> str:
    """Render the answered questions so far as a plain-text Q/A transcript."""
    pairs = [
        f"Q: {question.text}\nA: {question.answer.text}"
        for question in interview.questions
        if question.answer is not None
    ]
    return "\n\n".join(pairs)
