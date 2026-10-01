"""One live smoke test against the real Groq API.

Excluded by default (see the `live` marker in pyproject.toml) — this costs
real API quota and is subject to real network/model flakiness, unlike the
rest of the suite. Run explicitly with `poetry run pytest -m live`.

This only checks that the pipeline runs and returns the right shapes, not
what the model actually says — LLM output isn't deterministic, so asserting
on content here would be asserting on the wrong thing.
"""

import pytest

from core import interviewer
from core.models import Analysis, InterviewPlan

pytestmark = pytest.mark.live


def _has_groq_api_key() -> bool:
    """Check the same way config.Settings does — env var or .env file — not
    just os.environ, which misses .env-sourced values entirely.
    """
    from pydantic import ValidationError

    from core.config import Settings

    try:
        Settings()
        return True
    except ValidationError:
        return False


@pytest.mark.skipif(not _has_groq_api_key(), reason="requires a real GROQ_API_KEY")
def test_full_interviewer_pipeline_runs_against_the_real_api() -> None:
    topic = "productivity tools"
    max_questions = 3

    plan = interviewer.generate_plan(topic=topic, max_questions=max_questions)
    assert isinstance(plan, InterviewPlan)
    assert len(plan.angles) > 0

    question_1 = interviewer.generate_next_question(
        topic=topic, history="", question_number=1, max_questions=max_questions, plan=plan
    )
    assert isinstance(question_1, str)
    assert question_1.strip() != ""

    history = f"Q: {question_1}\nA: I mostly use a to-do list app and calendar blocking."
    question_2 = interviewer.generate_next_question(
        topic=topic, history=history, question_number=2, max_questions=max_questions, plan=plan
    )
    assert isinstance(question_2, str)
    assert question_2.strip() != ""

    transcript = f"{history}\n\nQ: {question_2}\nA: Mainly Notion, since it keeps everything in one place."

    summary = interviewer.generate_summary(topic=topic, transcript=transcript)
    assert isinstance(summary, str)
    assert summary.strip() != ""

    analysis = interviewer.analyze(transcript)
    assert isinstance(analysis, Analysis)
    assert -1.0 <= analysis.sentiment_score <= 1.0
    assert len(analysis.keywords) > 0
    assert len(analysis.key_points) > 0
