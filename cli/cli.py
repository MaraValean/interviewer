"""Terminal interface for the AI Interviewer application."""

import random
from datetime import datetime

from core import engagement, interviewer, storage
from core.models import Answer, Interview, InterviewStatus, Question
from core.transcript import format_transcript


def run_interview() -> None:
    """Run an interactive interview end-to-end: ask, collect answers, summarize, save."""
    topic = input("What topic would you like to be interviewed about? ").strip()
    max_questions = random.randint(3, 5)
    plan = interviewer.generate_plan(topic=topic, max_questions=max_questions)
    interview = Interview(topic=topic, plan=plan)

    for question_number in range(1, max_questions + 1):
        history = format_transcript(interview)
        question_text = interviewer.generate_next_question(
            topic=topic,
            history=history,
            question_number=question_number,
            max_questions=max_questions,
            plan=plan,
        )

        asked_at = datetime.now()
        print(f"\nQ{question_number}: {question_text}")
        answer_text = input("> ").strip()
        answered_at = datetime.now()

        interview.questions.append(
            Question(
                id=question_number,
                text=question_text,
                asked_at=asked_at,
                answer=Answer(question_id=question_number, text=answer_text, answered_at=answered_at),
            )
        )

    interview.status = InterviewStatus.COMPLETED
    interview.completed_at = datetime.now()

    transcript = format_transcript(interview)
    interview.summary = interviewer.generate_summary(topic=topic, transcript=transcript)
    interview.analysis = interviewer.analyze(transcript)
    interview.engagement = engagement.compute_engagement(interview)

    _print_results(interview)
    path = storage.save_interview(interview)
    print(f"\nSaved to {path}")


def _print_results(interview: Interview) -> None:
    """Print the summary, analysis, and engagement stats to the terminal."""
    assert interview.analysis is not None
    assert interview.engagement is not None

    print("\n--- Summary ---")
    print(interview.summary)

    print("\n--- Analysis ---")
    print(
        f"Sentiment: {interview.analysis.overall_sentiment.value} "
        f"({interview.analysis.sentiment_score:+.2f})"
    )
    print(f"Keywords: {', '.join(interview.analysis.keywords)}")
    print("Key points:")
    for point in interview.analysis.key_points:
        print(f"  - {point}")

    print("\n--- Engagement ---")
    print(f"Average word count: {interview.engagement.average_word_count:.1f}")
    print(f"Average response time: {interview.engagement.average_response_time_seconds:.1f}s")


if __name__ == "__main__":
    run_interview()
