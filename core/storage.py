"""Save and load Interview transcripts as JSON files in transcripts/."""

from datetime import datetime
from pathlib import Path

from core.models import Interview

TRANSCRIPTS_DIR = Path("transcripts")


def save_interview(interview: Interview) -> Path:
    """Save an Interview as a timestamped JSON file in transcripts/.

    Returns the path the interview was written to.
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = TRANSCRIPTS_DIR / f"{timestamp}_{interview.id}.json"
    path.write_text(interview.model_dump_json(indent=2))
    return path


def load_interview(path: Path) -> Interview:
    """Load an Interview from a JSON file previously written by save_interview."""
    return Interview.model_validate_json(path.read_text())
