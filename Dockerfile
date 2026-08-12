FROM python:3.12-slim

# Install straight into the system Python environment — the container itself
# is the isolation boundary, so a nested virtualenv would be redundant.
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false

WORKDIR /app

# Install dependencies first, so this layer is cached unless pyproject.toml
# or poetry.lock actually change. README.md is included because pyproject.toml
# references it as the package readme.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Now copy the application itself.
COPY core ./core
COPY cli ./cli
COPY web ./web

# Run as a non-root user.
RUN useradd --create-home appuser \
    && mkdir -p /app/transcripts \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Runs the web interface by default. GROQ_API_KEY must be supplied at
# `docker run` time (e.g. --env-file .env) — it is never baked into the image.
# To run the terminal interface instead:
#   docker run -it --env-file .env <image> python -m cli.cli
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
