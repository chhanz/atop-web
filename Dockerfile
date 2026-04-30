FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ATOP_LOG_DIR=/app/logs

WORKDIR /app

COPY pyproject.toml README.md ./
COPY atop_web/ ./atop_web/

RUN pip install --no-cache-dir -e ".[dev,bedrock]"

EXPOSE 8000

CMD ["uvicorn", "atop_web.main:app", "--host", "0.0.0.0", "--port", "8000"]
