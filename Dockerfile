# CoreCoder API service image (referenced by docker-compose.yml `build: .`).
#
# corecoder's optional backends (jieba / fastembed / sqlite-vec / numpy) are all
# lazily imported and degrade gracefully, so a slim base stays valid: the API
# only needs the core package + the service-layer trio (fastapi/uvicorn/redis).

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY corecoder/ corecoder/
COPY config/ config/
COPY api/ api/

RUN pip install --no-cache-dir . fastapi uvicorn redis

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
