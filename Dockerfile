# syntax=docker/dockerfile:1
# Multi-stage build for the SentinelAI prediction API service.

FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-api.txt .
RUN pip install --no-cache-dir --user -r requirements-api.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
RUN useradd -m -u 1000 sentinel \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local /home/sentinel/.local
COPY --chown=sentinel:sentinel . .
USER sentinel
ENV PATH=/home/sentinel/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
