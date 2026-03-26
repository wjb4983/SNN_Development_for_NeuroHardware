FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY snn_bench ./snn_bench
COPY scripts ./scripts
COPY .env.example ./.env.example

RUN pip install --upgrade pip \
    && pip install -e .

ENTRYPOINT ["python", "-m", "snn_bench.scripts.train"]
CMD ["--ticker", "AAPL", "--timeframe", "1D", "--epochs", "5", "--batch-size", "32", "--lr", "0.001", "--out-dir", "artifacts"]
