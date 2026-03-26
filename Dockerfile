FROM continuumio/miniconda3:24.7.1-0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY environment.yml requirements.txt pyproject.toml README.md ./
COPY snn_bench ./snn_bench
COPY scripts ./scripts
COPY .env.example ./.env.example

RUN mkdir -p /etc/Massive \
    && conda env create -f environment.yml \
    && conda clean -afy

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "snnbench"]
CMD ["python", "-m", "snn_bench.scripts.train", "--ticker", "AAPL", "--timeframe", "1D", "--epochs", "5", "--batch-size", "32", "--lr", "0.001", "--out-dir", "artifacts"]
