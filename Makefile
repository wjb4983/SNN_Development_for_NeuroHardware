PYTHON ?= python
PIP ?= pip
IMAGE ?= snn-bench:latest
API_KEY_FILE ?= /etc/Massive/api-key

.PHONY: setup lint unit-test smoke-run train-run docker-build docker-smoke docker-train

setup:
	timeout 180s $(PIP) install -e .

lint:
	timeout 120s $(PYTHON) -m compileall snn_bench tests

unit-test:
	timeout 120s $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

smoke-run:
	timeout 120s $(PYTHON) -m snn_bench.scripts.smoke_pipeline --ticker AAPL --timeframe 1D

train-run:
	timeout 120s $(PYTHON) -m snn_bench.scripts.train --ticker AAPL --timeframe 1D --epochs 5 --batch-size 32 --lr 0.001 --out-dir artifacts

docker-build:
	timeout 1800s docker build -t $(IMAGE) .

docker-smoke:
	timeout 300s docker run --rm \
	  --entrypoint python \
	  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
	  -v $(PWD)/src/data:/app/src/data \
	  -v $(API_KEY_FILE):/etc/Massive/api-key:ro \
	  $(IMAGE) -m snn_bench.scripts.smoke_pipeline --ticker AAPL --timeframe 1D

docker-train:
	timeout 600s docker run --rm \
	  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
	  -v $(PWD)/src/data:/app/src/data \
	  -v $(PWD)/artifacts:/app/artifacts \
	  -v $(API_KEY_FILE):/etc/Massive/api-key:ro \
	  $(IMAGE) --ticker AAPL --timeframe 1D --epochs 5 --batch-size 32 --lr 0.001 --out-dir artifacts
