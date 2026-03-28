PYTHON ?= python
PIP ?= pip
IMAGE ?= snn-bench:latest
API_KEY_FILE ?= /etc/Massive/api-key

.PHONY: setup lint unit-test cache-data smoke-run train-run experiment-run template-run template-run-snn docker-build docker-build-clean docker-cache docker-smoke docker-train

setup:
	timeout 180s $(PIP) install -e .

lint:
	timeout 120s $(PYTHON) -m compileall snn_bench tests

unit-test:
	timeout 120s $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

cache-data:
	timeout 900s $(PYTHON) -m snn_bench.scripts.cache_market_data --ticker AAPL --timeframe 1D --stock-years 5 --option-years 2

smoke-run:
	timeout 120s $(PYTHON) -m snn_bench.scripts.smoke_pipeline --ticker AAPL --timeframe 1D

train-run:
	timeout 120s $(PYTHON) -m snn_bench.scripts.train --ticker AAPL --timeframe 1D --epochs 5 --batch-size 32 --lr 0.001 --out-dir artifacts


experiment-run:
	timeout 1200s ./scripts/run_experiments.sh snn_bench/configs/experiments/aapl_model_sweep.yaml

template-run:
	timeout 120s $(PYTHON) -m src.quant_template.cli --config configs/template/experiments/ann_baseline.yaml

template-run-snn:
	timeout 120s $(PYTHON) -m src.quant_template.cli --config configs/template/experiments/snn_proxy.yaml

docker-build:
	timeout 2400s docker build -t $(IMAGE) .

docker-build-clean:
	timeout 2400s docker build --no-cache -t $(IMAGE) .

docker-cache:
	timeout 1200s docker run --rm \
	  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
	  -v $(PWD)/src/data:/app/src/data \
	  -v $(API_KEY_FILE):/etc/Massive/api-key:ro \
	  $(IMAGE) python -m snn_bench.scripts.cache_market_data --ticker AAPL --timeframe 1D --stock-years 5 --option-years 2

docker-smoke:
	timeout 300s docker run --rm \
	  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
	  -v $(PWD)/src/data:/app/src/data \
	  -v $(API_KEY_FILE):/etc/Massive/api-key:ro \
	  $(IMAGE) python -m snn_bench.scripts.smoke_pipeline --ticker AAPL --timeframe 1D

docker-train:
	timeout 600s docker run --rm \
	  -e MASSIVE_API_KEY_FILE=/etc/Massive/api-key \
	  -v $(PWD)/src/data:/app/src/data \
	  -v $(PWD)/artifacts:/app/artifacts \
	  -v $(API_KEY_FILE):/etc/Massive/api-key:ro \
	  $(IMAGE) python -m snn_bench.scripts.train --ticker AAPL --timeframe 1D --epochs 5 --batch-size 32 --lr 0.001 --out-dir artifacts
