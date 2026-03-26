PYTHON ?= python
PIP ?= pip

.PHONY: setup lint unit-test smoke-run train-run

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
