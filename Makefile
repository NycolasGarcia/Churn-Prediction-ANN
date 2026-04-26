.PHONY: install lint format test run train-baseline train-mlp mlflow-ui clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

test:
	pytest tests/ -v

train-baseline:
	python -m churn.training.trainer --model baseline

train-mlp:
	python -m churn.training.trainer --model mlp

run:
	uvicorn churn.api.main:app --reload --port 8000

mlflow-ui:
	mlflow ui --port 5000

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
