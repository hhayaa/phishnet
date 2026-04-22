.PHONY: help install dev test serve train docker-build

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install phishnet
	python -m pip install .

dev: ## Install dev mode
	python -m pip install -e ".[dev]"

test: ## Run tests
	python -m pytest tests/ -v --tb=short

serve: ## Start API server
	uvicorn phishnet.serving:app --host 0.0.0.0 --port 8000

train: ## Train model
	python scripts/train.py

docker-build: ## Build Docker image
	docker build -t phishnet:latest .
