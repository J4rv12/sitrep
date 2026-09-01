.PHONY: setup run test lint

PY := .venv/bin/python

setup:
	uv venv --python 3.12
	uv pip install --python $(PY) -r requirements-dev.txt

run:
	$(PY) -m uvicorn app.main:app --reload --port 8000

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
