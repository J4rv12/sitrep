.PHONY: setup run test lint eval eval-record

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

# Offline and free: replays eval/recordings.jsonl. Replay never authenticates,
# so the secrets are placeholders and this runs on a clone with no .env.
eval:
	WEBHOOK_TOKEN=offline-replay-placeholder-not-a-secret ANTHROPIC_API_KEY=offline \
	TELEGRAM_BOT_TOKEN=offline TELEGRAM_CHAT_ID=offline $(PY) -m eval.run_eval

# Online and billed, about $$0.05: calls Claude with the key in .env.
eval-record:
	$(PY) -m eval.run_eval --record
