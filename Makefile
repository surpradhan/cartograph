.PHONY: install run check test lint clean

install:
	uv venv && . .venv/bin/activate && uv sync

run:
	. .venv/bin/activate && python app.py

check:
	. .venv/bin/activate && python scripts/check.py

test:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/ -v

lint:
	. .venv/bin/activate && ruff check src/ app.py

clean:
	rm -rf .venv __pycache__ src/__pycache__ history.db
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
