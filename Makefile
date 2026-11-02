.PHONY: install run debug clean lint lint-strict
ARG = index

install:
	uv sync

run:
	uv run python -m src $(ARG)

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
