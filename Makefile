.PHONY: install test format lint clean help

help:
	@echo "Available commands:"
	@echo "  make install   - Install package with dev dependencies"
	@echo "  make test      - Run tests with coverage"
	@echo "  make format    - Format code with Black"
	@echo "  make lint      - Lint code with Ruff"
	@echo "  make clean     - Remove build artifacts"

install:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src/salary_sacrifice --cov-report=term-missing

format:
	black .
	ruff check --fix .

lint:
	black --check .
	ruff check .

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage coverage.xml htmlcov/
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .ruff_cache/
