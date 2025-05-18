.PHONY: install test lint format clean

# Activate virtual environment
ACTIVATE = . .venv/bin/activate

# Install dependencies
install:
	uv sync --all-extras

# Run tests
test:
	$(ACTIVATE) && hatch test

# Run linter
lint:
	$(ACTIVATE) && hatch fmt --linter --check

# Format code
format:
	$(ACTIVATE) && hatch fmt --formatter

# Clean up
clean:
	rm -rf .venv
	rm -rf build
	rm -rf dist
	rm -rf .pytest_cache
	rm -rf .ruff_cache

# Show help
help:
	@echo "Available commands:"
	@echo "  make install    Install dependencies"
	@echo "  make test       Run tests"
	@echo "  make lint       Run linter"
	@echo "  make format     Format code"
	@echo "  make clean      Clean up"
	@echo "  make help       Show this help"
