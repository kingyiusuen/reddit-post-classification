.PHONY: help
help:
	@echo "Commands:"
	@echo "install            : installs requirements."
	@echo "install-dev        : installs development requirements."
	@echo "install-test       : installs test requirements."
	@echo "venv               : sets up virtual environment for development."
	@echo "test               : runs all tests."
	@echo "test-non-training  : runs non-training tests."
	@echo "lint               : runs linting."
	@echo "clean              : cleans all unnecessary files."

# Installation
.PHONY: install
install:
	pip install -e . --no-cache-dir

.PHONY: install-dev
install-dev:
	pip install -e ".[dev]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate

.PHONY: install-test
install-test:
	pip install -e ".[test]" --no-cache-dir

# Set up virtual environment
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	make install-dev

# Linting
.PHONY: lint
lint:
	isort .
	black .
	flake8 .
	mypy .
	pydocstyle .

# Tests
.PHONY: test
test:
	pytest --cov reddit_post_classification --cov backend --cov-report html

.PHONY: test-non-training
test-non-training:
	pytest -m "not training"

# Cleaning
.PHONY: clean
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . -type f -name ".coverage*" -ls -delete
	rm -rf htmlcov
	rm -rf .mypy_cache