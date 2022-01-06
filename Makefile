.PHONY: help
help:
	@echo "Commands:"
	@echo "install            : installs requirements."
	@echo "venv               : sets up virtual environment for development."
	@echo "clean              : cleans all unnecessary files."

# Installation
.PHONY: install
install:
	pip install -r requirements.txt

# Set up virtual environment
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python -m pip install --upgrade pip && \
	make install

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