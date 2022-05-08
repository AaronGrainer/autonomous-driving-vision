# Initialize project
.PHONY: install
install:
	python -m pip install -e ".[dev]" --no-cache-dir
	pip install git+https://github.com/facebookresearch/detectron2.git#egg=detectron2


# Pre commit hooks
.PHONY: install-pre-commit
install-pre-commit:
	pre-commit install
	pre-commit autoupdate

.PHONY: run-pre-commit
run-pre-commit:
	pre-commit run --all-files


# Styling
.PHONY: style
style:
	black .
	isort .
	flake8


# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E "pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage