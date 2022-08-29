# Initialize project
.PHONY: init
init:
	python -m pip install -e ".[dev]" --no-cache-dir
	git clone https://github.com/facebookresearch/detectron2.git
	python -m pip install -e detectron2
	rmdir /s detectron2

	wget https://github.com/Eromera/erfnet_pytorch/blob/master/trained_models/erfnet_encoder_pretrained.pth.tar -P ./checkpoint
	wget https://github.com/Eromera/erfnet_pytorch/blob/master/trained_models/erfnet_pretrained.pth -P ./checkpoint

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
clean:
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E "pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage