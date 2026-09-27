PYTHON ?= python3.12
VENV := .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
DATA_DIR ?= data/ARC-AGI-2
SOLVER ?= primitive
SPLIT ?= evaluation
TASK ?=
LIMIT ?=
OUT ?= results/submission.json
KAGGLE := KAGGLE_API_TOKEN=$$(cat .kaggle/access_token) $(VENV)/bin/kaggle

.PHONY: help setup data test list-tasks inspect evaluate submission-local notebook submit status clean

help:
	@echo "make setup"
	@echo "make data"
	@echo "make test"
	@echo "make list-tasks SPLIT=evaluation"
	@echo "make inspect TASK=<task_id> SPLIT=evaluation"
	@echo "make evaluate SOLVER=primitive SPLIT=evaluation"
	@echo "make submission-local SOLVER=primitive SPLIT=evaluation OUT=results/submission.json"
	@echo "make notebook"
	@echo "make submit"
	@echo "make status"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e ".[dev]"

data:
	$(VENV_PY) scripts/fetch_public_data.py --destination $(DATA_DIR)

test:
	$(VENV_PY) -m pytest -q

list-tasks:
	$(VENV_PY) scripts/list_tasks.py --data-dir $(DATA_DIR) --split $(SPLIT)

inspect:
	@if [ -z "$(TASK)" ]; then echo "Set TASK=<task_id>"; exit 1; fi
	$(VENV_PY) scripts/inspect_task.py --data-dir $(DATA_DIR) --split $(SPLIT) --task $(TASK)

evaluate:
	$(VENV_PY) scripts/evaluate.py --data-dir $(DATA_DIR) --split $(SPLIT) --solver $(SOLVER) $(if $(LIMIT),--limit $(LIMIT))

submission-local:
	mkdir -p $$(dirname $(OUT))
	$(VENV_PY) scripts/build_local_submission.py --data-dir $(DATA_DIR) --split $(SPLIT) --solver $(SOLVER) --output $(OUT)

notebook:
	$(VENV_PY) scripts/build_submission_notebook.py

submit: notebook
	@if [ ! -s .kaggle/access_token ]; then echo "Missing .kaggle/access_token"; exit 1; fi
	@if grep -q REPLACE_WITH_YOUR_USERNAME notebooks/kernel-metadata.json; then echo "Edit notebooks/kernel-metadata.json with your Kaggle username first."; exit 1; fi
	$(KAGGLE) kernels push -p notebooks/

status:
	@if [ ! -s .kaggle/access_token ]; then echo "Missing .kaggle/access_token"; exit 1; fi
	@KERNEL_ID=$$($(VENV_PY) -c "import json; print(json.load(open('notebooks/kernel-metadata.json'))['id'])"); \
	$(KAGGLE) kernels status $$KERNEL_ID

clean:
	rm -rf $(VENV) results .pytest_cache __pycache__ notebooks/submission.ipynb
