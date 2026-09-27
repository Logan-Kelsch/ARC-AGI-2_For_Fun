PYTHON ?= python3.12
VENV := .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
FRAMEWORK_DIR := vendor/ARC-AGI-3-Agents
FRAMEWORK_REPO := https://github.com/arcprize/ARC-AGI-3-Agents.git
GAME ?=
STEPS ?= 120
POLICY ?= null
SEED ?= 0
MODE ?= offline
RENDER ?=
KAGGLE := KAGGLE_API_TOKEN=$$(cat .kaggle/access_token) $(VENV)/bin/kaggle

.PHONY: help setup test list-games arcade-cache arcade-list evaluate-game play-local verify-local notebook submit status clean

help:
	@echo "make setup"
	@echo "make test"
	@echo "make list-games"
	@echo "make arcade-cache"
	@echo "make arcade-list"
	@echo "make evaluate-game GAME=ls20 POLICY=null SEED=0 STEPS=80"
	@echo "make play-local GAME=ls20 STEPS=120"
	@echo "make verify-local"
	@echo "make notebook"
	@echo "make submit"
	@echo "make status"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install "arc-agi>=0.9.6" "kaggle>=2.2" python-dotenv numpy pandas pyarrow nbformat pytest
	@if [ ! -d "$(FRAMEWORK_DIR)/.git" ]; then \
		mkdir -p vendor && git clone --depth 1 $(FRAMEWORK_REPO) $(FRAMEWORK_DIR); \
	else \
		git -C $(FRAMEWORK_DIR) pull --ff-only; \
	fi
	$(VENV_PY) scripts/slim_framework.py

test:
	PYTHONPATH=src $(VENV_PY) -m pytest -q

list-games:
	PYTHONPATH=src $(VENV_PY) scripts/play_local.py --list

arcade-cache:
	PYTHONPATH=src $(VENV_PY) scripts/cache_arcade.py

arcade-list:
	PYTHONPATH=src $(VENV_PY) scripts/cache_arcade.py --list

evaluate-game:
	@if [ -z "$(GAME)" ]; then echo "Set GAME, e.g. make evaluate-game GAME=ls20"; exit 1; fi
	PYTHONPATH=src $(VENV_PY) scripts/evaluate_arcade.py --game $(GAME) --policy $(POLICY) --seed $(SEED) --max-actions $(STEPS) --mode $(MODE) $(if $(RENDER),--render $(RENDER))

play-local:
	PYTHONPATH=src $(VENV_PY) scripts/play_local.py $(if $(GAME),--game $(GAME)) --max-steps $(STEPS)

verify-local:
	PYTHONPATH=src $(VENV_PY) scripts/play_local.py --game ls20,vc33 --max-steps 40

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
	rm -rf $(VENV) vendor environment_files recordings reference results .pytest_cache __pycache__ notebooks/submission.ipynb
