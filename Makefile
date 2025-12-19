# Define the name of the virtual environment directory
VENV := venv

# Detect OS to adjust commands (Windows vs Mac/Linux)
ifeq ($(OS),Windows_NT)
    PYTHON := python
    PIP := $(VENV)/Scripts/pip
    UVICORN := $(VENV)/Scripts/uvicorn
    VENV_ACTIVATE := $(VENV)/Scripts/activate
else
    PYTHON := python3
    PIP := $(VENV)/bin/pip
    UVICORN := $(VENV)/bin/uvicorn
    VENV_ACTIVATE := $(VENV)/bin/activate
endif

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make venv     - Create virtual environment"
	@echo "  make install  - Install dependencies into venv"
	@echo "  make dev      - Run development server (reload enabled)"
	@echo "  make run      - Run production server"
	@echo "  make clean    - Remove venv and cache files"

# 1. Create Virtual Environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)
	@echo "✅ Virtual environment created at ./$(VENV)"

# 2. Install Dependencies
.PHONY: install
install: venv
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

# 3. Run Dev Server
.PHONY: dev
dev:
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

# 4. Run Prod Server
.PHONY: run
run:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000

# 5. Clean Up
.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	@echo "🧹 Cleaned up venv and cache"