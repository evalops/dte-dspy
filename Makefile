PY=.venv/bin/python
VENV?=.venv
VERIFIER_A?=llama3.2:latest
VERIFIER_B?=llama3:8b
REFEREE?=llama3.1:8b
GAMMA?=0.7
DATASET?=test

.PHONY: setup install run run-quick run-sweep run-ablation test test-verbose fmt clean help

help:
	@echo "DTE: Disagreement-Triggered Escalation Framework"
	@echo ""
	@echo "Available commands:"
	@echo "  setup         - Create virtual environment and install dependencies"
	@echo "  install       - Install package in development mode"
	@echo "  run           - Run single evaluation"
	@echo "  run-quick     - Run quick evaluation (10 claims)"
	@echo "  run-sweep     - Run gamma threshold sweep"
	@echo "  run-ablation  - Run full ablation study"
	@echo "  test          - Run tests"
	@echo "  test-verbose  - Run tests with verbose output"
	@echo "  fmt           - Format code with ruff and black"
	@echo "  lint          - Run linting checks"
	@echo "  clean         - Clean build artifacts"
	@echo ""
	@echo "Environment variables:"
	@echo "  VERIFIER_A    - Model for verifier A (default: $(VERIFIER_A))"
	@echo "  VERIFIER_B    - Model for verifier B (default: $(VERIFIER_B))"
	@echo "  REFEREE       - Model for referee (default: $(REFEREE))"
	@echo "  GAMMA         - Confidence threshold (default: $(GAMMA))"
	@echo "  DATASET       - Dataset to use (default: $(DATASET))"

setup:
	python -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install -r requirements.txt

install: setup
	. $(VENV)/bin/activate && pip install -e ".[dev]"

run:
	DTE_VERIFIER_A_MODEL=$(VERIFIER_A) DTE_VERIFIER_B_MODEL=$(VERIFIER_B) DTE_REFEREE_MODEL=$(REFEREE) DTE_GAMMA=$(GAMMA) $(PY) -m dte.main --dataset $(DATASET) --verbose

run-quick:
	DTE_VERIFIER_A_MODEL=$(VERIFIER_A) DTE_VERIFIER_B_MODEL=$(VERIFIER_B) DTE_REFEREE_MODEL=$(REFEREE) DTE_GAMMA=$(GAMMA) $(PY) -m dte.main --quick --dataset $(DATASET)

run-sweep:
	DTE_VERIFIER_A_MODEL=$(VERIFIER_A) DTE_VERIFIER_B_MODEL=$(VERIFIER_B) DTE_REFEREE_MODEL=$(REFEREE) $(PY) -m dte.main --gamma-sweep --dataset $(DATASET) --report

run-ablation:
	DTE_VERIFIER_A_MODEL=$(VERIFIER_A) DTE_VERIFIER_B_MODEL=$(VERIFIER_B) DTE_REFEREE_MODEL=$(REFEREE) $(PY) -m dte.main --ablation --dataset $(DATASET) --report

run-edge-cases:
	$(MAKE) run DATASET=edge_cases

run-controversial:
	$(MAKE) run DATASET=controversial

test:
	$(PY) -m pytest tests/ -v

test-verbose:
	@echo "Running tests with maximum verbosity..."
	$(PY) -m pytest tests/ -v -s --tb=long

test-coverage:
	$(PY) -m pytest tests/ --cov=dte --cov-report=html --cov-report=term

fmt:
	$(PY) -m ruff check --fix . || true
	$(PY) -m ruff format . || true
	$(PY) -m isort . || true

lint:
	$(PY) -m ruff check .
	$(PY) -m mypy dte/

check: lint test

demo:
	@echo "Running DTE demonstration..."
	@echo "1. Quick evaluation:"
	$(MAKE) run-quick
	@echo ""
	@echo "2. Edge cases:"
	$(MAKE) run-edge-cases DATASET=edge_cases
	@echo ""
	@echo "3. Gamma sweep:"
	$(MAKE) run-sweep DATASET=test

benchmark:
	@echo "Running comprehensive benchmark..."
	$(MAKE) run-ablation DATASET=test
	$(MAKE) run-ablation DATASET=edge_cases
	$(MAKE) run-ablation DATASET=controversial

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

clean-results:
	rm -rf results/

install-ollama:
	@echo "Installing Ollama and required models..."
	@echo "Please run the following commands:"
	@echo ""
	@echo "# Install Ollama"
	@echo "curl -fsSL https://ollama.ai/install.sh | sh"
	@echo ""
	@echo "# Pull required models"
	@echo "ollama pull $(VERIFIER_A)"
	@echo "ollama pull $(VERIFIER_B)"
	@echo "ollama pull $(REFEREE)"
	@echo ""
	@echo "# Start Ollama server"
	@echo "ollama serve"